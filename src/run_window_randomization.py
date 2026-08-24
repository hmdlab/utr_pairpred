"""Sliding-window randomization (in-silico mutagenesis) analysis for HRUs.

For each Highly Related UTR pair (HRU; relation score >= threshold), this script
randomizes a contiguous window (default 10% of the sequence length) of either the
5' UTR or the 3' UTR *independently* (while keeping the partner region unchanged),
re-embeds the mutated sequence with the RNA language model (RiNALMo / RNA-FM), and
recomputes the relation score (cosine similarity from the contrastive-learning
model ``PairPredCR``). By sliding the window across the sequence we obtain, for each
HRU, a profile of how much each region contributes to the predicted 5'-3' UTR
relationship, and we can locate the window whose randomization changes the relation
score the most.

The "relation score" is defined exactly as in the paper / ``run_cross_species_eval``:
the cosine similarity between the L2-normalized 5' and 3' UTR feature vectors
produced by ``PairPredCR``.

Example
-------
    poetry run python run_window_randomization.py \
        --cfg ../config/human_cl_cross_species_rinalmo.yaml \
        --model_path ../results/runs/cross_species_human_cl/best_model.pth \
        --seq_data ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
        --cv_results_dir ../results/runs/contrastive_learning_10fold_rinalmo_whole_ave_seed1 \
        --out_dir ../results/runs/window_randomization_human \
        --window_frac 0.1 --stride_frac 0.05 --n_trials 10
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from _model_dict import MODEL_DICT
from utils import create_total_df

RNA_BASES = np.array(["A", "C", "G", "U"])


def _argparse() -> argparse.Namespace:
    args = argparse.ArgumentParser(description=__doc__)
    # --- model / data ---
    args.add_argument("--cfg", required=True, type=str, help="path to config yaml (for model arch params)")
    args.add_argument(
        "--model_path",
        required=True,
        type=str,
        help="path to trained contrastive-learning model (.pth state_dict of PairPredCR)",
    )
    args.add_argument(
        "--seq_data",
        type=str,
        default=None,
        help="path to sequence csv (ENST_ID,GENE,5UTR,3UTR,...). Defaults to cfg.seq_data",
    )
    args.add_argument(
        "--emb_type",
        type=str,
        default="rinalmo",
        choices=["rinalmo", "rnafm"],
        help="RNA language model used for embedding (default: rinalmo, the paper's main model)",
    )
    args.add_argument(
        "--input_dim",
        type=int,
        default=None,
        help="override model.input_dim (1280 for RiNALMo, 640 for RNA-FM). Defaults to cfg.model.input_dim",
    )
    # --- HRU selection ---
    args.add_argument(
        "--cv_results_dir",
        type=str,
        default=None,
        help="dir of the k-fold CV results (contains <fold>/pred_results.pkl). "
        "Relation scores are read via utils.create_total_df, matching the notebook analysis.",
    )
    args.add_argument(
        "--hru_csv",
        type=str,
        default=None,
        help="alternative to --cv_results_dir: csv listing HRUs. Must contain an 'ENST_ID' column "
        "(and optionally a 'cos_sim'/'relation_score' column).",
    )
    args.add_argument("--kfold", type=int, default=10, help="number of CV folds for create_total_df")
    args.add_argument(
        "--relation_threshold",
        type=float,
        default=0.7,
        help="relation score threshold defining HRUs (paper uses 0.7-1.0 bin)",
    )
    args.add_argument(
        "--max_hrus",
        type=int,
        default=None,
        help="cap on the number of HRUs analyzed (sorted by relation score, highest first)",
    )
    # --- control group (for comparison with non-HRUs) ---
    args.add_argument(
        "--control_high",
        type=float,
        default=None,
        help="if set, also analyze a control group of pairs whose relation score is in "
        "[control_low, control_high) (e.g. 0.1) for comparison with HRUs",
    )
    args.add_argument("--control_low", type=float, default=0.0, help="lower bound of control relation score bin")
    args.add_argument("--n_control", type=int, default=None, help="number of control pairs to sample")
    # --- sliding window params ---
    args.add_argument("--window_frac", type=float, default=0.1, help="window length as a fraction of sequence length")
    args.add_argument(
        "--stride_frac",
        type=float,
        default=0.05,
        help="window stride as a fraction of sequence length (smaller = finer / more overlap)",
    )
    args.add_argument("--n_trials", type=int, default=10, help="number of independent randomizations per window")
    args.add_argument(
        "--randomize_mode",
        type=str,
        nargs="+",
        default=["uniform"],
        choices=["uniform", "shuffle", "dinucleotide"],
        help="randomization scheme(s). 'uniform' = random A/C/G/U (destroys composition "
        "+ structure); 'shuffle' = permute existing bases (preserves composition, "
        "destroys order/structure); 'dinucleotide' = permute while preserving exact "
        "dinucleotide counts (retains base-pairing propensity, so contrasting it with "
        "'shuffle' isolates the base-pairing contribution). Pass several (e.g. "
        "--randomize_mode uniform shuffle dinucleotide) to compare.",
    )
    args.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=["utr5", "utr3"],
        choices=["utr5", "utr3"],
        help="which region(s) to randomize, each independently (partner region kept intact)",
    )
    args.add_argument("--n_bins", type=int, default=10, help="number of relative-position bins for aggregation")
    # --- runtime ---
    args.add_argument("--batch_size", type=int, default=16, help="max sequences per RNA-LM forward pass")
    args.add_argument(
        "--token_budget",
        type=int,
        default=12000,
        help="max total nucleotides per forward pass; long sequences automatically use a smaller batch",
    )
    args.add_argument("--seed", type=int, default=0, help="random seed")
    args.add_argument("--out_dir", required=True, type=str, help="output directory")
    args.add_argument(
        "--no_plot",
        action="store_true",
        help="skip figure generation (CSV outputs are always written)",
    )
    return args.parse_args()


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
class Embedder:
    """RNA language-model embedder with batched inference.

    Reproduces the embedding procedures used to build the training inputs:
      * rinalmo: mean pooling over all token embeddings ("whole_ave").
      * rnafm:   the [CLS] token embedding (sequences > 1022 nt are segmented and
                 their segment-level [CLS] embeddings averaged, matching "average").

    Batched inference assumes all sequences in a call have the **same length**
    (true within a transcript+region, where only the window content changes), so no
    padding-mask handling is required.
    """

    MAX_SEG_LEN = 1022

    def __init__(self, emb_type: str, device: str):
        from multimolecule import RiNALMoModel, RnaFmModel, RnaTokenizer

        self.emb_type = emb_type
        self.device = device
        if emb_type == "rinalmo":
            self.model = RiNALMoModel.from_pretrained("multimolecule/rinalmo")
            self.tokenizer = RnaTokenizer.from_pretrained("multimolecule/rinalmo")
        elif emb_type == "rnafm":
            self.model = RnaFmModel.from_pretrained("multimolecule/rnafm")
            self.tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
        else:
            raise ValueError(f"Unknown emb_type: {emb_type}")
        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def _forward_pool(self, seqs: list) -> torch.Tensor:
        """Run one forward pass on equal-length sequences and pool per emb_type."""
        inputs = self.tokenizer(seqs, return_tensors="pt").to(self.device)
        if self.emb_type == "rinalmo":
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                out = self.model(**inputs)
            hidden = out["last_hidden_state"].float()  # (B, L+2, D)
            return hidden.mean(dim=1).detach().cpu()  # whole_ave
        else:  # rnafm
            out = self.model(**inputs)
            hidden = out["last_hidden_state"]  # (B, L+2, D)
            return hidden[:, 0, :].detach().cpu()  # [CLS]

    @torch.no_grad()
    def embed_equal_length(self, seqs: list, batch_size: int, token_budget: int) -> torch.Tensor:
        """Embed a list of equal-length sequences, returning (N, D) on CPU."""
        if len(seqs) == 0:
            return torch.empty(0)
        seq_len = len(seqs[0])
        # Shrink the batch for long sequences to bound memory / attention cost.
        eff_bs = max(1, min(batch_size, token_budget // max(seq_len, 1)))
        if self.emb_type == "rnafm" and seq_len > self.MAX_SEG_LEN:
            # Segmentation differs per length bucket; fall back to per-sequence path.
            return torch.stack([self._embed_one_long_rnafm(s) for s in seqs])
        outs = []
        for i in range(0, len(seqs), eff_bs):
            outs.append(self._forward_pool(seqs[i : i + eff_bs]))
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def _embed_one_long_rnafm(self, seq: str) -> torch.Tensor:
        """RNA-FM 'average' over_length: average [CLS] across <=1022 nt segments."""
        frags = [seq[i : i + self.MAX_SEG_LEN] for i in range(0, len(seq), self.MAX_SEG_LEN)]
        embs = [self._forward_pool([f])[0] for f in frags]
        return torch.stack(embs).mean(dim=0)

    @torch.no_grad()
    def embed_one(self, seq: str) -> torch.Tensor:
        """Embed a single sequence (any length), returning (D,) on CPU."""
        if self.emb_type == "rnafm" and len(seq) > self.MAX_SEG_LEN:
            return self._embed_one_long_rnafm(seq)
        return self._forward_pool([seq])[0]


# --------------------------------------------------------------------------- #
# Relation score
# --------------------------------------------------------------------------- #
@torch.no_grad()
def relation_score_batch(
    model, emb5: torch.Tensor, emb3: torch.Tensor, device: str
) -> np.ndarray:
    """Cosine-similarity relation score for paired embeddings.

    Mirrors ``PairPredCR.predict``: pass through each tower, L2-normalize, dot.

    Args:
        emb5: (B, D) 5' UTR embeddings.
        emb3: (B, D) 3' UTR embeddings.

    Returns:
        (B,) relation scores.
    """
    emb5 = emb5.to(device)
    emb3 = emb3.to(device)
    f5 = model.network_utr5(emb5)
    f3 = model.network_utr3(emb3)
    f5 = f5 / f5.norm(dim=1, keepdim=True)
    f3 = f3 / f3.norm(dim=1, keepdim=True)
    return (f5 * f3).sum(dim=1).detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Window utilities
# --------------------------------------------------------------------------- #
def window_starts(seq_len: int, window_frac: float, stride_frac: float) -> list:
    """Start indices of sliding windows; always includes the terminal window."""
    w = max(1, round(window_frac * seq_len))
    w = min(w, seq_len)
    stride = max(1, round(stride_frac * seq_len))
    starts = list(range(0, seq_len - w + 1, stride))
    if not starts:
        starts = [0]
    if starts[-1] != seq_len - w:
        starts.append(seq_len - w)
    return starts


def dinucleotide_shuffle(seq: str, rng: np.random.RandomState, max_tries: int = 100) -> str:
    """Shuffle ``seq`` while preserving its exact dinucleotide counts.

    Implements the Altschul-Erikson algorithm: the sequence is a walk on the graph
    whose vertices are nucleotides and whose edges are the observed dinucleotides, so
    any Eulerian walk with the same first and last base realizes the same dinucleotide
    composition. A uniformly random such walk is drawn by shuffling each vertex's
    outgoing edges after reserving one "last edge" per vertex such that the reserved
    edges form a tree rooted at the final base (which guarantees the walk stays
    connected and consumes every edge).

    Preserving dinucleotide counts approximately preserves base-pairing propensity, so
    contrasting this mode with ``"shuffle"`` (which preserves only mononucleotide
    composition) isolates how much of the score drop is attributable to base pairing.

    Sequences shorter than 3 nt have no degrees of freedom and are returned unchanged.
    """
    if len(seq) < 3:
        return seq

    first, last = seq[0], seq[-1]
    edges = {}
    for i in range(len(seq) - 1):
        edges.setdefault(seq[i], []).append(seq[i + 1])

    vertices = list(edges)
    for _ in range(max_tries):
        # Reserve one outgoing edge per vertex as the edge traversed last, then check
        # that following the reserved edges from any vertex reaches `last`.
        reserved = {}
        for v in vertices:
            if v != last:
                reserved[v] = edges[v][rng.randint(len(edges[v]))]
        if not _reserved_edges_form_tree(reserved, last, vertices):
            continue

        # Shuffle the remaining edges of each vertex and append its reserved edge.
        order = {}
        for v in vertices:
            rest = list(edges[v])
            if v in reserved:
                rest.remove(reserved[v])
            rng.shuffle(rest)
            order[v] = rest + ([reserved[v]] if v in reserved else [])

        # Walk the graph, consuming each vertex's edges in the shuffled order.
        cursor = dict.fromkeys(vertices, 0)
        out = [first]
        node = first
        for _ in range(len(seq) - 1):
            nxt = order[node][cursor[node]]
            cursor[node] += 1
            out.append(nxt)
            node = nxt
        return "".join(out)

    # Extremely rare: fall back to composition-preserving shuffle rather than failing.
    chars = list(seq)
    rng.shuffle(chars)
    return "".join(chars)


def _reserved_edges_form_tree(reserved: dict, last: str, vertices: list) -> bool:
    """True if following ``reserved`` from every vertex terminates at ``last``."""
    for v in vertices:
        node, steps = v, 0
        while node != last:
            if node not in reserved or steps > len(vertices):
                return False
            node = reserved[node]
            steps += 1
    return True


def randomize_window(
    seq: str, start: int, w: int, rng: np.random.RandomState, mode: str = "uniform"
) -> str:
    """Perturb ``seq[start:start+w]`` and return the mutated sequence.

    Args:
        mode:
            * ``"uniform"``: replace the window with uniformly random RNA bases
              (A/C/G/U). Both nucleotide composition (e.g. GC content) and the
              order/structure of the window are destroyed.
            * ``"shuffle"``: randomly permute the window's existing bases. Nucleotide
              composition is *preserved*; only the order — and therefore the local
              secondary structure — is destroyed. This isolates the contribution of
              sequence order/structure from that of base composition.
            * ``"dinucleotide"``: permute the window preserving its exact dinucleotide
              counts (Altschul-Erikson). Base-pairing propensity is largely retained,
              so ``shuffle`` minus ``dinucleotide`` estimates the base-pairing share of
              the score drop.
    """
    window = seq[start : start + w]
    if mode == "uniform":
        new_window = "".join(rng.choice(RNA_BASES, size=w))
    elif mode == "shuffle":
        chars = list(window)
        rng.shuffle(chars)  # in-place; preserves composition
        new_window = "".join(chars)
    elif mode == "dinucleotide":
        new_window = dinucleotide_shuffle(window, rng)
    else:
        raise ValueError(f"Unknown randomize mode: {mode}")
    return seq[:start] + new_window + seq[start + w :]


# --------------------------------------------------------------------------- #
# Per-transcript analysis
# --------------------------------------------------------------------------- #
def analyze_transcript(
    enst_id: str,
    gene: str,
    group: str,
    seq5: str,
    seq3: str,
    emb5_base: torch.Tensor,
    emb3_base: torch.Tensor,
    baseline_score: float,
    regions: list,
    embedder: Embedder,
    model,
    device: str,
    opt: argparse.Namespace,
    rng: np.random.RandomState,
) -> list:
    """Run sliding-window randomization for one transcript; return list of row dicts."""
    rows = []
    region_seq = {"utr5": seq5, "utr3": seq3}
    for region in regions:
        seq = region_seq[region]
        L = len(seq)
        w = min(max(1, round(opt.window_frac * L)), L)
        starts = window_starts(L, opt.window_frac, opt.stride_frac)

        for mode in opt.randomize_mode:
            # Build every mutated sequence for this region+mode (all share length L)
            # so they can be embedded in one batched sweep.
            mutated_seqs = []
            meta = []  # (start, trial)
            for start in starts:
                for trial in range(opt.n_trials):
                    mutated_seqs.append(randomize_window(seq, start, w, rng, mode))
                    meta.append((start, trial))

            mut_emb = embedder.embed_equal_length(mutated_seqs, opt.batch_size, opt.token_budget)

            # Score each mutated sequence against the *intact* partner region.
            if region == "utr5":
                partner = emb3_base.unsqueeze(0).repeat(mut_emb.size(0), 1)
                scores = relation_score_batch(model, mut_emb, partner, device)
            else:
                partner = emb5_base.unsqueeze(0).repeat(mut_emb.size(0), 1)
                scores = relation_score_batch(model, partner, mut_emb, device)

            # Aggregate trials per window position.
            by_start = {}
            for (start, _), s in zip(meta, scores):
                by_start.setdefault(start, []).append(float(s))
            for start in starts:
                vals = np.asarray(by_start[start])
                mut_mean = float(vals.mean())
                rows.append(
                    {
                        "ENST_ID": enst_id,
                        "GENE": gene,
                        "group": group,
                        "region": region,
                        "mode": mode,
                        "seq_len": L,
                        "window_len": w,
                        "win_start": int(start),
                        "win_end": int(start + w),
                        "rel_start": start / L,
                        "rel_center": (start + w / 2) / L,
                        "rel_end": (start + w) / L,
                        "baseline_score": baseline_score,
                        "mut_score_mean": mut_mean,
                        "mut_score_std": float(vals.std()),
                        # delta > 0  =>  randomizing this window *lowers* the relation
                        # score  =>  the region is important for the predicted relation.
                        "delta": baseline_score - mut_mean,
                        "n_trials": int(len(vals)),
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# HRU / control selection
# --------------------------------------------------------------------------- #
def select_groups(opt: argparse.Namespace, seq_df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with columns [seq_idx, ENST_ID, relation_score, group].

    ``seq_idx`` is the positional index into ``seq_df`` (used by the dataset).
    """
    enst_to_idx = {e: i for i, e in enumerate(seq_df["ENST_ID"].values)}
    enst_to_idx_pre = {e.split(".")[0]: i for i, e in enumerate(seq_df["ENST_ID"].values)}

    def _map_idx(enst):
        if enst in enst_to_idx:
            return enst_to_idx[enst]
        return enst_to_idx_pre.get(str(enst).split(".")[0], None)

    if opt.hru_csv is not None:
        df = pd.read_csv(opt.hru_csv)
        score_col = next((c for c in ["cos_sim", "relation_score", "score"] if c in df.columns), None)
        recs = []
        for _, r in df.iterrows():
            idx = _map_idx(r["ENST_ID"])
            if idx is None:
                continue
            recs.append(
                {
                    "seq_idx": idx,
                    "ENST_ID": seq_df.iloc[idx]["ENST_ID"],
                    "relation_score": float(r[score_col]) if score_col else np.nan,
                    "group": "HRU",
                }
            )
        groups = pd.DataFrame(recs)
    elif opt.cv_results_dir is not None:
        total_df = create_total_df(opt.cv_results_dir, seq_df, kfold=opt.kfold)
        # create_total_df keeps label==1 positives; 'utr5' is the positional seq_df idx.
        total_df = total_df.drop_duplicates(subset="utr5")
        hru = total_df[total_df["cos_sim"] >= opt.relation_threshold].copy()
        hru["group"] = "HRU"
        sel = [hru]
        if opt.control_high is not None:
            ctrl = total_df[
                (total_df["cos_sim"] >= opt.control_low) & (total_df["cos_sim"] < opt.control_high)
            ].copy()
            if opt.n_control is not None and len(ctrl) > opt.n_control:
                ctrl = ctrl.sample(n=opt.n_control, random_state=opt.seed)
            ctrl["group"] = "control"
            sel.append(ctrl)
        cat = pd.concat(sel, ignore_index=True)
        groups = pd.DataFrame(
            {
                "seq_idx": cat["utr5"].astype(int).values,
                "ENST_ID": seq_df.iloc[cat["utr5"].astype(int).values]["ENST_ID"].values,
                "relation_score": cat["cos_sim"].values,
                "group": cat["group"].values,
            }
        )
    else:
        raise ValueError("Provide either --cv_results_dir or --hru_csv to define HRUs.")

    groups = groups.sort_values("relation_score", ascending=False).reset_index(drop=True)
    if opt.max_hrus is not None:
        hru_part = groups[groups["group"] == "HRU"].head(opt.max_hrus)
        other = groups[groups["group"] != "HRU"]
        groups = pd.concat([hru_part, other], ignore_index=True)
    return groups


# --------------------------------------------------------------------------- #
# Aggregation & plotting
# --------------------------------------------------------------------------- #
def aggregate_and_save(df: pd.DataFrame, opt: argparse.Namespace) -> pd.DataFrame:
    """Bin by relative position and summarize delta per (group, region, bin)."""
    bins = np.linspace(0, 1, opt.n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    df = df.copy()
    df["pos_bin"] = pd.cut(df["rel_center"], bins=bins, labels=False, include_lowest=True)
    agg = (
        df.groupby(["group", "region", "mode", "pos_bin"])["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    agg["rel_pos"] = agg["pos_bin"].map(lambda b: centers[int(b)] if pd.notna(b) else np.nan)
    agg.to_csv(os.path.join(opt.out_dir, "position_aggregate.csv"), index=False)
    return agg


def summarize_max_windows(df: pd.DataFrame, opt: argparse.Namespace) -> pd.DataFrame:
    """For each transcript+region+mode, find the window with the largest delta."""
    idx = df.groupby(["ENST_ID", "group", "region", "mode"])["delta"].idxmax()
    summary = df.loc[idx].reset_index(drop=True)
    summary.to_csv(os.path.join(opt.out_dir, "max_window_per_transcript.csv"), index=False)
    return summary


def make_plots(agg: pd.DataFrame, summary: pd.DataFrame, opt: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (1) mean delta vs relative position, one line per (group, mode).
    for region in opt.regions:
        plt.figure(figsize=(6, 4))
        sub_all = agg[agg["region"] == region]
        for (group, mode), gdf in sub_all.groupby(["group", "mode"]):
            gdf = gdf.sort_values("rel_pos")
            label = f"{group} ({mode})" if len(opt.randomize_mode) > 1 else group
            plt.plot(gdf["rel_pos"], gdf["mean"], marker="o", label=label)
            plt.fill_between(
                gdf["rel_pos"], gdf["mean"] - gdf["sem"], gdf["mean"] + gdf["sem"], alpha=0.2
            )
        plt.axhline(0, color="grey", lw=0.8, ls="--")
        plt.xlabel(f"Relative position in {region} (0 = 5' end, 1 = 3' end)")
        plt.ylabel("Δ relation score (baseline − randomized)")
        plt.title(f"Window randomization sensitivity: {region}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(opt.out_dir, f"delta_vs_position_{region}.png"), dpi=300)
        plt.close()

    # (2) distribution of the most-sensitive relative position (HRU only), per mode.
    hru_sum = summary[summary["group"] == "HRU"]
    for region in opt.regions:
        for mode in opt.randomize_mode:
            sub = hru_sum[(hru_sum["region"] == region) & (hru_sum["mode"] == mode)]
            if len(sub) == 0:
                continue
            plt.figure(figsize=(6, 4))
            plt.hist(sub["rel_center"], bins=opt.n_bins, range=(0, 1), color="#c0392b", alpha=0.8)
            plt.xlabel(f"Relative position of max-Δ window in {region} ({mode})")
            plt.ylabel("# HRUs")
            plt.title(f"Most-sensitive window location: {region} ({mode})")
            plt.tight_layout()
            plt.savefig(os.path.join(opt.out_dir, f"max_window_hist_{region}_{mode}.png"), dpi=300)
            plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(opt: argparse.Namespace) -> None:
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    rng = np.random.RandomState(opt.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(opt.out_dir, exist_ok=True)

    cfg = OmegaConf.load(opt.cfg)
    seq_data = opt.seq_data or cfg.seq_data
    seq_df = pd.read_csv(seq_data, index_col=0).reset_index(drop=True)
    print(f"Loaded {len(seq_df)} transcripts from {seq_data}")

    # --- model ---
    if opt.input_dim is not None:
        cfg.model.input_dim = opt.input_dim
    model = MODEL_DICT[cfg.model.arch](cfg.model)
    state = torch.load(opt.model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print(f"Loaded model ({cfg.model.arch}, input_dim={cfg.model.input_dim}) from {opt.model_path}")

    embedder = Embedder(opt.emb_type, device)
    print(f"Embedder ready: {opt.emb_type} on {device}")
    print(f"Randomize mode(s): {opt.randomize_mode}")

    # --- HRU / control selection ---
    groups = select_groups(opt, seq_df)
    groups.to_csv(os.path.join(opt.out_dir, "analyzed_groups.csv"), index=False)
    n_hru = int((groups["group"] == "HRU").sum())
    n_ctrl = int((groups["group"] == "control").sum())
    msg = f"Selected {n_hru} HRUs (relation score >= {opt.relation_threshold})"
    if n_ctrl:
        msg += f" and {n_ctrl} control pairs"
    print(msg)

    # --- per-transcript sweep ---
    all_rows = []
    for n, (_, g) in enumerate(groups.iterrows(), start=1):
        idx = int(g["seq_idx"])
        row = seq_df.iloc[idx]
        seq5, seq3 = str(row["5UTR"]), str(row["3UTR"])
        enst_id, gene = row["ENST_ID"], row["GENE"]

        emb5_base = embedder.embed_one(seq5)
        emb3_base = embedder.embed_one(seq3)
        baseline_score = float(
            relation_score_batch(model, emb5_base.unsqueeze(0), emb3_base.unsqueeze(0), device)[0]
        )

        rows = analyze_transcript(
            enst_id, gene, g["group"], seq5, seq3,
            emb5_base, emb3_base, baseline_score,
            opt.regions, embedder, model, device, opt, rng,
        )
        all_rows.extend(rows)
        print(
            f"[{n}/{len(groups)}] {enst_id} ({g['group']}) "
            f"baseline={baseline_score:.3f} 5'UTR_len={len(seq5)} 3'UTR_len={len(seq3)}"
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(opt.out_dir, "window_randomization_results.csv"), index=False)
    print(f"Wrote per-window results: {len(df)} rows")

    agg = aggregate_and_save(df, opt)
    summary = summarize_max_windows(df, opt)

    if not opt.no_plot:
        try:
            make_plots(agg, summary, opt)
            print("Wrote figures.")
        except Exception as e:  # plotting must never break the data outputs
            print(f"[warn] plotting failed: {e}")

    print(f"\nDone. Outputs in: {opt.out_dir}")
    print("  - analyzed_groups.csv               : transcripts analyzed + relation scores")
    print("  - window_randomization_results.csv  : per-window delta (long format)")
    print("  - position_aggregate.csv            : mean delta per relative-position bin")
    print("  - max_window_per_transcript.csv     : most-sensitive window per transcript")


if __name__ == "__main__":
    main(_argparse())
