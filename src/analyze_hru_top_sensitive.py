"""Post-hoc analysis of the most mutation-sensitive HRUs in a window-randomization run.

``run_window_randomization.py`` writes the raw per-window rows
(``window_randomization_results.csv``) and their per-transcript argmax
(``max_window_per_transcript.csv``). This script turns those into the
interpretation-level artifacts used in the manuscript / handover report — no model,
no GPU, only the CSVs plus the sequence table:

1. ``analysis/hru_top_sensitive_uniform_<tag>.csv`` — one row per HRU: its single
   most-sensitive window under ``mode=uniform`` (max Δ over both UTRs), with the
   relative drop (Δ/baseline) and the z-score of Δ within the HRU distribution.
2. ``analysis/hru_top_mode_decomposition_<tag>.csv`` — for the outliers
   (Δ > mean + ``--n_sd`` × SD), the three randomization modes evaluated *at the same
   window*, split into order/structure, base-pairing and composition contributions:

     order/struct% = Δ_shuffle / Δ_uniform                 (order + secondary structure)
     basepair%     = (Δ_shuffle − Δ_dinuc) / Δ_uniform     (dinucleotide = base-pairing)
     composition%  = (Δ_uniform − Δ_shuffle) / Δ_uniform   (base composition, e.g. GC)

3. ``analysis/hru_top_motif_scan_<tag>.csv`` — the actual window sequence of each
   outlier: GC content, positional flags (Kozak context / stop-codon proximity) and
   hits of known RBP-binding and RNA-structure motifs (literature consensus regexes).
4. ``<fig_dir>/hru_top_sensitive_profiles_<tag>.{png,svg}`` — per-transcript Δ profile
   of the outliers (one panel each, both UTRs, uniform + shuffle, star on the
   most-sensitive window).

Numbers, panel layout and motif counts reproduce the published w20 artifacts exactly.
Three details are worth spelling out:

* ``L`` in the *figure* panel titles is the **5' UTR length**, also for the two
  transcripts whose most-sensitive window sits in the 3' UTR (TGFB2 ``L=1366``, not
  its 3' UTR length 3257; OR13G1 ``L=338``, not 1306). ``L`` in the *decomposition
  CSV* is instead the length of the selected region. Both are kept as published.
* the m6A motif label reads ``(RRACH)`` while the pattern implemented is the usual
  DRACH consensus (``D`` = A/G/U); the published counts are DRACH counts.
* the ``flags`` column and the ARE-core label are in English here, whereas the first
  (internal) version of the csv had them in Japanese. Only the wording differs.

Example
-------
    poetry run python src/analyze_hru_top_sensitive.py \
        --result_dir results/window_randomization_human_w20 \
        --seq_csv data/human/gencode44_utr_gene_unique_cdhit09.csv \
        --fig_dir docs/figures --window_tag w20
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGIONS = ["utr5", "utr3"]
REGION_LABEL = {"utr5": "5'UTR", "utr3": "3'UTR"}
# region key -> sequence column in the gencode csv (sequences are in U / RNA notation)
SEQ_COL = {"utr5": "5UTR", "utr3": "3UTR"}

# randomization modes: uniform destroys composition + order, shuffle only order,
# dinucleotide preserves the dinucleotide (base-pairing) propensity.
MODE_UNIFORM, MODE_SHUFFLE, MODE_DINUC = "uniform", "shuffle", "dinucleotide"

# ---- figure style: 5'UTR = blue, 3'UTR = orange; uniform solid + markers, shuffle
# ---- the same colour dashed and translucent; star = most-sensitive window.
REGION_COLOR = {"utr5": "#2c7fb8", "utr3": "#e6550d"}
MODE_STYLE = {
    MODE_UNIFORM: {"ls": "-", "marker": "o", "ms": 3, "alpha": 1.0},
    MODE_SHUFFLE: {"ls": "--", "marker": "", "ms": 0, "alpha": 0.55},
}
STAR_STYLE = {"marker": "*", "s": 120, "c": "#f1c40f", "edgecolors": "#c0392b", "linewidths": 1.0, "zorder": 5}
N_COLS = 4

# Known RBP-binding / RNA-structure motifs, as literature-consensus regexes on RNA
# (U, not T). IUPAC: R = A/G, Y = C/U, W = A/U, H = A/C/U, D = A/G/U, N = any.
# Order is the reporting order of the ``motifs`` column. Counts are non-overlapping.
MOTIFS = [
    ("HuR/ELAVL1 ARE (AUUUA)", "AUUUA"),
    ("ARE core (WUAUUUAUW)", "[AU]UAUUUAU[AU]"),
    ("PUM1/2 PRE (UGUANAUA)", "UGUA[ACGU]AUA"),
    ("PTBP1 (UCUU/CU-rich)", "UCUU"),
    ("poly-pyrimidine (Y>=8)", "[CU]{8,}"),
    ("poly-U/TIA1 (U>=5)", "U{5,}"),
    # label kept as published; the pattern is the DRACH consensus (D = A/G/U).
    ("m6A DRACH (RRACH)", "[AGU][AG]AC[ACU]"),
    ("CPEB CPE (UUUUAW)", "UUUUA[AU]"),
    ("SRSF1 ESE (GGAGGA/RGAAGA)", "(?:GGAGGA|[AG]GAAGA)"),
    ("hnRNPK poly-C (C>=4)", "C{4,}"),
    ("G-quadruplex (rG4)", "G{3,}[ACGU]{1,7}G{3,}[ACGU]{1,7}G{3,}[ACGU]{1,7}G{3,}"),
    ("G-run (G>=4)", "G{4,}"),
]
NO_HIT = "-"


def _use_serif() -> None:
    """Prefer Times New Roman (paper style); silently fall back to a generic serif."""
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
        for name in ("Times New Roman", "Times", "Nimbus Roman"):
            if name in available:
                plt.rcParams["font.family"] = name
                return
    except Exception:
        pass
    plt.rcParams["font.family"] = "serif"


def _argparse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="experiment dir with window_randomization_results.csv and max_window_per_transcript.csv "
        "(default: <repo>/results/window_randomization_human_<window_tag>)",
    )
    p.add_argument(
        "--seq_csv",
        type=str,
        default=os.path.join(ROOT, "data", "human", "gencode44_utr_gene_unique_cdhit09.csv"),
        help="sequence csv (ENST_ID,GENE,5UTR,CDS,3UTR,...) used for the motif scan",
    )
    p.add_argument(
        "--fig_dir",
        type=str,
        default=os.path.join(ROOT, "docs", "figures"),
        help="where to write the per-transcript profile figure",
    )
    p.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help="where to write the analysis csvs (default: <result_dir>/analysis)",
    )
    p.add_argument("--window_tag", type=str, default="w20", help="window-size tag used in the output file names")
    p.add_argument(
        "--n_sd",
        type=float,
        default=2.0,
        help="outlier threshold on the most-sensitive-window Δ, in SDs above the HRU mean",
    )
    return p.parse_args()


def top_sensitive_per_transcript(max_win: pd.DataFrame) -> pd.DataFrame:
    """One row per HRU: its most-sensitive window under ``uniform`` (max Δ over both UTRs)."""
    hru = max_win[(max_win["group"] == "HRU") & (max_win["mode"] == MODE_UNIFORM)]
    top = hru.loc[hru.groupby("ENST_ID")["delta"].idxmax()]
    top = top.sort_values("delta", ascending=False).reset_index(drop=True)

    out = top[
        ["ENST_ID", "GENE", "region", "rel_center", "seq_len", "baseline_score", "mut_score_mean", "delta"]
    ].copy()
    # Δ is bounded by the baseline, so also report the relative drop; z locates each
    # transcript in the HRU-wide distribution of most-sensitive-window Δ.
    out["rel_drop"] = out["delta"] / out["baseline_score"]
    out["z"] = (out["delta"] - out["delta"].mean()) / out["delta"].std()
    return out


def mode_decomposition(results: pd.DataFrame, max_win: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    """Δ of all three randomization modes *at the most-sensitive window* of each outlier."""
    rows = []
    for _, top in outliers.iterrows():
        win = _max_window_row(max_win, top["ENST_ID"], top["region"])
        same = results[
            (results["ENST_ID"] == top["ENST_ID"])
            & (results["region"] == top["region"])
            & (results["win_start"] == win["win_start"])
        ]
        delta = {mode: float(same[same["mode"] == mode]["delta"].iloc[0]) for mode in same["mode"].unique()}
        d_uni, d_shuf, d_dinuc = delta[MODE_UNIFORM], delta[MODE_SHUFFLE], delta[MODE_DINUC]
        rows.append(
            {
                "GENE": top["GENE"],
                "region": top["region"],
                "rel_center": round(float(top["rel_center"]), 2),
                "win_nt": f"{int(win['win_start'])}-{int(win['win_end'])}",
                "L": int(win["seq_len"]),
                "baseline": round(float(top["baseline_score"]), 3),
                "d_uniform": round(d_uni, 3),
                "d_shuffle": round(d_shuf, 3),
                "d_dinuc": round(d_dinuc, 3),
                "order/struct%": int(round(100 * d_shuf / d_uni)),
                "basepair%": int(round(100 * (d_shuf - d_dinuc) / d_uni)),
                "composition%": int(round(100 * (d_uni - d_shuf) / d_uni)),
            }
        )
    return pd.DataFrame(rows)


def scan_motifs(window_seq: str) -> str:
    """``label×n; ...`` for every motif with at least one non-overlapping hit."""
    hits = [f"{label}×{n}" for label, pattern in MOTIFS if (n := len(re.findall(pattern, window_seq)))]
    return "; ".join(hits) if hits else NO_HIT


def positional_flags(region: str, win_start: int, win_end: int, seq: str) -> str:
    """Translation-relevant position of the window inside its UTR."""
    flags = []
    if region == "utr5" and win_end >= len(seq):
        # the window runs into the CDS start: it contains the Kozak context, whose
        # position -3 (3 nt upstream of the AUG) is the strongest determinant.
        flags.append(f"immediately upstream of CDS (Kozak context: -3={seq[-3]})")
    if region == "utr3" and win_start == 0:
        flags.append("immediately downstream of stop codon")
    return "; ".join(flags) if flags else NO_HIT


def motif_scan(seq_df: pd.DataFrame, max_win: pd.DataFrame, outliers: pd.DataFrame) -> pd.DataFrame:
    """Window sequence, GC content, positional flags and motif hits per outlier."""
    rows = []
    for _, top in outliers.iterrows():
        win = _max_window_row(max_win, top["ENST_ID"], top["region"])
        start, end = int(win["win_start"]), int(win["win_end"])
        seq = str(seq_df.loc[seq_df["ENST_ID"] == top["ENST_ID"], SEQ_COL[top["region"]]].iloc[0])
        window_seq = seq[start:end]
        rows.append(
            {
                "GENE": top["GENE"],
                "region": top["region"],
                "win": f"{start}-{end}",
                "win_len": len(window_seq),
                "gc": int(round(100 * sum(base in "GC" for base in window_seq) / len(window_seq))),
                "flags": positional_flags(top["region"], start, end, seq),
                "motifs": scan_motifs(window_seq),
            }
        )
    return pd.DataFrame(rows)


def _max_window_row(max_win: pd.DataFrame, enst_id: str, region: str) -> pd.Series:
    """The ``uniform`` argmax window of one (transcript, region)."""
    sub = max_win[
        (max_win["ENST_ID"] == enst_id) & (max_win["region"] == region) & (max_win["mode"] == MODE_UNIFORM)
    ]
    return sub.iloc[0]


def _window_percent(window_tag: str) -> str:
    """``w20`` -> ``20`` (for the figure title); anything else is used verbatim."""
    m = re.search(r"(\d+)", window_tag)
    return m.group(1) if m else window_tag


def plot_profiles(results: pd.DataFrame, outliers: pd.DataFrame, window_tag: str, out_stem: str) -> None:
    """One panel per outlier: Δ vs relative position, both UTRs, uniform + shuffle."""
    n_rows = -(-len(outliers) // N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(15.0, 3.6 * n_rows), sharey=True, squeeze=False)

    for i, (_, top) in enumerate(outliers.iterrows()):
        ax = axes[i // N_COLS][i % N_COLS]
        sub = results[results["ENST_ID"] == top["ENST_ID"]]
        for region in REGIONS:
            for mode, style in MODE_STYLE.items():
                gdf = sub[(sub["region"] == region) & (sub["mode"] == mode)].sort_values("rel_center")
                ax.plot(
                    gdf["rel_center"], gdf["delta"],
                    color=REGION_COLOR[region], ls=style["ls"], marker=style["marker"],
                    ms=style["ms"], lw=1.7, alpha=style["alpha"],
                )
        ax.scatter([top["rel_center"]], [top["delta"]], **STAR_STYLE)
        ax.axhline(0, color="grey", lw=0.7, ls=":")
        ax.set_xlim(0, 1)
        ax.grid(True, which="major", axis="both", color="#eeeeee", lw=0.7)
        ax.set_axisbelow(True)
        # published quirk: L is the 5' UTR length even when the top window is in the 3' UTR.
        utr5_len = int(sub[sub["region"] == "utr5"]["seq_len"].iloc[0])
        ax.set_title(
            f"{top['GENE']}  (L={utr5_len} nt)\n"
            f"baseline={top['baseline_score']:.2f}, max Δ={top['delta']:.2f} @{top['region']}",
            fontsize=10.5,
        )
        if i + N_COLS >= len(outliers):  # bottom-most panel of its column
            ax.set_xlabel("relative position (0=5'->1=3')")
        if i % N_COLS == 0:
            ax.set_ylabel("Δ relation score")
    for j in range(len(outliers), n_rows * N_COLS):
        axes[j // N_COLS][j % N_COLS].set_visible(False)

    handles = [
        Line2D(
            [0], [0], color=REGION_COLOR[region], lw=1.7, marker="o", ms=3,
            label=f"{REGION_LABEL[region]} uniform",
        )
        for region in REGIONS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.025), fontsize=11)
    fig.suptitle(
        f"HRUs with the largest mutation-induced score drop (window={_window_percent(window_tag)}%, uniform)"
        " - per-transcript Δ profile",
        fontsize=14, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.94,
        "yellow star = most-sensitive window;  solid = uniform, dashed = shuffle;  blue = 5'UTR, orange = 3'UTR",
        ha="center", fontsize=10.5, color="#555",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))

    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_stem + ".svg", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_stem}.png / .svg")


def main(opt: argparse.Namespace) -> None:
    _use_serif()
    tag = opt.window_tag
    result_dir = opt.result_dir or os.path.join(ROOT, "results", f"window_randomization_human_{tag}")
    analysis_dir = opt.analysis_dir or os.path.join(result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(opt.fig_dir, exist_ok=True)

    results = pd.read_csv(os.path.join(result_dir, "window_randomization_results.csv"))
    max_win = pd.read_csv(os.path.join(result_dir, "max_window_per_transcript.csv"))
    seq_df = pd.read_csv(opt.seq_csv)

    top = top_sensitive_per_transcript(max_win)
    threshold = top["delta"].mean() + opt.n_sd * top["delta"].std()
    outliers = top[top["delta"] > threshold].reset_index(drop=True)
    print(
        f"{len(top)} HRUs: most-sensitive-window Δ = {top['delta'].mean():.3f} ± {top['delta'].std():.3f} "
        f"(mean±SD); {len(outliers)} above mean+{opt.n_sd:g}SD = {threshold:.3f}"
    )

    for name, df in (
        (f"hru_top_sensitive_uniform_{tag}.csv", top),
        (f"hru_top_mode_decomposition_{tag}.csv", mode_decomposition(results, max_win, outliers)),
        (f"hru_top_motif_scan_{tag}.csv", motif_scan(seq_df, max_win, outliers)),
    ):
        path = os.path.join(analysis_dir, name)
        df.to_csv(path, index=False)
        print(f"wrote {path}")

    plot_profiles(results, outliers, tag, os.path.join(opt.fig_dir, f"hru_top_sensitive_profiles_{tag}"))


if __name__ == "__main__":
    main(_argparse())
