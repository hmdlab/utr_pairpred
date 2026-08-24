"""Per-position statistics for the window-randomization (in-silico mutagenesis) run.

``run_window_randomization.py`` writes ``window_randomization_results.csv`` (one row per
transcript x region x mode x window). This script turns it into the two aggregated tables
that the manuscript figures depend on:

  * ``analysis/binned_per_transcript.csv`` -- Delta averaged within each
    transcript x relative-position bin. Averaging per transcript first avoids
    pseudo-replication, since one transcript contributes several overlapping windows
    to the same bin.
  * ``analysis/per_bin_stats.csv`` -- per (region, mode, bin) test of HRU vs control
    (Mann-Whitney U, two-sided). If the run has no control group, HRU Delta is instead
    tested against 0 (Wilcoxon signed-rank). p-values are BH/FDR corrected *within*
    each region x mode table.

These are the same computations as the first half of
``notebooks/window_randomization_analysis.ipynb``; they live here as well so the whole
figure pipeline is runnable as scripts (see ``scripts/run_ism_pipeline.sh``) without
executing a notebook. The notebook remains the interactive companion for the heatmaps
and boxplots.

Requires ``scipy``; ``statsmodels`` is optional (without it, ``p_fdr`` falls back to the
uncorrected ``p``, and a warning is printed).

Example
-------
    poetry run python analyze_window_randomization_stats.py \
        --result_dir ../results/window_randomization_human_w20
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

try:
    from statsmodels.stats.multitest import multipletests

    HAS_SM = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_SM = False


def _argparse() -> argparse.Namespace:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument(
        "--result_dir",
        type=str,
        default="../results/window_randomization_human_w20",
        help="experiment output directory written by run_window_randomization.py",
    )
    args.add_argument(
        "--n_bins",
        type=int,
        default=10,
        help="number of relative-position bins (must match the figures; default: 10)",
    )
    return args.parse_args()


def bin_and_aggregate(df: pd.DataFrame, n_bins: int) -> tuple:
    """Average Delta within each transcript x relative-position bin.

    Returns the long-format table and the bin centres.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    d = df.copy()
    d["pos_bin"] = pd.cut(d["rel_center"], bins=bins, labels=False, include_lowest=True)
    binned = d.groupby(["ENST_ID", "GENE", "group", "region", "mode", "pos_bin"])["delta"].mean().reset_index()
    binned["rel_pos"] = binned["pos_bin"].map(lambda b: centers[int(b)])
    return binned, centers


def per_bin_stats(
    binned: pd.DataFrame,
    region: str,
    mode: str,
    n_bins: int,
    has_control: bool,
    centers: np.ndarray,
) -> pd.DataFrame:
    """Test HRU vs control (or HRU vs 0) in every position bin of one region x mode."""
    sub = binned[(binned["region"] == region) & (binned["mode"] == mode)]
    rows = []
    for b in range(n_bins):
        bb = sub[sub["pos_bin"] == b]
        hru = bb[bb["group"] == "HRU"]["delta"].values
        ctrl = bb[bb["group"] == "control"]["delta"].values
        stat, p, test = np.nan, np.nan, "NA"
        if has_control and len(hru) > 0 and len(ctrl) > 0:
            stat, p = mannwhitneyu(hru, ctrl, alternative="two-sided")
            test = "MWU(HRU vs control)"
        elif len(hru) > 1 and np.any(hru != 0):
            try:
                stat, p = wilcoxon(hru)
                test = "Wilcoxon(HRU vs 0)"
            except ValueError:
                pass
        rows.append(
            {
                "region": region,
                "mode": mode,
                "pos_bin": b,
                "rel_pos": centers[b],
                "hru_n": len(hru),
                "ctrl_n": len(ctrl),
                "hru_mean_delta": float(np.mean(hru)) if len(hru) else np.nan,
                "ctrl_mean_delta": float(np.mean(ctrl)) if len(ctrl) else np.nan,
                "stat": stat,
                "p": p,
                "test": test,
            }
        )
    res = pd.DataFrame(rows)
    valid = res["p"].notna()
    if HAS_SM and valid.sum() > 0:
        res.loc[valid, "p_fdr"] = multipletests(res.loc[valid, "p"], method="fdr_bh")[1]
    else:
        res["p_fdr"] = res["p"]
    return res


def main(opt: argparse.Namespace) -> None:
    results_csv = os.path.join(opt.result_dir, "window_randomization_results.csv")
    if not os.path.isfile(results_csv):
        raise SystemExit(
            f"required input missing: {results_csv}\n"
            "(run scripts/run_window_randomization.sh first, or pass --result_dir)"
        )
    if not HAS_SM:
        print("[warn] statsmodels unavailable; p_fdr will be the uncorrected p-value")

    analysis_dir = os.path.join(opt.result_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    df = pd.read_csv(results_csv)
    print(f"{len(df)} window rows | groups: {df['group'].value_counts().to_dict()}")

    binned, centers = bin_and_aggregate(df, opt.n_bins)
    binned_path = os.path.join(analysis_dir, "binned_per_transcript.csv")
    binned.to_csv(binned_path, index=False)
    print(f"wrote {binned_path} ({len(binned)} rows)")

    regions = sorted(df["region"].unique())
    modes = sorted(df["mode"].unique())
    has_control = (binned["group"] == "control").any()
    print(f"regions={regions} modes={modes} has_control={has_control}")

    stats_df = pd.concat(
        [per_bin_stats(binned, r, m, opt.n_bins, has_control, centers) for r in regions for m in modes],
        ignore_index=True,
    )
    stats_path = os.path.join(analysis_dir, "per_bin_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"wrote {stats_path} ({len(stats_df)} rows)")


if __name__ == "__main__":
    main(_argparse())
