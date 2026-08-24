"""Compare window-randomization sensitivity between window sizes (10% vs 20%).

``run_window_randomization.py`` writes ``position_aggregate.csv`` (mean Δ relation
score per group × region × mode × relative-position bin) for each experiment. This
script overlays two experiments (e.g. window = 10% vs 20% of the sequence length)
in a 2×2 grid (5'UTR / 3'UTR rows, window-size columns), so the effect of the
randomized-window width on the Δ profile is directly visible:

  * larger window  → larger Δ (a wider region is destroyed),
  * uniform > shuffle (base composition contributes on top of order/structure),
  * HRU ≫ control at every relative position.

No model / GPU needed — it only reads the two CSVs. Output: SVG + PNG (+ optional PDF).

Example
-------
    poetry run python plot_window_size_comparison.py \
        --w10_dir ../results/window_randomization_human_w10 \
        --w20_dir ../results/window_randomization_human_w20 \
        --out_dir ../docs/figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# replot convention: the publication figures keep uniform + shuffle and drop the
# mononucleotide-/dinucleotide-preserving control modes from the plot.
DROP_MODES = ["dinucleotide"]
REGIONS = ["utr5", "utr3"]
REGION_LABEL = {"utr5": "5'UTR", "utr3": "3'UTR"}

# (group, mode) -> style. HRU in red, control in grey; uniform solid, shuffle dashed.
SERIES = [
    ("HRU", "uniform", {"color": "#c0392b", "ls": "-", "marker": "o", "label": "HRU · uniform"}),
    ("HRU", "shuffle", {"color": "#e08e84", "ls": "--", "marker": "s", "label": "HRU · shuffle"}),
    ("control", "uniform", {"color": "#5d6d7e", "ls": "-", "marker": "o", "label": "control · uniform"}),
    ("control", "shuffle", {"color": "#a6b1bb", "ls": "--", "marker": "s", "label": "control · shuffle"}),
]


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
    p.add_argument("--w10_dir", required=True, help="experiment dir for the 10% window (has position_aggregate.csv)")
    p.add_argument("--w20_dir", required=True, help="experiment dir for the 20% window")
    p.add_argument("--out_dir", required=True, help="where to write the figure")
    p.add_argument("--w10_label", default="window = 10% of L", help="column title for the first experiment")
    p.add_argument("--w20_label", default="window = 20% of L", help="column title for the second experiment")
    p.add_argument("--name", default="window_size_comparison", help="output file stem")
    p.add_argument("--pdf", action="store_true", help="also write a PDF")
    return p.parse_args()


def _load(result_dir: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(result_dir, "position_aggregate.csv"))
    return df[~df["mode"].isin(DROP_MODES)].copy()


def main(opt: argparse.Namespace) -> None:
    _use_serif()
    os.makedirs(opt.out_dir, exist_ok=True)

    cols = [(_load(opt.w10_dir), opt.w10_label), (_load(opt.w20_dir), opt.w20_label)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharex=True, sharey=True)

    for ci, (agg, col_title) in enumerate(cols):
        for ri, region in enumerate(REGIONS):
            ax = axes[ri, ci]
            sub = agg[agg["region"] == region]
            for group, mode, st in SERIES:
                gdf = sub[(sub["group"] == group) & (sub["mode"] == mode)].sort_values("rel_pos")
                if gdf.empty:
                    continue
                ax.plot(
                    gdf["rel_pos"], gdf["mean"],
                    color=st["color"], ls=st["ls"], marker=st["marker"],
                    ms=4, lw=1.8, label=st["label"],
                )
                ax.fill_between(
                    gdf["rel_pos"], gdf["mean"] - gdf["sem"], gdf["mean"] + gdf["sem"],
                    color=st["color"], alpha=0.15, lw=0,
                )
            ax.axhline(0, color="grey", lw=0.8, ls=":")
            ax.set_xlim(0, 1)
            ax.grid(True, which="major", axis="both", color="#eeeeee", lw=0.8)
            ax.set_axisbelow(True)
            if ri == 0:
                ax.set_title(col_title, fontsize=14, fontweight="bold", pad=8)
            if ci == 0:
                ax.set_ylabel(f"{REGION_LABEL[region]}\nΔ relation score", fontsize=12.5)
            if ri == 1:
                ax.set_xlabel("relative position  (0 = 5' end → 1 = 3' end)", fontsize=12)

    # one shared legend (taken from the last axis), placed under the grid.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005), fontsize=12,
    )

    fig.suptitle(
        "Window-randomization sensitivity: Δ relation score vs. relative position",
        fontsize=15.5, fontweight="bold", y=0.985,
    )
    fig.text(
        0.5, 0.945,
        "wider randomized window → larger Δ;  uniform > shuffle (composition + structure);  HRU >> control",
        ha="center", fontsize=11.5, color="#555",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))

    stem = os.path.join(opt.out_dir, opt.name)
    fig.savefig(stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(stem + ".svg", bbox_inches="tight")
    if opt.pdf:
        fig.savefig(stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .svg" + (" / .pdf" if opt.pdf else ""))


if __name__ == "__main__":
    main(_argparse())
