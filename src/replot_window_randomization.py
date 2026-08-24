"""Re-plot the window-randomization sensitivity figure from existing CSV output.

``run_window_randomization.py`` writes ``position_aggregate.csv`` (mean Δ relation
score per group × region × mode × relative-position bin) and, on the GPU machine, a
companion ``delta_vs_position_{region}.png`` figure. This script regenerates *only*
that figure from the CSV — no model, no GPU — so the publication styling can be
iterated anywhere:

  * drops the ``dinucleotide`` mode (keeps ``uniform`` + ``shuffle``),
  * relabels the region as ``5'UTR`` / ``3'UTR`` in the title and x-axis,
  * places the legend outside the axes (upper-right).

The underlying CSV is left untouched (the dropped mode is only excluded from the
plot, exactly as in the w20 figures).

Example
-------
    python replot_window_randomization.py \
        --result_dir ../results/window_randomization_human_w10
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Region keys used in the CSV -> publication labels for title / axis / legend.
REGION_LABEL = {"utr5": "5'UTR", "utr3": "3'UTR"}

# Legend / draw order: groups first, then modes within each group. The figure
# therefore reads HRU (uniform), HRU (shuffle), control (uniform), control (shuffle).
GROUP_ORDER = ["HRU", "control"]
MODE_ORDER = ["uniform", "shuffle", "dinucleotide"]


def _ordered_keys(present: list) -> list:
    """Sort (group, mode) pairs by GROUP_ORDER then MODE_ORDER (unknowns last)."""

    def key(gm):
        group, mode = gm
        g = GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)
        m = MODE_ORDER.index(mode) if mode in MODE_ORDER else len(MODE_ORDER)
        return (g, m)

    return sorted(present, key=key)


def _argparse() -> argparse.Namespace:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument(
        "--result_dir",
        required=True,
        type=str,
        help="experiment output dir containing position_aggregate.csv",
    )
    args.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="where to write the figures (default: same as --result_dir)",
    )
    args.add_argument(
        "--drop_modes",
        type=str,
        nargs="*",
        default=["dinucleotide"],
        help="randomization mode(s) to exclude from the plot (default: dinucleotide)",
    )
    return args.parse_args()


def plot_delta_vs_position(agg: pd.DataFrame, region: str, out_path: str) -> None:
    """Mean Δ vs relative position for one region; one line per (group, mode)."""
    label_region = REGION_LABEL.get(region, region)
    n_modes = agg["mode"].nunique()
    sub_all = agg[agg["region"] == region]

    present = list(sub_all.groupby(["group", "mode"]).groups.keys())
    # Keep each series' colour stable (default cycle in canonical sorted order)
    # regardless of the legend/draw order chosen below.
    color_map = {key: f"C{i}" for i, key in enumerate(sorted(present))}

    plt.figure(figsize=(6, 4))
    for group, mode in _ordered_keys(present):
        gdf = sub_all[(sub_all["group"] == group) & (sub_all["mode"] == mode)].sort_values("rel_pos")
        color = color_map[(group, mode)]
        label = f"{group} ({mode})" if n_modes > 1 else group
        plt.plot(gdf["rel_pos"], gdf["mean"], marker="o", color=color, label=label)
        plt.fill_between(
            gdf["rel_pos"], gdf["mean"] - gdf["sem"], gdf["mean"] + gdf["sem"],
            color=color, alpha=0.2,
        )
    plt.axhline(0, color="grey", lw=0.8, ls="--")
    plt.xlabel(f"Relative position in {label_region} (0 = 5' end, 1 = 3' end)")
    plt.ylabel("Δ relation score (baseline − randomized)")
    plt.title(f"Window randomization sensitivity: {label_region}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main(opt: argparse.Namespace) -> None:
    out_dir = opt.out_dir or opt.result_dir
    os.makedirs(out_dir, exist_ok=True)

    agg = pd.read_csv(os.path.join(opt.result_dir, "position_aggregate.csv"))
    if opt.drop_modes:
        agg = agg[~agg["mode"].isin(opt.drop_modes)]

    for region in sorted(agg["region"].unique()):
        out_path = os.path.join(out_dir, f"delta_vs_position_{region}.png")
        plot_delta_vs_position(agg, region, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main(_argparse())
