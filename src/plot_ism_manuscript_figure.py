#!/usr/bin/env python3
"""Manuscript figure for the in-silico mutagenesis (sliding-window
randomization) analysis added in response to reviewer comments.

Produces a 2-panel position-profile figure matching the visual style of the
existing manuscript figures (seaborn 'deep' palette):

  a) HRU vs. control mean Delta(relation score) along the 5' UTR
  b) HRU vs. control mean Delta(relation score) along the 3' UTR

Each panel shows all four series -- HRU/control x uniform/shuffle -- as in
``results/window_randomization_human_w{10,20}/delta_vs_position_*.png``:
group is encoded by colour (HRU = blue, control = orange) and randomization
mode by line style (uniform = solid, shuffle = dashed), with a mean +/- SEM
band around each curve. A single shared legend is placed outside the axes.

Inputs (per result dir):
  - position_aggregate.csv            (mean / sem per group x region x mode x bin)
  - analysis/per_bin_stats.csv        (HRU-vs-control MWU p_fdr per bin)

Outputs:
  - w20 (main):       docs/figures/fig_ism_main.{png,svg}
  - w10 (supplement): docs/figures/fig_ism_w10.{png,svg}

Example
-------
    python plot_ism_manuscript_figure.py \
        --w20_dir ../results/window_randomization_human_w20 \
        --w10_dir ../results/window_randomization_human_w10 \
        --out_dir ../docs/figures
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]

# ---- style: match existing manuscript figures (seaborn deep palette) ----
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.facecolor": "#EAEAF2",   # seaborn darkgrid-ish background used in Figs 3/5
    "axes.edgecolor": "white",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 0.9,
    "axes.axisbelow": True,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "svg.fonttype": "none",
})
C_HRU = "#4C72B0"     # seaborn deep blue   -> HRU (relation 0.7-1.0)
C_CTRL = "#DD8452"    # seaborn deep orange -> control (relation 0.0-0.1)

# group -> colour ; mode -> line style + marker
GROUP_STYLE = {"HRU": C_HRU, "control": C_CTRL}
MODE_STYLE = {
    "uniform": {"ls": "-", "marker": "o"},
    "shuffle": {"ls": "--", "marker": "^"},
}
# legend / draw order
SERIES = [("HRU", "uniform"), ("HRU", "shuffle"),
          ("control", "uniform"), ("control", "shuffle")]
GROUP_LABEL = {"HRU": "HRU (0.7-1.0)", "control": "control (0.0-0.1)"}
MODES = ["uniform", "shuffle"]


def stars(p):
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def make_figure(result_dir: Path, out_base: str, window_pct: int, out_dir: Path) -> None:
    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        raise SystemExit(
            f"result dir not found: {result_dir} "
            "(run scripts/run_window_randomization.sh first, or pass --w10_dir/--w20_dir)"
        )
    agg_path = result_dir / "position_aggregate.csv"
    pb_path = result_dir / "analysis" / "per_bin_stats.csv"
    for p in (agg_path, pb_path):
        if not p.is_file():
            raise SystemExit(f"required input missing: {p}")

    agg = pd.read_csv(agg_path)
    agg = agg[agg["mode"].isin(MODES)]
    pb = pd.read_csv(pb_path)

    fig = plt.figure(figsize=(10.0, 4.3))
    gs = fig.add_gridspec(1, 2, wspace=0.24,
                          left=0.07, right=0.985, top=0.88, bottom=0.30)

    for ax_i, (region, label, panel) in enumerate(
            [("utr5", "5' UTR", "a"), ("utr3", "3' UTR", "b")]):
        ax = fig.add_subplot(gs[0, ax_i])
        sub = agg[agg["region"] == region]
        for group, mode in SERIES:
            s = sub[(sub["group"] == group) & (sub["mode"] == mode)].sort_values("rel_pos")
            if s.empty:
                continue
            color = GROUP_STYLE[group]
            st = MODE_STYLE[mode]
            ax.plot(s["rel_pos"], s["mean"], color=color, lw=1.8,
                    ls=st["ls"], marker=st["marker"], ms=4)
            ax.fill_between(s["rel_pos"], s["mean"] - s["sem"], s["mean"] + s["sem"],
                            color=color, alpha=0.18, lw=0)
        ax.axhline(0, color="0.4", lw=0.8, ls=":")

        # significance: HRU vs control under uniform (headline comparison)
        u = pb[(pb["region"] == region) & (pb["mode"] == "uniform")]
        if not u.empty:
            minp = u["p_fdr"].min()
            ax.text(0.5, 0.96, f"HRU vs. control (uniform): all bins {stars(minp)}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=7.5, color="0.25")

        ax.set_xlim(0, 1)
        ax.set_xlabel(f"relative position in {label} (0 = 5' end → 1 = 3' end)")
        if ax_i == 0:
            ax.set_ylabel(r"$\Delta$ relation score (mean $\pm$ SEM)")
        ax.set_title(f"{label}: per-position mutation sensitivity", loc="center")
        ax.text(-0.13, 1.08, panel, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    # ---- single shared legend, outside the axes (below the panels) ----
    handles = [
        Line2D([0], [0], color=GROUP_STYLE[g], lw=1.8,
               ls=MODE_STYLE[m]["ls"], marker=MODE_STYLE[m]["marker"], ms=5,
               label=f"{GROUP_LABEL[g]} – {m}")
        for g, m in SERIES
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=4, frameon=True, framealpha=0.95,
               title=f"window = {window_pct}% of UTR length; "
                     "uniform = solid, shuffle = dashed")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out_dir / f"{out_base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / f"{out_base}.png")


def _argparse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--w20_dir", type=str, default=str(ROOT / "results/window_randomization_human_w20"),
        help="experiment dir for the 20%% window / main analysis (default: %(default)s)",
    )
    p.add_argument(
        "--w10_dir", type=str, default=str(ROOT / "results/window_randomization_human_w10"),
        help="experiment dir for the 10%% window / supplement (default: %(default)s)",
    )
    p.add_argument(
        "--out_dir", type=str, default=str(ROOT / "docs/figures"),
        help="where to write fig_ism_main / fig_ism_w10 (default: %(default)s)",
    )
    return p.parse_args()


def main(opt: argparse.Namespace) -> None:
    make_figure(Path(opt.w20_dir), "fig_ism_main", 20, Path(opt.out_dir))
    make_figure(Path(opt.w10_dir), "fig_ism_w10", 10, Path(opt.out_dir))


if __name__ == "__main__":
    main(_argparse())
