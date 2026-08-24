#!/bin/bash
# CPU-only post-processing pipeline for the in-silico mutagenesis (ISM) /
# window-randomization analysis.
#
# Two-stage pipeline overall:
#   stage 1 (GPU, NOT run here): scripts/run_window_randomization.sh
#       Slides a randomized window along each HRU's 5'UTR / 3'UTR, re-embeds
#       with RiNALMo, and writes results/window_randomization_human_w{20,10}/
#       (window_randomization_results.csv, max_window_per_transcript.csv,
#       position_aggregate.csv, analyzed_groups.csv). Run that script first;
#       this one only reads its raw CSV output.
#   stage 2 (CPU, this script): turns those CSVs into the manuscript figures,
#       per-position statistics and per-transcript outlier tables, in order:
#         1. analyze_window_randomization_stats.py - analysis/{binned_per_transcript,per_bin_stats}.csv (w10 + w20)
#         2. analyze_hru_top_sensitive.py           - per-transcript outlier analysis (w20)
#         3. replot_window_randomization.py         - delta_vs_position_{utr5,utr3}.png (w10 + w20)
#         4. plot_window_size_comparison.py         - window_size_comparison.{png,svg}
#         5. plot_ism_manuscript_figure.py          - fig_ism_main / fig_ism_w10.{png,svg}
#
# Dependencies: step 1 below needs scipy (required) and statsmodels (optional --
# without it, p_fdr falls back to the uncorrected p-value). Steps 2-5 only need
# matplotlib / pandas / numpy.
#
# Run from the `scripts/` directory:  sh run_ism_pipeline.sh
#
# The two result dirs, the sequence csv, and the figure output dir default to
# the paths below but can be overridden via environment variables without
# editing this file, e.g.:
#   FIG_DIR=/tmp/figs W10_DIR=/tmp/w10 W20_DIR=/tmp/w20 sh run_ism_pipeline.sh

set -e

# --- paths (override via env vars, or edit these defaults) ---
W10_DIR=${W10_DIR:-../results/window_randomization_human_w10}
W20_DIR=${W20_DIR:-../results/window_randomization_human_w20}
SEQ_DATA=${SEQ_DATA:-../data/human/gencode44_utr_gene_unique_cdhit09.csv}
FIG_DIR=${FIG_DIR:-../docs/figures}

# --- sanity checks: fail fast with a readable message instead of a traceback ---
for RESULT_DIR in "${W10_DIR}" "${W20_DIR}"; do
    if [ ! -d "${RESULT_DIR}" ]; then
        echo "error: ${RESULT_DIR} not found -- run scripts/run_window_randomization.sh first" >&2
        exit 1
    fi
    for REQUIRED in window_randomization_results.csv max_window_per_transcript.csv \
                    position_aggregate.csv analyzed_groups.csv; do
        if [ ! -f "${RESULT_DIR}/${REQUIRED}" ]; then
            echo "error: ${RESULT_DIR}/${REQUIRED} not found -- run scripts/run_window_randomization.sh first" >&2
            exit 1
        fi
    done
done
if [ ! -f "${SEQ_DATA}" ]; then
    echo "error: ${SEQ_DATA} not found -- pass SEQ_DATA=... pointing at the gencode44 sequence csv" >&2
    exit 1
fi

mkdir -p "${FIG_DIR}"

echo "=== [1/5] per-bin statistics -- HRU vs control (w10, w20) ==="
poetry run python ../src/analyze_window_randomization_stats.py --result_dir ${W10_DIR}
poetry run python ../src/analyze_window_randomization_stats.py --result_dir ${W20_DIR}

echo "=== [2/5] per-transcript outlier analysis (w20) ==="
poetry run python ../src/analyze_hru_top_sensitive.py \
    --result_dir ${W20_DIR} \
    --seq_csv ${SEQ_DATA} \
    --fig_dir ${FIG_DIR} \
    --window_tag w20

echo "=== [3/5] replot window-randomization sensitivity (w10, w20) ==="
poetry run python ../src/replot_window_randomization.py --result_dir ${W10_DIR}
poetry run python ../src/replot_window_randomization.py --result_dir ${W20_DIR}

echo "=== [4/5] window-size comparison figure (w10 vs w20) ==="
poetry run python ../src/plot_window_size_comparison.py \
    --w10_dir ${W10_DIR} \
    --w20_dir ${W20_DIR} \
    --out_dir ${FIG_DIR}

echo "=== [5/5] manuscript ISM figure (fig_ism_main / fig_ism_w10) ==="
poetry run python ../src/plot_ism_manuscript_figure.py \
    --w20_dir ${W20_DIR} \
    --w10_dir ${W10_DIR} \
    --out_dir ${FIG_DIR}

echo "ISM pipeline completed! Figures in ${FIG_DIR}"
