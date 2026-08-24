#!/bin/bash
# Sliding-window randomization (in-silico mutagenesis) analysis for HRUs.
#
# For each Highly Related UTR pair (HRU; relation score >= --relation_threshold) a
# contiguous window of the 5'UTR / 3'UTR is randomized while the partner region is
# kept intact, the mutated sequence is re-embedded with RiNALMo, and the change in
# relation score (cosine similarity from the contrastive-learning model) is measured
# as the window slides along the sequence.
#
# Delta relation score = baseline - randomized. Delta > 0 means randomizing that
# window lowers the score, i.e. the region matters for the predicted relationship.
# Three randomization modes are run so their contrast separates the contribution of
# base composition from that of sequence order / secondary structure:
#   uniform       random A/C/G/U     -> composition AND order/structure destroyed
#   shuffle       bases reordered    -> composition kept, order/structure destroyed
#   dinucleotide  dinucleotide freq. -> base-pairing propensity approximately kept
#
# This stage needs a GPU (RiNALMo inference). The CPU-only post-processing that turns
# its CSV output into the manuscript figures and tables is scripts/run_ism_pipeline.sh.
#
# Two window sizes are produced, matching the manuscript: window = 20% of the region
# length (main analysis) and 10% (supplementary). Both write to their own directory.
#
# Run from the `scripts/` directory:  sh run_window_randomization.sh
#
# --- paths (EDIT THESE) ---
CFG=../config/human_cl_cross_species_rinalmo.yaml
# A trained PairPredCR (contrastive) model. Using a model trained on the full
# dataset gives a clean, high baseline for every HRU.
MODEL_PATH=../results/runs/cross_species_human_cl_rinalmo/best_model.pth
SEQ_DATA=../data/human/gencode44_utr_gene_unique_cdhit09.csv
# Directory of the 10-fold CV results used in the notebook analysis. The HRU set is
# read from here (relation score >= --relation_threshold), matching the manuscript.
CV_RESULTS_DIR=../results/runs/contrastive_learning_10fold_rinalmo_whole_ave_seed1
OUT_ROOT=../results

set -e

# window_frac -> output directory suffix. 0.2 is the main analysis, 0.1 the supplement.
for WINDOW_FRAC in 0.2 0.1; do
    TAG="w$(python3 -c "print(int(${WINDOW_FRAC} * 100))")"
    OUT_DIR=${OUT_ROOT}/window_randomization_human_${TAG}

    echo "=== window_frac=${WINDOW_FRAC} -> ${OUT_DIR} ==="
    poetry run python ../src/run_window_randomization.py \
        --cfg ${CFG} \
        --model_path ${MODEL_PATH} \
        --seq_data ${SEQ_DATA} \
        --cv_results_dir ${CV_RESULTS_DIR} \
        --emb_type rinalmo \
        --input_dim 1280 \
        --relation_threshold 0.7 \
        --window_frac ${WINDOW_FRAC} \
        --stride_frac 0.05 \
        --n_trials 10 \
        --randomize_mode uniform shuffle dinucleotide \
        --regions utr5 utr3 \
        --n_bins 10 \
        --batch_size 16 \
        --control_high 0.1 \
        --n_control 157 \
        --seed 0 \
        --out_dir ${OUT_DIR}
done

echo "Window randomization analysis completed! Results in ${OUT_ROOT}/window_randomization_human_w{20,10}"
echo "Next: sh run_ism_pipeline.sh   (CPU-only figures and tables)"
