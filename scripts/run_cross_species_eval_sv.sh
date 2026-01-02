#!/bin/bash
# Cross-species evaluation script for **Supervised Learning** models
# This script trains models on one species and evaluates on another species' data
# Supports both RNAFM and RiNALMo embeddings

# ============================================
# RNAFM Embedding Experiments
# ============================================
echo "============================================"
echo "Starting RNAFM embedding experiments..."
echo "============================================"

# Train on Human -> Evaluate on Mouse (RNAFM)
echo "Training SV model on Human data with RNAFM..."
poetry run python ../src/run_train_sv.py \
    --cfg ../config/human_sv_cross_species_rnafm.yaml

echo "Evaluating Human SV model (RNAFM) on Mouse data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/human_sv_cross_species_rnafm.yaml \
    --model_path ../results/runs/cross_species_human_sv_rnafm/best_model.pth \
    --eval_species mouse \
    --method sv

# Train on Mouse -> Evaluate on Human (RNAFM)
echo "Training SV model on Mouse data with RNAFM..."
poetry run python ../src/run_train_sv.py \
    --cfg ../config/mouse_sv_cross_species_rnafm.yaml

echo "Evaluating Mouse SV model (RNAFM) on Human data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/mouse_sv_cross_species_rnafm.yaml \
    --model_path ../results/runs/cross_species_mouse_sv_rnafm/best_model.pth \
    --eval_species human \
    --method sv

# ============================================
# RiNALMo Embedding Experiments
# ============================================
echo "============================================"
echo "Starting RiNALMo embedding experiments..."
echo "============================================"

# Train on Human -> Evaluate on Mouse (RiNALMo)
echo "Training SV model on Human data with RiNALMo..."
poetry run python ../src/run_train_sv.py \
    --cfg ../config/human_sv_cross_species_rinalmo.yaml

echo "Evaluating Human SV model (RiNALMo) on Mouse data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/human_sv_cross_species_rinalmo.yaml \
    --model_path ../results/runs/cross_species_human_sv_rinalmo/best_model.pth \
    --eval_species mouse \
    --method sv

# Train on Mouse -> Evaluate on Human (RiNALMo)
echo "Training SV model on Mouse data with RiNALMo..."
poetry run python ../src/run_train_sv.py \
    --cfg ../config/mouse_sv_cross_species_rinalmo.yaml

echo "Evaluating Mouse SV model (RiNALMo) on Human data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/mouse_sv_cross_species_rinalmo.yaml \
    --model_path ../results/runs/cross_species_mouse_sv_rinalmo/best_model.pth \
    --eval_species human \
    --method sv

echo "============================================"
echo "Cross-species training and evaluation for SV models completed!"
echo "============================================"
