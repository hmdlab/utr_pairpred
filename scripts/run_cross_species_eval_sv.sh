#!/bin/bash
# Cross-species evaluation script for Supervised Learning models
# This script trains models on one species and evaluates on another species' data

# ============================================
# Train on Human -> Evaluate on Mouse
# ============================================
echo "Training SV model on Human data..."
poetry run python ../src/run_train_sv.py \
    --cfg ../config/human_sv_cross_species.yaml

echo "Evaluating Human SV model on Mouse data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/human_sv_cross_species.yaml \
    --model_path ../results/runs/cross_species_human_sv_seed0/best_model.pth \
    --eval_species mouse \
    --method sv

# ============================================
# Train on Mouse -> Evaluate on Human
# ============================================
echo "Training SV model on Mouse data..."
poetry run python ../src/run_train_sv.py \
    --cfg ../config/mouse_sv_cross_species.yaml

echo "Evaluating Mouse SV model on Human data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/mouse_sv_cross_species.yaml \
    --model_path ../results/runs/cross_species_mouse_sv_seed0/best_model.pth \
    --eval_species human \
    --method sv

echo "Cross-species training and evaluation for SV models completed!"
