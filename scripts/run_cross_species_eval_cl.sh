#!/bin/bash
# Cross-species evaluation script for **Contrastive Learning** models with **RiNALMo** embedding
# This script trains models on one species and evaluates on another species' data

# ============================================
# Train on Human -> Evaluate on Mouse
# ============================================
echo "Training CL model on Human data..."
poetry run python ../src/run_train_cl.py \
    --cfg ../config/human_cl_cross_species_rnafm.yaml

echo "Evaluating Human CL model on Mouse data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/human_cl_cross_species_rnafm.yaml \
    --model_path ../results/runs/cross_species_human_cl_rnafm/best_model.pth \
    --eval_species mouse \
    --method cl

# ============================================
# Train on Mouse -> Evaluate on Human
# ============================================
echo "Training CL model on Mouse data..."
poetry run python ../src/run_train_cl.py \
    --cfg ../config/mouse_cl_cross_species_rnafm.yaml

echo "Evaluating Mouse CL model on Human data..."
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/mouse_cl_cross_species_rnafm.yaml \
    --model_path ../results/runs/cross_species_mouse_cl_rnafm/best_model.pth \
    --eval_species human \
    --method cl

echo "Cross-species training and evaluation for CL models completed!"
