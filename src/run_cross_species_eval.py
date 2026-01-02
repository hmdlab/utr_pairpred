"""Cross-species evaluation script for UTR pair prediction models"""

import argparse
import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from _model_dict import MODEL_DICT
from dataset import CreateDataset, PairDatasetCL_test, PairDataset
from utils import discretize, metrics


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", required=True, type=str, help="path to config yaml")
    args.add_argument("--model_path", required=True, type=str, help="path to trained model")
    args.add_argument("--eval_species", required=True, type=str, help="species to evaluate (human or mouse)")
    args.add_argument("--method", required=True, type=str, help="method type (cl or sv)")
    args = args.parse_args()
    return args


def _parse_config(cfg_path: str) -> dict:
    """Load and return yaml format config

    Args:
        cfg_path (str): yaml config file path

    Returns:
        config (dict): config dict
    """
    config = OmegaConf.load(cfg_path)
    return config


def _random_seeds(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_cl(cfg, model, test_dataloader, device, result_dir):
    """Evaluation for contrastive learning models"""
    model.eval()
    preds = None

    for embs, labels, pair_idx in tqdm(test_dataloader, desc="Evaluating"):
        inputs = (embs[0].to(device), embs[1].to(device))
        logits, _ = model.predict(inputs)
        cos_sim = logits[0].diag()  # get diagonal values
        logits_sigmoid = F.sigmoid(cos_sim)

        if preds is None:
            pair_idx_list = [pair_idx]
            out_logits = logits_sigmoid.detach().cpu().numpy()
            out_cos_sim = cos_sim.detach().cpu().numpy()
            preds = discretize(logits_sigmoid.detach().cpu().numpy())
            out_labels = labels

        else:
            pair_idx_list.append(pair_idx)
            out_logits = np.append(
                out_logits, logits_sigmoid.detach().cpu().numpy(), axis=0
            )
            out_cos_sim = np.append(out_cos_sim, cos_sim.detach().cpu().numpy())
            preds = np.append(
                preds, discretize(logits_sigmoid.detach().cpu().numpy())
            )
            out_labels = np.append(out_labels, labels)

    scores = metrics(preds, out_labels, out_logits, "test")

    # Save results
    with open(os.path.join(result_dir, "score_dict.pkl"), "wb") as f:
        pickle.dump(scores, f)

    with open(os.path.join(result_dir, "pred_results.pkl"), "wb") as f:
        pickle.dump((out_cos_sim, out_logits, pair_idx_list), f)

    return scores


def evaluate_sv(cfg, model, test_dataloader, device, result_dir):
    """Evaluation for supervised learning models"""
    model.eval()
    preds = None
    sigmoid = torch.nn.Sigmoid()

    for data, labels, pair_idx in tqdm(test_dataloader, desc="Evaluating"):
        if "split" in cfg.model.arch:
            inputs = (data[0].to(device), data[1].to(device))
        else:
            inputs = torch.cat([data[0], data[1]], dim=1)
            if "cnn" in cfg.model.arch:
                inputs = inputs.unsqueeze(dim=-1)
            inputs = inputs.to(device)

        logit = model(inputs)

        if preds is None:
            logits = sigmoid(logit)
            pair_idx_list = [pair_idx]
            out_logits = logits.detach().cpu().numpy()
            preds = discretize(logits.detach().cpu().numpy())
            out_labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        else:
            logits = sigmoid(logit)
            pair_idx_list.append(pair_idx)
            out_logits = np.append(
                out_logits, logits.detach().cpu().numpy(), axis=0
            )
            preds = np.append(
                preds, discretize(logits.detach().cpu().numpy()), axis=0
            )
            out_labels_new = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
            out_labels = np.append(out_labels, out_labels_new, axis=0)

    scores = metrics(preds, out_labels, out_logits, "test")

    # Save results
    with open(os.path.join(result_dir, "score_dict.pkl"), "wb") as f:
        pickle.dump(scores, f)

    with open(os.path.join(result_dir, "pred_results.pkl"), "wb") as f:
        pickle.dump([pair_idx_list, preds, out_labels, out_logits], f)

    return scores


def override_data_paths(cfg, eval_species):
    """Override data paths based on eval_species and embedding_type"""
    if eval_species == "human":
        cfg.seq_data = "../data/human/gencode44_utr_gene_unique_cdhit09.csv"
        if cfg.emb_type == "rinalmo":
            cfg.emb_data = "../data/human/gencode44_embedding_rinalmo_whole_ave_cdhit09.pt"
        elif cfg.emb_type == "rnafm":
            cfg.emb_data = "../data/human/gencode44_embedding_ave_cdhit09.pkl"
        else:
            raise ValueError(f"Unknown embedding_type: {cfg.emb_type}. Must be 'rinalmo' or 'rnafm'")

    elif eval_species == "mouse":
        cfg.seq_data = "../data/mouse/gencode_vM33_utr_gene_unique_cdhit09.csv"
        if cfg.emb_type == "rinalmo":
            cfg.emb_data = "../data/mouse/gencode_vM33_embedding_rinalmo_whole_ave_cdhit09.pt"
        elif cfg.emb_type == "rnafm":
            cfg.emb_data = "../data/mouse/gencode_vM33_embedding_ave_cdhit09.pkl"
        else:
            raise ValueError(f"Unknown embedding_type: {cfg.emb_type}. Must be 'rinalmo' or 'rnafm'")
    else:
        raise ValueError(f"Unknown species: {eval_species}")
    
    return cfg

def main(opt: argparse.Namespace):
    """main func"""
    cfg = _parse_config(opt.cfg)

    # Override data paths based on eval_species and embedding_type
    cfg = override_data_paths(cfg, opt.eval_species)

    # For cross-species evaluation, use all data from eval_species
    cfg.cross_species = True
    cfg.conduct_test = True

    # Setup result directory
    model_name = os.path.basename(os.path.dirname(opt.model_path))
    result_dir = os.path.join(
        cfg.result_dir,
        f"{model_name}_eval_{opt.eval_species}"
    )
    os.makedirs(result_dir, exist_ok=True)

    # Setup
    _random_seeds(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dataset for evaluation (use all data)
    if opt.method == "cl":
        dataset_creator = CreateDataset(
            cfg=cfg,
            datasetclass=PairDatasetCL_test,
            datasetclass_test=PairDatasetCL_test,
            kfold=1,
        )
    elif opt.method == "sv":
        dataset_creator = CreateDataset(
            cfg=cfg,
            datasetclass=PairDataset,
            datasetclass_test=PairDataset,
            kfold=1,
        )
    else:
        raise ValueError(f"Unknown method: {opt.method}")

    dataset_dict = dataset_creator.load_dataset()
    # For cross-species evaluation, use test dataset (which contains all data as pos/neg pairs)
    eval_dataloader = DataLoader(
        dataset_dict["test"],
        batch_size=cfg.train.val_bs,
        shuffle=False,
        drop_last=True,
    )

    # Load model
    model = MODEL_DICT[cfg.model.arch](cfg.model)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))
    model = model.to(device)

    print(f"Loaded model from: {opt.model_path}")
    print(f"Evaluating on {opt.eval_species} data")
    print(f"Results will be saved to: {result_dir}")

    # Evaluate
    if opt.method == "cl":
        scores = evaluate_cl(cfg, model, eval_dataloader, device, result_dir)
    elif opt.method == "sv":
        scores = evaluate_sv(cfg, model, eval_dataloader, device, result_dir)

    # Print results
    print("\nEvaluation Results:")
    for k, v in scores.items():
        print(f"{k}: {v}")

    # Save summary as JSON (convert numpy types to native Python types)
    summary = {
        "model_path": opt.model_path,
        "eval_species": opt.eval_species,
        "method": opt.method,
        "scores": {k: float(v) if hasattr(v, 'item') else v for k, v in scores.items()}
    }

    # Save as JSON
    with open(os.path.join(result_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also save scores as CSV for easy viewing
    scores_df = pd.DataFrame([summary["scores"]])
    scores_df.to_csv(os.path.join(result_dir, "scores.csv"), index=False)

    print(f"\nResults saved to: {result_dir}")
    print(f"  - evaluation_summary.json")
    print(f"  - scores.csv")


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
