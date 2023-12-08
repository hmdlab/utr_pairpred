"""Main trainer code"""
import os
import argparse
import random
from attrdict import AttrDict
from typing import Tuple, Union
import yaml
import hydra
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn


from dataset import PairDatasetRF


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", required=True, type=str, help="path to config yaml")
    opt = args.parse_args()
    return opt


def _parse_config(cfg_path: str) -> dict:
    """Load and return yaml format config

    Args:
        cfg_path (str): yaml config file path

    Returns:
        config (dict): config dict
    """

    with open(cfg_path) as f:
        config = yaml.safe_load(f.read())
    config = AttrDict(config)
    return config


def _random_seeds(seed=0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_pair_set(data_path):
    df = pd.read_csv(data_path, index_col=0)
    pair_list = [
        [i, i, 1] for i in range(len(df))
    ]  # dim=(sample_num*2 , 3) [5utr_idx,3utr_idx,label]. label 1->positive, 0->negative
    ## add negative pairs
    for i in range(len(df)):
        flg = 1
        while flg:
            utr3_idx = random.randint(0, len(df) - 1)
            if utr3_idx != i:
                flg = 0
        pair_list.append([i, utr3_idx, 0])
    assert len(pair_list) == len(df) * 2
    pair_list = np.array(pair_list)
    return pair_list


def metrics(pred: np.array, label: np.array) -> dict:
    scores = dict()
    scores["accuracy"] = accuracy_score(pred, label)
    scores["precision"] = precision_score(pred, label)
    scores["recall"] = recall_score(pred, label)
    scores["f1"] = f1_score(pred, label)

    return scores


def main(opt: argparse.Namespace):
    cfg = _parse_config(opt.cfg)

    ## Setup section
    _random_seeds(cfg.seed)

    cfg.result_dir = os.path.join("results", cfg.result_dir)
    os.makedirs("results", exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)

    ## Creating dataset and dataloader
    pair_list = create_pair_set(data_path=cfg.seq_data)
    train_data, val_data = train_test_split(
        pair_list, test_size=0.2, random_state=cfg.seed
    )

    print("Creating train dataset ...")
    train_dataset = PairDatasetRF(train_data, seq_emb_path=cfg.emb_data)
    X_train, y_train = train_dataset.get()
    print("Creating val dataset ...")
    val_dataset = PairDatasetRF(val_data, seq_emb_path=cfg.emb_data)
    X_val, y_val = val_dataset.get()

    print("Model fitting ...")
    model = RandomForestClassifier(random_state=cfg.seed)
    model.fit(X_train, y_train)

    print("Predicting ...")
    pred = model.predict(X_val)
    scores = metrics(pred, y_val)
    print("Prediction finished !!!\n Metrics")
    for k, v in scores.items():
        print(f"{k}:{v:.4f}")


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
