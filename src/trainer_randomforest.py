"""Main trainer code"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import yaml
from attrdict import AttrDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from dataset import PairDatasetRF, PairDatasetRF_feature


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


def metrics(label: np.array,pred: np.array,  pred_scores:np.array=None) -> dict:
    scores = {}
    scores["accuracy"] = accuracy_score(pred, label)
    scores["precision"] = precision_score(pred, label)
    scores["recall"] = recall_score(pred, label)
    scores["f1"] = f1_score(pred, label)

    if pred_scores is not None:
        scores["auc_roc"] = roc_auc_score(label,pred_scores)
        scores["auc_prc"] = average_precision_score(label, pred_scores)

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
    if cfg.input_type == "emb":
        print("Creating train dataset ...")
        train_dataset = PairDatasetRF(train_data, seq_emb_path=cfg.emb_data)
        X_train, y_train = train_dataset.get()
        print("Creating val dataset ...")
        val_dataset = PairDatasetRF(val_data, seq_emb_path=cfg.emb_data)
        X_val, y_val = val_dataset.get()

    elif cfg.input_type == "feature":
        print("Creating train dataset ...")
        train_dataset = PairDatasetRF_feature(train_data, cfg)
        X_train, y_train = train_dataset.get()
        print("Creating val dataset ...")
        val_dataset = PairDatasetRF_feature(val_data, cfg)
        X_val, y_val = val_dataset.get()

    print("Model fitting ...")
    if cfg.model.arch == "rf":
        model = RandomForestClassifier(random_state=cfg.seed)
        model.fit(X_train, y_train)
    else:
        raise NotImplementedError()

    print("Predicting ...")
    pred = model.predict(X_val)
    pred_scores = model.predict_proba(X_val)[:, 0]
    scores = metrics(y_val,pred,pred_scores)
    print("Prediction finished !!!\n Metrics")
    for k, v in scores.items():
        print(f"{k}:{v:.4f}")


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
