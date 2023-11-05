"""Main trainer code"""
import os
import gc
import argparse
import json
import random
import math
from attrdict import AttrDict
from typing import Tuple, Union
import yaml
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import PairDataset
from .model import PairPredMLP


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


def main(opt: argparse.Namespace):
    cfg = _parse_config(opt.cfg)

    # Setting logger

    wandb.init(
        name=f"{os.path.basename(cfg.result_dir)}",
        project=cfg.wandb_project,
        config=cfg,
    )

    cfg.result_dir = os.path.join("results", cfg.result_dir)
    os.makedirs("results", exist_ok=True)
    os.makedirs(cfg.result_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpu = len(cfg.gpus)

    logger.info(f"loading data from {cfg.data}")
    data, label = load_data(cfg.data, col=cfg.dataset.regions)
    train_data, val_data, train_label, val_label = train_test_split(
        data, label, test_size=0.2, random_state=cfg.seed
    )

    print("Creating train dataset ...")
    train_dataset = UTRDataset_CNN(
        train_data, train_label, phase="train", regions=cfg.dataset.regions
    )
    print("Creating val dataset...")
    val_dataset = UTRDataset_CNN(
        val_data, val_label, phase="val", regions=cfg.dataset.regions
    )

    model = MRNA_CNN(cfg)
    model.to(device)
    if num_gpu > 1:
        model = DP(model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))

    train(cfg, model, train_dataset, val_dataset, optimizer)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
