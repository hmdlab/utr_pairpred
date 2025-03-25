"""Main trainer code"""

import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import yaml
from attrdict import AttrDict
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from _model_dict import MODEL_DICT
from base_trainer import Trainer
from dataset import CreateDataset, PairDatasetCL, PairDatasetCL_test
from utils import NpairLoss


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", required=True, type=str, help="path to config yaml")
    args.add_argument("--seed", type=int, help="seed", default=0)
    args = args.parse_args()
    return args


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


class Runner:
    def __init__(self, cfg: AttrDict, trainer: Trainer) -> None:
        self.cfg = cfg
        self.trainer = trainer
        self.kfold = self.cfg.kfold

    def run(self):
        """General controling method"""
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, self.cfg.train.epoch + 1):
            self.trainer.iterate(epoch, phase="train")
            if epoch % self.cfg.train.val_epoch == 0:
                epoch_loss = self.trainer.iterate(epoch, phase="val")
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    self.trainer.best_model = self.trainer.model.state_dict()
                    best_loss = epoch_loss
        print(f"Best epoch:{best_epoch}")
        torch.save(self.trainer.best_model, self.trainer.best_model_path)

    def test(self):
        """Run test iteration and save results"""
        scores, pred_results = self.trainer.test_iteration(phase="test")
        for k, v in scores.items():
            print(f"{k}:{v}")

        with open(os.path.join(self.cfg.result_dir, "score_dict.pkl"), "wb") as f:
            pickle.dump(scores, f)

        with open(os.path.join(self.cfg.result_dir, "pred_results.pkl"), "wb") as f:
            pickle.dump(pred_results, f)

    def test_kfold(self, k: int):
        """Run test iteration and save results"""
        scores, pred_results = self.trainer.test_iteration(phase="test")
        for key, values in scores.items():
            print(f"{key}:{values}")

        with open(os.path.join(self.cfg.result_dir, "score_dict.pkl"), "wb") as f:
            pickle.dump(scores, f)

        with open(os.path.join(self.cfg.result_dir, "pred_results.pkl"), "wb") as f:
            pickle.dump(pred_results, f)

        return scores


def main(opt: argparse.Namespace):
    """main func"""
    cfg = _parse_config(opt.cfg)
    cfg.seed = opt.seed
    cfg.result_dir = cfg.result_dir + f"_seed{cfg.seed}/"

    ## Setup section
    _random_seeds(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.result_dir, exist_ok=True)
    dataset_creator = CreateDataset(
        cfg=cfg,
        datasetclass=PairDatasetCL,
        datasetclass_test=PairDatasetCL_test,
        kfold=cfg.kfold,
    )

    if cfg.kfold == 1:
        model = MODEL_DICT[cfg.model.arch](cfg.model)
        model = model.to(device)
        loss_fn = NpairLoss(device)
        optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))
        scheduler = MultiStepLR(optimizer, milestones=[60], gamma=0.1)
        dataset_dict = dataset_creator.load_dataset()
        trainer = Trainer(
            cfg, model, dataset_dict, loss_fn, optimizer, scheduler, device
        )
        runner = Runner(cfg, trainer)
        runner.run()
        if cfg.conduct_test:
            runner.test()

    else:
        score_of_score_dic = {}
        for k in range(cfg.kfold):
            model = MODEL_DICT[cfg.model.arch](cfg.model)
            model = model.to(device)
            loss_fn = NpairLoss(device)
            optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))
            scheduler = MultiStepLR(optimizer, milestones=[60], gamma=0.1)
            dataset_dict = dataset_creator.load_dataset_kfold(k=k)
            cfg.result_dir = os.path.join(
                "/".join(cfg.result_dir.split("/")[:-1]), str(k)
            )
            os.makedirs(cfg.result_dir, exist_ok=True)
            trainer = Trainer(
                cfg, model, dataset_dict, loss_fn, optimizer, scheduler, device
            )
            runner = Runner(cfg, trainer)
            runner.run()
            scores = runner.test_kfold(k=k)
            score_of_score_dic[k] = scores
        res_name = os.path.join(
            "/".join(cfg.result_dir.split("/")[:-1]), "score_of_score_dic.csv"
        )
        res_df = pd.DataFrame(score_of_score_dic)
        res_df.to_csv(res_name)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
