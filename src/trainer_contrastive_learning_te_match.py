"""Main trainer code"""
import os
import argparse
import random
import pickle
from attrdict import AttrDict
import wandb
import yaml
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from dataset import PairDatasetCL, PairDatasetCL_test
from _model_dict import MODEL_DICT
from utils import NpairLoss


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", required=True, type=str, help="path to config yaml")
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


def create_pos_neg_pair(idx_list: np.array):
    """Create pos/neg pair list

    Args:
        idx_list (np.array): list of idx for the dataset

    Returns:
        _type_: pair list array. dim=(sample_num*2 , 3) [5utr_idx,3utr_idx,label]. label 1->positive, 0->negative
    """
    ## add positive pairs
    pair_list = [[i, i, 1] for i in idx_list]
    ## add negative pairs
    for utr5_idx in idx_list:
        flg = 1
        while flg:
            utr3_idx = np.random.choice(idx_list)
            if utr3_idx != utr5_idx:
                flg = 0
        pair_list.append([utr5_idx, utr3_idx, 0])

    return pair_list


def create_split_pair_set(cfg, data_path, test_size=0.2) -> dict:
    """Split all sequences into for training and evaluating (testing).
       Create pos/neg pairs within splited sequence ids.

    Args:
        cfg (_type_): _description_
        data_path (_type_): _description_
        test_size (float, optional): _description_. Defaults to 0.2.

    Returns:
        dict : dict of pair_idx_list for each phase.
    """
    df = pd.read_csv(data_path, index_col=0)
    print(f"Total sample size:{len(df)}")
    all_idx = np.arange(len(df))
    with open(cfg.te_idx, "rb") as f:
        test_idx = pickle.load(f)
    remain_idx = list(set(all_idx) - set(test_idx))
    train_idx, val_idx = train_test_split(remain_idx, test_size=0.1)
    pair_set_dict = {"train": train_idx, "val": val_idx}
    pair_set_dict["test"] = create_pos_neg_pair(test_idx)

    return pair_set_dict


def _discretize(logits, threshold=0.5):
    discretized = logits >= threshold
    return discretized


def metrics(pred: np.array, label: np.array, out_logits: np.array, phase="val") -> dict:
    """evaluation metrics"""
    scores = dict()
    scores["accuracy"] = accuracy_score(label, pred)
    scores["precision"] = precision_score(label, pred)
    scores["recall"] = recall_score(label, pred)
    scores["f1"] = f1_score(label, pred)

    if phase == "test":
        conf_mat = confusion_matrix(label, pred)
        scores["confusion_matrix"] = conf_mat
        fpr_all, tpr_all, thresholds_all = roc_curve(
            label, out_logits, drop_intermediate=False
        )
        scores["fpr_all"] = fpr_all
        scores["tpr_all"] = tpr_all
        scores["roc_thresh"] = thresholds_all
        scores["auc_roc"] = roc_auc_score(label, out_logits)
    return scores


def load_split_dataset(cfg: AttrDict) -> dict:
    ## Creating dataset and dataloader

    dataset_class = PairDatasetCL
    dataset_class_test = PairDatasetCL_test
    pair_set_dict = create_split_pair_set(cfg, data_path=cfg.seq_data)
    dataset_dict = dict()

    print("Loading embedding data ...")
    with open(cfg.emb_data, "rb") as f:
        seq_emb = pickle.load(f)
    print("Successflly loaded embedding data !!!")

    for phase, pair_list in pair_set_dict.items():
        print(f"Creating {phase} dataset ...")
        print(f"{phase},{len(pair_list)}")
        if phase == "test":
            dataset_dict[phase] = dataset_class_test(seq_emb, pair_list)
        else:
            dataset_dict[phase] = dataset_class(seq_emb, pair_list)

    return dataset_dict


class Trainer:
    def __init__(
        self,
        cfg: yaml,
        model: nn.Module,
        dataset_dict: dict,
        loss_fn,
        optimizer,
        scheduler,
        device=str,
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.sigmoid = nn.Sigmoid()

        self.train_dataloader = DataLoader(
            dataset_dict["train"],
            batch_size=self.cfg.train.train_bs,
            shuffle=True,
            drop_last=True,
        )

        self.val_dataloader = DataLoader(
            dataset_dict["val"],
            batch_size=cfg.train.val_bs,
            shuffle=False,
            drop_last=True,
        )
        self.dataloader_dict = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
        }
        if self.cfg.conduct_test:
            self.test_dataloader = DataLoader(
                dataset_dict["test"],
                batch_size=cfg.train.val_bs,
                shuffle=False,
                drop_last=True,
            )
            self.dataloader_dict["test"] = self.test_dataloader

        self.best_model = None
        self.best_model_path = os.path.join(self.cfg.result_dir, "best_model.pth")

    def iterate(self, epoch: int, phase: str):
        """Main iteration"""
        if phase == "train":
            self.optimizer.zero_grad()
            self.model.train()
        elif (phase == "val") or (phase == "test"):
            self.model.eval()
        else:
            raise NotImplementedError()

        running_loss = 0.0
        steps = 0

        for embs in tqdm(self.dataloader_dict[phase], desc=f"Epoch: {epoch}"):
            inputs = (embs[0].to(self.device), embs[1].to(self.device))
            logits = self.model(inputs)

            loss = self.loss_fn(
                logits[0], logits[1]
            )  # (logits_per_utr5,logits_per_utr3)

            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()
            steps += 1
        if phase == "train":
            self.scheduler.step()

        epoch_loss = running_loss / steps
        lr = self.scheduler.get_last_lr()
        wandb.log(
            {
                f"{phase}/loss": epoch_loss,
                f"{phase}/lr": lr[0],
            },
            step=epoch,
        )
        print(f"Phase:{phase},Epoch {epoch}, loss:{epoch_loss}")

        return epoch_loss

    def test_iteration(self, phase="test"):
        """iteration for test. this function returns scores and raw pred results."""
        self.model.load_state_dict(torch.load(self.best_model_path))
        self.model.eval()
        preds = None

        for embs, labels, pair_idx in tqdm(self.dataloader_dict[phase]):
            inputs = (embs[0].to(self.device), embs[1].to(self.device))
            logits = self.model.predict(inputs)
            cos_sim = logits[0].diag()  # get diagonal values
            logits_sigmoid = F.sigmoid(cos_sim)

            if preds is None:
                pair_idx_list = [pair_idx]
                out_logits = logits_sigmoid.detach().cpu().numpy()
                out_cos_sim = cos_sim.detach().cpu().numpy()
                preds = _discretize(logits_sigmoid.detach().cpu().numpy())
                out_labels = labels

            else:
                pair_idx_list.append(pair_idx)
                out_logits = np.append(
                    out_logits, logits_sigmoid.detach().cpu().numpy(), axis=0
                )
                out_cos_sim = np.append(out_cos_sim, cos_sim.detach().cpu().numpy())
                preds = np.append(
                    preds, _discretize(logits_sigmoid.detach().cpu().numpy())
                )
                out_labels = np.append(out_labels, labels)

        scores = metrics(preds, out_labels, out_logits, phase)

        return scores, (out_cos_sim, out_logits, pair_idx_list)

    def run(self):
        """General controling method"""
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, self.cfg.train.epoch + 1):
            self.iterate(epoch, phase="train")
            if epoch % self.cfg.train.val_epoch == 0:
                epoch_loss = self.iterate(epoch, phase="val")
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    self.best_model = self.model.state_dict()
        print(f"Best epoch:{best_epoch}")
        torch.save(self.best_model, self.best_model_path)

    def test(self):
        scores, pred_results = self.test_iteration(phase="test")
        for k, v in scores.items():
            print(f"{k}:{v}")

        with open(os.path.join(self.cfg.result_dir, "score_dict.pkl"), "wb") as f:
            pickle.dump(scores, f)

        with open(os.path.join(self.cfg.result_dir, "pred_results.pkl"), "wb") as f:
            pickle.dump(pred_results, f)


def main(opt: argparse.Namespace):
    """main func"""
    cfg = _parse_config(opt.cfg)

    ## Setup section
    _random_seeds(cfg.seed)

    wandb.init(
        project=cfg.wandb_project,
        group=cfg.wandb_group,
        name=f"{os.path.basename(cfg.result_dir)}",
        config=cfg,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg.result_dir, exist_ok=True)
    # dataset_dict = load_dataset(cfg)
    dataset_dict = load_split_dataset(cfg)
    model = MODEL_DICT[cfg.model.arch](cfg.model)
    model.to(device)
    loss_fn = NpairLoss(device)
    optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))
    scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

    trainer = Trainer(cfg, model, dataset_dict, loss_fn, optimizer, scheduler, device)
    trainer.run()
    if cfg.conduct_test:
        trainer.test()


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
