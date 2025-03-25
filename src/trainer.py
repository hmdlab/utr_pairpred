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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from _model_dict import MODEL_DICT
from dataset import PairDataset, PairDataset_Multi


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


def create_pos_neg_pair(idx_list: np.array, sample_counts: list = None):
    """Create pos/neg pair list

    Args:
        idx_list (np.array): list of idx for the dataset

    Returns:
        _type_: pair list array. dim=(sample_num*2 , 3) [5utr_idx,3utr_idx,label]. label 1->positive, 0->negative
    """
    ## Positive pairs
    pair_list = [[i, i, 1] for i in idx_list]

    ## Negative pairs
    if sample_counts is not None:  # = multi_species==True
        total_sample = 0
        for sample_count in sample_counts:
            species_idx = idx_list[
                (total_sample <= idx_list) & (idx_list < total_sample + sample_count)
            ]
            for utr5_idx in species_idx:
                flg = 1
                while flg:
                    utr3_idx = np.random.choice(species_idx)
                    if utr3_idx != utr5_idx:
                        flg = 0
                pair_list.append([utr5_idx, utr3_idx, 0])

            total_sample += sample_count

    else:
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
    ## split idx for train/val/test
    if cfg.multi_species:
        df_list = [pd.read_csv(path) for path in data_path]
        sample_counts = []
        for df in df_list:
            sample_counts.append(len(df))
        all_idx = np.arange(sum(sample_counts))

    else:
        df = pd.read_csv(data_path, index_col=0)
        print(f"Total sample size:{len(df)}")
        all_idx = np.arange(len(df))

    train_idx, val_idx = train_test_split(all_idx, test_size=test_size)
    val_idx, test_idx = train_test_split(val_idx, test_size=0.5)
    pair_set_dict = {"train": train_idx, "val": val_idx, "test": test_idx}

    ## Create pos/neg pair sets
    for phase, idx_list in pair_set_dict.items():
        if cfg.multi_species:
            pair_set_dict[phase] = create_pos_neg_pair(idx_list, sample_counts)
        else:
            pair_set_dict[phase] = create_pos_neg_pair(idx_list)

    return pair_set_dict


def _discretize(logits, threshold=0.5):
    discretized = logits >= threshold
    return discretized


def metrics(pred: np.array, label: np.array, out_logits: np.array, phase="val") -> dict:
    "evaluation metrics"
    scores = {}
    scores["accuracy"] = accuracy_score(label, pred)
    scores["precision"] = precision_score(label, pred)
    scores["recall"] = recall_score(label, pred)
    scores["f1"] = f1_score(label, pred)
    scores["matthews"] = matthews_corrcoef(label, pred)

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
        scores["auc_prc"] = precision_recall_curve(label, out_logits)
    return scores


def load_split_dataset(cfg: AttrDict) -> dict:
    ## Creating dataset and dataloader
    print("Loading embedding data ...")

    if cfg.multi_species:
        dataset_class = PairDataset_Multi

        seq_embs = []
        for path in cfg.emb_data:
            with open(path, "rb") as f:
                seq_emb = pickle.load(f)
            seq_embs.append(seq_emb)
    else:
        dataset_class = PairDataset
        with open(cfg.emb_data, "rb") as f:
            seq_embs = pickle.load(f)

    print("Successflly loaded embedding data !!!")

    pair_set_dict = create_split_pair_set(cfg, data_path=cfg.seq_data)
    dataset_dict = {}

    for phase, pair_list in pair_set_dict.items():
        print(f"Creating {phase} dataset ...")
        dataset_dict[phase] = dataset_class(seq_embs, pair_list)

    return dataset_dict


class Trainer:
    def __init__(
        self,
        cfg: yaml,
        model: nn.Module,
        dataset_dict: dict,
        loss_fn,
        optimizer,
        device=str,
    ):
        self.cfg = cfg
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
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
        self.test_dataloader = DataLoader(
            dataset_dict["test"],
            batch_size=cfg.train.val_bs,
            shuffle=False,
            drop_last=True,
        )

        self.dataloader_dict = {
            "train": self.train_dataloader,
            "val": self.val_dataloader,
            "test": self.test_dataloader,
        }
        self.best_model = None
        self.best_model_path = os.path.join(self.cfg.result_dir, "best_model.pth")

    def iterate(self, epoch: int, phase: str): # noqa: C901
        if phase == "train":
            self.optimizer.zero_grad()
            self.model.train()
        elif (phase == "val") or (phase == "test"):
            self.model.eval()
            preds = None
        else:
            raise NotImplementedError()

        running_loss = 0.0
        steps = 0

        for data, labels, pair_idx in tqdm(
            self.dataloader_dict[phase], desc=f"Epoch: {epoch}"
        ):
            if "split" in self.cfg.model.arch:
                inputs = (data[0].to(self.device), data[1].to(self.device))

            else:
                inputs = torch.cat(
                    [data[0], data[1]], dim=1
                )  # concat 5UTR and 3UTR embeddings
                if "cnn" in self.cfg.model.arch:
                    inputs = inputs.unsqueeze(dim=-1)
                inputs = inputs.to(self.device)

            logit = self.model(inputs)

            labels = torch.Tensor(labels, dtype=torch.float).to(self.device)
            loss = self.loss_fn(logit.view(-1), labels.view(-1))

            # loss = loss / cfg.train.grad_acc
            if phase == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            running_loss += loss.item()
            steps += 1

            if (phase == "val") or (phase == "test"):
                if preds is None:
                    logits = self.sigmoid(logit)
                    pair_idx_list = [pair_idx]
                    out_logits = logits.detach().cpu().numpy()
                    preds = _discretize(logits.detach().cpu().numpy())
                    out_labels = labels.detach().cpu().numpy()
                else:
                    logits = self.sigmoid(logit)
                    pair_idx_list.append(pair_idx)
                    out_logits = np.append(
                        out_logits, logits.detach().cpu().numpy(), axis=0
                    )
                    preds = np.append(
                        preds, _discretize(logits.detach().cpu().numpy()), axis=0
                    )
                    out_labels = np.append(
                        out_labels, labels.detach().cpu().numpy(), axis=0
                    )

        epoch_loss = running_loss / steps
        wandb.log({f"{phase}/loss": epoch_loss}, step=epoch)
        print(f"Phase:{phase},Epoch {epoch}, loss:{epoch_loss}")
        if (phase == "val") or (phase == "test"):
            scores = metrics(preds, out_labels, out_logits, phase)
            for key, value in scores.items():
                if phase == "val":
                    wandb.log({f"{phase}/{key}": value}, step=epoch)
            return epoch_loss, out_labels, preds, out_logits, scores, pair_idx_list

    def run(self):
        best_loss = float("inf")
        best_epoch = 0
        for epoch in range(1, self.cfg.train.epoch + 1):
            self.iterate(epoch, phase="train")
            if (epoch + 1) % self.cfg.train.val_epoch == 0:
                epoch_loss, _, _, _, _, _ = self.iterate(epoch, phase="val")
                if epoch_loss < best_loss:
                    best_epoch = epoch
                    best_loss = epoch_loss
                    self.best_model = self.model.state_dict()
        print(f"Best epoch:{best_epoch}")
        torch.save(self.best_model, self.best_model_path)

    def test(self):
        self.model.load_state_dict(torch.load(self.best_model_path))
        _, out_labels, preds, out_logits, score, pair_idx_list = self.iterate(
            epoch=0, phase="test"
        )
        with open(os.path.join(self.cfg.result_dir, "pred_results.pkl"), "wb") as f:
            pickle.dump([pair_idx_list, preds, out_labels, out_logits], f)

        with open(os.path.join(self.cfg.result_dir, "score_dict.pkl"), "wb") as f:
            pickle.dump(score, f)


def main(opt: argparse.Namespace):
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
    loss_fn = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))

    trainer = Trainer(cfg, model, dataset_dict, loss_fn, optimizer, device)
    trainer.run()
    trainer.test()


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
