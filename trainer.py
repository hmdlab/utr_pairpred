"""Main trainer code"""
import os
import argparse
import random
from attrdict import AttrDict
from typing import Tuple, Union
import yaml
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader

from dataset import PairDataset, PairDataset_Multi
from _model_dict import MODEL_DICT


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


def create_pair_set(cfg, data_path):
    if cfg.multi_species:
        df_list = [pd.read_csv(path) for path in data_path]
        all_pair_list = []
        sample_count = 0
        for df in df_list:
            pair_list = [
                [sample_count + i, sample_count + i, 1] for i in range(len(df))
            ]  # dim=(sample_num*2 , 3) [5utr_idx,3utr_idx,label]. label 1->positive, 0->negative
            ## add negative pairs
            for i in range(len(df)):
                flg = 1
                while flg:
                    utr3_idx = random.randint(0, len(df) - 1)
                    if utr3_idx != i:
                        flg = 0
                pair_list.append([i + sample_count, utr3_idx + sample_count, 0])
            assert len(pair_list) == len(df) * 2
            all_pair_list.extend(pair_list)
            sample_count += len(df)

        all_pair_list = np.array(all_pair_list)
        return all_pair_list

    else:
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


def _discretize(logits, threshold=0.5):
    discretized = logits >= threshold
    return discretized


def metrics(pred: np.array, label: np.array) -> dict:
    scores = dict()
    scores["accuracy"] = accuracy_score(pred, label)
    scores["precision"] = precision_score(pred, label)
    scores["recall"] = recall_score(pred, label)
    scores["f1"] = f1_score(pred, label)

    return scores


def load_dataset(cfg: AttrDict) -> (Dataset, Dataset):
    ## Creating dataset and dataloader
    if cfg.multi_species:
        dataset_class = PairDataset_Multi
    else:
        dataset_class = PairDataset

    data = create_pair_set(cfg, data_path=cfg.seq_data)
    print(f"Total sample size:{len(data)}")

    train_data, val_data = train_test_split(data, test_size=0.2, random_state=cfg.seed)

    print("Creating train dataset ...")
    train_dataset = dataset_class(train_data, seq_emb_path=cfg.emb_data)
    print("Creating val dataset ...")
    val_dataset = dataset_class(val_data, seq_emb_path=cfg.emb_data)

    return (train_dataset, val_dataset)


def val(
    cfg: yaml, model: nn.Module, val_dataset: Dataset, loss_fn, device: str, epoch: int
) -> dict:
    val_dataloader = DataLoader(
        val_dataset, batch_size=cfg.train.val_bs, shuffle=False, drop_last=True
    )
    model.eval()
    sigmoid = nn.Sigmoid()
    running_loss = 0
    eval_steps = 0
    preds = None

    for data, labels in val_dataloader:
        if "split" in cfg.model.arch:
            inputs = (data[0].to(device), data[1].to(device))

        else:
            inputs = torch.cat(
                [data[0], data[1]], dim=1
            )  # concat 5UTR and 3UTR embeddings
            if "cnn" in cfg.model.arch:
                inputs = inputs.unsqueeze(dim=-1)
            inputs = inputs.to(device)

        logits = model(inputs)

        labels = torch.tensor(labels, dtype=torch.float).to(device)
        loss = loss_fn(logits.view(-1), labels.view(-1))
        if len(cfg.gpus) > 1:
            loss = loss.mean()
        running_loss += loss.item()
        eval_steps += 1

        if preds is None:
            logits = sigmoid(logits)
            preds = _discretize(logits.detach().cpu().numpy())
            out_labels = labels.detach().cpu().numpy()
        else:
            logits = sigmoid(logits)
            preds = np.append(preds, _discretize(logits.detach().cpu().numpy()), axis=0)
            out_labels = np.append(out_labels, labels.detach().cpu().numpy(), axis=0)

    running_loss = running_loss / eval_steps
    print(f"Eval loss:{running_loss}")
    wandb.log({"val/loss": running_loss}, step=epoch)
    scores = metrics(preds, out_labels)
    return scores


def train(
    cfg: yaml,
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    loss_fn,
    optimizer,
    device: str,
) -> None:
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.train.train_bs, shuffle=True, drop_last=True
    )
    for epoch in range(cfg.train.epoch):
        optimizer.zero_grad()
        model.train()
        running_loss = 0.0
        train_steps = 0
        for data, labels in tqdm(train_dataloader, desc=f"Epoch: {epoch}"):
            if "split" in cfg.model.arch:
                inputs = (data[0].to(device), data[1].to(device))

            else:
                inputs = torch.cat(
                    [data[0], data[1]], dim=1
                )  # concat 5UTR and 3UTR embeddings
                if "cnn" in cfg.model.arch:
                    inputs = inputs.unsqueeze(dim=-1)
                inputs = inputs.to(device)

            logit = model(inputs)

            labels = torch.tensor(labels, dtype=torch.float).to(device)
            loss = loss_fn(logit.view(-1), labels.view(-1))

            # loss = loss / cfg.train.grad_acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            train_steps += 1

        epoch_loss = running_loss / train_steps

        wandb.log(
            {
                "train/loss": epoch_loss,
                "learning rate": optimizer.param_groups[0]["lr"],
            },
            step=epoch,
        )

        print(f"Epoch {epoch} loss:{epoch_loss}")
        if (epoch + 1) % cfg.train.val_epoch == 0:
            scores = val(cfg, model, val_dataset, loss_fn, device, epoch)
            for key, value in scores.items():
                wandb.log({f"val/{key}": value}, step=epoch)
                print(f"{key}:{value:.4f}")


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

    train_dataset, val_dataset = load_dataset(cfg)
    model = MODEL_DICT[cfg.model.arch](cfg.model)

    if "mlp" in cfg.model.arch:
        optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))
    elif "cnn" in cfg.model.arch:
        optimizer = Adam(model.parameters(), lr=float(cfg.train.lr))

    model.to(device)

    loss_fn = BCEWithLogitsLoss()

    train(cfg, model, train_dataset, val_dataset, loss_fn, optimizer, device)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
