"""Main trainer code"""
import os
from attrdict import AttrDict
import wandb
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import metrics, discretize


class Trainer:
    def __init__(
        self,
        cfg: AttrDict,
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

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            mode=cfg.wandb_mode,
            name=f"{os.path.basename(cfg.result_dir)}",
            config=cfg,
        )

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
            self.scheduler.step(epoch)

        epoch_loss = running_loss / steps

        wandb.log(
            {
                f"{phase}/loss": epoch_loss,
                f"{phase}/lr": self.optimizer.param_groups[0]["lr"],
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
            logits, _ = self.model.predict(inputs)
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

        scores = metrics(preds, out_labels, out_logits, phase)

        return scores, (out_cos_sim, out_logits, pair_idx_list)


class TrainerMLP(Trainer):
    def __init__(
        self,
        cfg: AttrDict,
        model: nn.Module,
        dataset_dict: dict,
        loss_fn,
        optimizer,
        scheduler,
        device=str,
    ):
        super(TrainerMLP, self).__init__(
            cfg, model, dataset_dict, loss_fn, optimizer, scheduler, device
        )

    def iterate(self, epoch: int, phase: str):
        if phase == "train":
            self.optimizer.zero_grad()
            self.model.train()
        elif (phase == "val") or (phase == "test"):
            self.model.eval()
            preds = None
        else:
            NotImplementedError

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

            labels = torch.tensor(labels, dtype=torch.float).to(self.device)
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
                    preds = discretize(logits.detach().cpu().numpy())
                    out_labels = labels.detach().cpu().numpy()
                else:
                    logits = self.sigmoid(logit)
                    pair_idx_list.append(pair_idx)
                    out_logits = np.append(
                        out_logits, logits.detach().cpu().numpy(), axis=0
                    )
                    preds = np.append(
                        preds, discretize(logits.detach().cpu().numpy()), axis=0
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
