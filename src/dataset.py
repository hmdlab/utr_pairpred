"""Dataset class"""
import pickle
import numpy as np
import pandas as pd
from attrdict import AttrDict
from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, seq_emb: list, data: np.array):
        super().__init__()
        self.seq_emb = seq_emb
        self.pair_idx = (
            data  # Embedding idx. dim=[sample_num,3]=(utr5idx,utr3idx,label)
        )
        self.utr5emb = np.stack(list(np.array(self.seq_emb)[:, 0]))
        self.utr3emb = np.stack(list(np.array(self.seq_emb)[:, 1]))

    def __getitem__(self, idx) -> Union[list, int]:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        return embs, pair_data[2], pair_data  # label

    def __len__(self):
        return len(self.pair_idx)


class PairDataset_Multi(Dataset):
    def __init__(self, data: np.array, seq_embs):
        super().__init__()
        self.seq_embs = seq_embs
        self.pair_idx = (
            data  # Embedding idx. dim=[sample_num,3]=(utr5idx,utr3idx,label)
        )
        self.utr5emb_list = list()
        self.utr3emb_list = list()

        for seq_emb in self.seq_embs:
            self.utr5emb_list.append(torch.stack(list(np.array(seq_emb)[:, 0])))
            self.utr3emb_list.append(torch.stack(list(np.array(seq_emb)[:, 1])))

        self.utr5emb = torch.cat(self.utr5emb_list)
        self.utr3emb = torch.cat(self.utr3emb_list)

    def __getitem__(self, idx) -> Union[list, int]:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        return embs, pair_data[2], pair_data  # label

    def __len__(self):
        return len(self.pair_idx)


class PairDatasetRF:
    def __init__(self, data: np.array, seq_emb_path: str):
        self.pair_list = data
        with open(seq_emb_path, "rb") as f:
            self.seq_emb = pickle.load(f)

        self.utr5emb = np.stack(list(np.array(self.seq_emb)[:, 0]))
        self.utr3emb = np.stack(list(np.array(self.seq_emb)[:, 1]))

    def get(self) -> (list, list):
        embeddings = []
        labels = []
        for pair_data in self.pair_list:
            emb5 = self.utr5emb[pair_data[0]]
            emb3 = self.utr3emb[pair_data[1]]
            emb = np.concatenate([emb5, emb3])
            embeddings.append(emb)
            labels.append(pair_data[2])

        return (embeddings, labels)

    def __len__(self):
        return len(self.pair_list)


class PairDatasetRF_feature:
    def __init__(self, data: np.array, cfg: AttrDict):
        self.pair_list = data  # [utr5_idx,utr3_idx,label]

        self.utr5feature = pd.read_csv(cfg.utr5_feature_path, index_col=0)
        self.utr3feature = pd.read_csv(cfg.utr3_feature_path, index_col=0)

    def get(self) -> (list, list):
        utr5_idx = self.pair_list[:, 0]
        utr3_idx = self.pair_list[:, 1]
        labels = self.pair_list[:, 2]

        utr5feature = self.utr5feature.iloc[utr5_idx]
        utr3feature = self.utr3feature.iloc[utr3_idx]
        features = np.concatenate([utr5feature, utr3feature], axis=1)

        return (features, labels)

    def __len__(self):
        return len(self.pair_list)


class PairDatasetCL(Dataset):
    """Dataset class for pairpred Contrastive Learning task"""

    def __init__(self, seq_emb: list, pair_idx: np.ndarray):
        super().__init__()
        self.seq_emb = seq_emb
        self.pair_idx = pair_idx

        self.utr5emb = np.stack(list(np.array(self.seq_emb)[:, 0]))
        self.utr3emb = np.stack(list(np.array(self.seq_emb)[:, 1]))

    def __getitem__(self, idx) -> list:
        embs = [self.utr5emb[self.pair_idx[idx]], self.utr3emb[self.pair_idx[idx]]]
        return embs

    def __len__(self):
        return len(self.pair_idx)


class PairDatasetCL_Multi(Dataset):
    """Dataset class for pairpred Contrastive Learning task"""

    def __init__(self, seq_embs: list, pair_idx: np.ndarray):
        super().__init__()
        self.seq_embs = seq_embs
        self.pair_idx = pair_idx

        self.utr5emb_list = list()
        self.utr3emb_list = list()

        for seq_emb in self.seq_embs:
            self.utr5emb_list.append(torch.stack(list(np.array(seq_emb)[:, 0])))
            self.utr3emb_list.append(torch.stack(list(np.array(seq_emb)[:, 1])))

        self.utr5emb = torch.cat(self.utr5emb_list)
        self.utr3emb = torch.cat(self.utr3emb_list)

    def __getitem__(self, idx) -> list:
        embs = [self.utr5emb[self.pair_idx[idx]], self.utr3emb[self.pair_idx[idx]]]
        return embs

    def __len__(self):
        return len(self.pair_idx)


class PairDatasetCL_test(PairDatasetCL):
    def __init__(self, seq_emb: list, pair_idx: np.ndarray):
        super().__init__(seq_emb, pair_idx)

    def __getitem__(self, idx) -> list:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        label = pair_data[2]
        return embs, label, pair_data


class PairDatasetCL_Multi_test(PairDatasetCL_Multi):
    def __init__(self, seq_embs: list, pair_idx: np.ndarray):
        super().__init__(seq_embs, pair_idx)

    def __getitem__(self, idx) -> list:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        label = pair_data[2]
        return embs, label, pair_data
