"""Dataset class"""
import numpy as np
import pickle
from typing import Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, data: np.array, seq_emb_path: str):
        super().__init__()
        self.pair_idx = (
            data  # Embedding idx. dim=[sample_num,3]=(utr5idx,utr3idx,label)
        )
        with open(seq_emb_path, "rb") as f:
            self.seq_emb = pickle.load(f)
        self.utr5emb = torch.stack(list(np.array(self.seq_emb)[:, 0]))
        self.utr3emb = torch.stack(list(np.array(self.seq_emb)[:, 1]))

    def __getitem__(self, idx) -> Union[list, int]:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        return embs, pair_data[2]  # label

    def __len__(self):
        return len(self.pair_idx)


class PairDataset_Multi(Dataset):
    def __init__(self, data: np.array, seq_emb_path: list):
        super().__init__()
        self.pair_idx = (
            data  # Embedding idx. dim=[sample_num,3]=(utr5idx,utr3idx,label)
        )
        self.utr5emb_list = list()
        self.utr3emb_list = list()
        for path in seq_emb_path:
            with open(path, "rb") as f:
                self.seq_emb = pickle.load(f)
                self.utr5emb_list.append(
                    torch.stack(list(np.array(self.seq_emb)[:, 0]))
                )
                self.utr3emb_list.append(
                    torch.stack(list(np.array(self.seq_emb)[:, 1]))
                )

        self.utr5emb = torch.cat(self.utr5emb_list)
        self.utr3emb = torch.cat(self.utr3emb_list)

    def __getitem__(self, idx) -> Union[list, int]:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        return embs, pair_data[2]  # label

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
