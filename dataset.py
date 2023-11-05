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
