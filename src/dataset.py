"""Dataset class"""

import pickle
from typing import Union
import numpy as np
import pandas as pd
from attrdict import AttrDict
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class CreateDataset:
    """dataset creating class"""

    def __init__(
        self,
        cfg: AttrDict,
        datasetclass: Dataset,
        datasetclass_test: Dataset = None,
        kfold=None,
    ) -> None:
        self.cfg = cfg
        self.seq_emb = self._load_emb(cfg.emb_data)

        self.dataset_class = datasetclass
        self.dataset_class_test = datasetclass_test

        self.df = pd.read_csv(self.cfg.seq_data, index_col=0)
        self.all_idx = np.arange(len(self.df))
        np.random.shuffle(self.all_idx)
        self.kfold = kfold
        if self.kfold is not None:
            self.kfold_set = self.kfold_split_list(self.all_idx, k=self.kfold)
            # kfold_set is each index list for using test datasets

    def _load_emb(self, emb_path: str) -> np.ndarray:
        print("Loading embedding data ...")
        if emb_path.endswith(".pkl"):
            with open(emb_path, "rb") as f:
                seq_emb = pickle.load(f)
        elif emb_path.endswith(".pt"):
            seq_emb = torch.load(emb_path)
        else:
            raise NotImplementedError

        print("Successflly loaded embedding data !!!")
        return seq_emb

    def create_pos_neg_pair(self, idx_list: np.ndarray, sample_counts: list = None):
        """Create pos/neg pair list

        Args:
            idx_list (np.ndarray): list of idx for the dataset

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
                    (total_sample <= idx_list)
                    & (idx_list < total_sample + sample_count)
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

    def kfold_split_list(self, input_list: list, k: int) -> list:
        """

        Args:
            input_list (list): list of idx for which want you divide
            k (int): K-fold.

        Returns:
            list: Devided idx list.
        """
        n = len(input_list)
        avg = n // k
        remainder = n % k
        kfold_set = []
        start = 0

        for i in range(k):
            end = start + avg + (1 if i < remainder else 0)
            kfold_set.append(input_list[start:end])
            start = end

        return kfold_set

    def create_split_pair_set_mlp(self, test_size=0.2) -> dict:
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
        if self.cfg.multi_species:
            raise NotImplementedError()
            df_list = [pd.read_csv(path) for path in data_path]
            sample_counts = []
            for df in df_list:
                sample_counts.append(len(df))
            all_idx = np.arange(sum(sample_counts))

        train_idx, val_idx = train_test_split(self.all_idx, test_size=test_size)
        val_idx, test_idx = train_test_split(val_idx, test_size=0.5)
        pair_set_dict = {"train": train_idx, "val": val_idx, "test": test_idx}

        ## Create pos/neg pair sets
        for phase, idx_list in pair_set_dict.items():
            if self.cfg.multi_species:
                pair_set_dict[phase] = self.create_pos_neg_pair(idx_list)
            else:
                pair_set_dict[phase] = self.create_pos_neg_pair(idx_list)

        return pair_set_dict

    def create_split_pair_set(self, test_size: float = 0.2) -> dict:
        """Split all sequences into for training and evaluating (testing).
        Create pos/neg pairs within splited sequence ids.

        Args:
            cfg (_type_): _description_
            data_path (_type_): _description_
            test_size (float, optional): _description_. Defaults to 0.2.

        Returns:
            dict : dict of pair_idx_list for each phase.
        """

        train_idx, val_idx = train_test_split(self.all_idx, test_size=test_size)
        pair_set_dict = {"train": train_idx, "val": val_idx}
        if self.cfg.conduct_test:
            val_idx, test_idx = train_test_split(val_idx, test_size=0.5)
            pair_set_dict["val"] = val_idx
            pair_set_dict["test"] = self.create_pos_neg_pair(test_idx)

        return pair_set_dict

    def load_dataset(self) -> dict:
        """Creating dataset and dataloader"""
        if "mlp" in self.cfg.model.arch:
            pair_set_dict: dict = self.create_split_pair_set_mlp()
        elif "contrastive" in self.cfg.model.arch:
            pair_set_dict: dict = self.create_split_pair_set()
        dataset_dict = dict()

        for phase, pair_list in pair_set_dict.items():
            print(f"Creating {phase} dataset ...")
            print(f"{phase},{len(pair_list)}")
            if phase == "test" and (self.dataset_class_test is not None):
                dataset_dict[phase] = self.dataset_class_test(self.seq_emb, pair_list)
            else:
                dataset_dict[phase] = self.dataset_class(self.seq_emb, pair_list)

        return dataset_dict

    def load_dataset_kfold(self, k: int) -> dict:
        """Creating dataset and dataloader for kfold cross val."""

        test_idx = self.kfold_set[k]
        remain_idx = [x for x in self.all_idx if x not in test_idx]
        test_idx = self.create_pos_neg_pair(test_idx)
        train_idx, val_idx = train_test_split(remain_idx, test_size=0.1)
        if "mlp" in self.cfg.model.arch:
            train_idx = self.create_pos_neg_pair(train_idx)
            val_idx = self.create_pos_neg_pair(val_idx)

        pair_set_dict = {"train": train_idx, "val": val_idx, "test": test_idx}

        dataset_dict = dict()

        for phase, pair_list in pair_set_dict.items():
            print(f"Creating {phase} dataset ...")
            print(f"{phase},{len(pair_list)}")
            if phase == "test" and (self.dataset_class_test is not None):
                dataset_dict[phase] = self.dataset_class_test(self.seq_emb, pair_list)
            else:
                dataset_dict[phase] = self.dataset_class(self.seq_emb, pair_list)

        return dataset_dict


class PairDataset(Dataset):
    def __init__(self, seq_emb: list, data: np.array):
        super().__init__()
        self.seq_emb = seq_emb
        self.pair_idx = (
            data  # Embedding idx. dim=[sample_num,3]=(utr5idx,utr3idx,label)
        )

        if type(self.seq_emb) == list:
            self.utr5emb = torch.stack([e[0] for e in self.seq_emb])
            self.utr3emb = torch.stack([e[1] for e in self.seq_emb])
        elif type(self.seq_emb) == torch.Tensor:
            self.utr5emb = self.seq_emb[: self.seq_emb.size()[0] // 2]
            self.utr3emb = self.seq_emb[self.seq_emb.size()[0] // 2 :]

    def __getitem__(self, idx) -> Union[list, int]:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        return embs, pair_data[2], pair_data  # label

    def __len__(self):
        return len(self.pair_idx)


class PairDataset_Multi(Dataset):
    def __init__(self, seq_embs: list, data: np.array):
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


class PairDatasetAttn(Dataset):
    """Dataset class for pairpred Contrastive Learning task"""

    def __init__(self, seq_emb: tuple, pair_idx: np.ndarray):
        super().__init__()
        self.seq_emb = seq_emb
        self.pair_idx = pair_idx

        self.utr5emb = self.seq_emb[0]  # (sample_num,max_length,hidden_dim)
        self.utr3emb = self.seq_emb[1]  # (sample_num,max_length,hidden_dim)

    def __getitem__(self, idx) -> tuple:
        pair_data = self.pair_idx[idx]
        embs = (self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]])
        labels = pair_data[2]
        return embs, labels, pair_data

    def __len__(self):
        return len(self.pair_idx)


class PairDatasetCL(Dataset):
    """Dataset class for pairpred Contrastive Learning task"""

    def __init__(self, seq_emb: list, pair_idx: np.ndarray):
        super().__init__()
        self.seq_emb = seq_emb
        self.pair_idx = pair_idx

        if type(self.seq_emb) == list:
            self.utr5emb = torch.stack([e[0] for e in self.seq_emb])
            self.utr3emb = torch.stack([e[1] for e in self.seq_emb])
        elif type(self.seq_emb) == torch.Tensor:
            self.utr5emb = self.seq_emb[: self.seq_emb.size()[0] // 2]
            self.utr3emb = self.seq_emb[self.seq_emb.size()[0] // 2 :]

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
        super(PairDatasetCL_Multi_test, self).__init__(seq_embs, pair_idx)

    def __getitem__(self, idx) -> list:
        pair_data = self.pair_idx[idx]
        embs = [self.utr5emb[pair_data[0]], self.utr3emb[pair_data[1]]]
        label = pair_data[2]
        return embs, label, pair_data
