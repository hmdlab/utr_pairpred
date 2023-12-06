import re
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    def __init__(self, gamma):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.bceloss = nn.BCELoss(reduction="none")
        self.sigmoid = nn.Sigmoid()

    def forward(self, outputs, targets):
        outputs = self.sigmoid(outputs)
        bce = self.bceloss(outputs, targets)
        bce_exp = torch.exp(-bce)
        focal_loss = (1 - bce_exp) ** self.gamma * bce
        return focal_loss.mean()


class NpairLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, logits_per_utr5, logits_per_utr3):
        label = torch.arange(len(logits_per_utr5), device=self.device, dtype=torch.long)

        loss_utr5 = F.cross_entropy(logits_per_utr5, label)
        loss_utr3 = F.cross_entropy(logits_per_utr3, label)
        loss = (loss_utr5 + loss_utr3) / 2

        return loss


def csv_to_fasta(seq_df: pd.DataFrame, output_fasta_path: str):
    enst_id = seq_df.ENST_ID
    gene = seq_df.GENE
    utr5 = seq_df["5UTR"]
    utr3 = seq_df["3UTR"]

    # Creating 5utr fasta
    with open(output_fasta_path + "_5utr.fa", "w") as f:
        for e, g, u5 in zip(enst_id, gene, utr5):
            f.write(f">{e} {g}\n{u5}\n")

    # Creating 3utr fasta
    with open(output_fasta_path + "_3utr.fa", "w") as f:
        for e, g, u3 in zip(enst_id, gene, utr3):
            f.write(f">{e} {g}\n{u3}\n")


def extract_enst_ids_from_file(file_path, species="human") -> np.array:
    """Return ENST_id from result file of CD-hit

    Args:
        file_path (_type_): Result .fa file of CD-hit.

    Returns:
        _type_:
    """
    # ファイルを読み込む
    with open(file_path, "r") as file:
        text_data = file.read()

    # ENST idを抽出する正規表現パターン
    if species == "human":
        pattern = re.compile(r"(ENST\d+\.\d+)")
    elif species == "mouse":
        pattern = re.compile(r"(ENSMUST\d+\.\d+)")

    # テキストデータからENST idを抽出
    enst_ids = re.findall(pattern, text_data)
    enst_ids = np.array(enst_ids)

    return enst_ids


def create_represent_seq_df(
    csv_data_path: str, cdhit_prefix: str, output_path: str, species="human"
) -> None:
    """Create represented seq df from results of CD-hit

    Args:
        csv_data_path (str): _description_
        cdhit_prefix (str): _description_
        output_path (str): _description_
    """
    df = pd.read_csv(csv_data_path, index_col=0)
    df.set_index("ENST_ID", inplace=True)
    cdhit_path_5utr = cdhit_prefix + "_5utr.fa"
    cdhit_path_3utr = cdhit_prefix + "_3utr.fa"

    rep_id_list_5utr = extract_enst_ids_from_file(cdhit_path_5utr, species)
    rep_id_list_3utr = extract_enst_ids_from_file(cdhit_path_3utr, species)

    rep_id_list_both = list(set(rep_id_list_5utr) & set(rep_id_list_3utr))

    rep_df = df.loc[rep_id_list_both]
    rep_df.reset_index(inplace=True)

    rep_df.to_csv(output_path)


def create_simple_fasta(csv_data_path: str) -> None:
    df = pd.read_csv(csv_data_path)
    utr5 = df["5UTR"].values
    utr3 = df["3UTR"].values
    enst_id = df["ENST_ID"].values
    gene = df["GENE"].values

    with open(csv_data_path.replace(".csv", "_5utr.fa"), "w") as f5, open(
        csv_data_path.replace(".csv", "_3utr.fa"), "w"
    ) as f3:
        for u5, u3, enst, g in zip(utr5, utr3, enst_id, gene):
            f5.write(f">{enst} {g}\n")
            f5.write(f"{u5}\n")

            f3.write(f">{enst} {g}\n")
            f3.write(f"{u3}\n")


def reconst_pair_idx(pred_results: list) -> np.ndarray:
    """Reconstruct pair idx from reulst pickle file

    Args:
        pred_results (list): list of tensor for each batch

    Returns:
        np.ndarray: pair_idx. dim=[sample,3]
    """
    utr5_idx = []
    utr3_idx = []
    label = []
    for batch in pred_results[0]:
        utr5_idx.extend(batch[0].numpy())
        utr3_idx.extend(batch[1].numpy())
        label.extend(batch[2].numpy())
    pair_idx = np.stack([utr5_idx, utr3_idx, label], axis=1)

    return pair_idx
