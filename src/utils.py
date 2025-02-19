import os
import pickle
import re
import numpy as np
import pandas as pd
from biomart import BiomartServer
import matplotlib.pyplot as plt
from typing import List, Tuple
from matplotlib import cbook
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)


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


def discretize(logits, threshold=0.5):
    discretized = logits >= threshold
    return discretized


def metrics(pred: np.array, label: np.array, out_logits: np.array, phase="val") -> dict:
    "evaluation metrics"
    scores = dict()
    scores["accuracy"] = accuracy_score(label, pred)
    scores["precision"] = precision_score(label, pred)
    scores["recall"] = recall_score(label, pred)
    scores["f1"] = f1_score(label, pred)
    scores["matthews"] = matthews_corrcoef(label, pred)
    scores["auc_roc"] = roc_auc_score(label, out_logits)
    scores["auc_prc"] = average_precision_score(label, out_logits)
    """
    if phase == "test":
        conf_mat = confusion_matrix(label, pred)
        scores["confusion_matrix"] = conf_mat
        fpr_all, tpr_all, thresholds_all = roc_curve(
            label, out_logits, drop_intermediate=False
        )
        scores["fpr_all"] = fpr_all
        scores["tpr_all"] = tpr_all
        scores["roc_thresh"] = thresholds_all
    """
    return scores


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


def reconst_pair_idx(pair_idx_list: list) -> np.ndarray:
    """Reconstruct pair idx from result pickle file

    Args:
        pred_results (list): list of tensor for each batch

    Returns:
        np.ndarray: pair_idx. dim=[sample,3]
    """
    utr5_idx = []
    utr3_idx = []
    label = []
    for batch in pair_idx_list:
        utr5_idx.extend(batch[0].numpy())
        utr3_idx.extend(batch[1].numpy())
        label.extend(batch[2].numpy())
    pair_idx = np.stack([utr5_idx, utr3_idx, label], axis=1)

    return pair_idx


class Seq2Feature:
    def __init__(self):
        pass

    def codonFreq(self, seq):
        codon_str = seq.translate()
        tot = len(codon_str)
        feature_map = dict()
        for a in codon_str:
            a = "codon_" + a
            if a not in feature_map:
                feature_map[a] = 0
            feature_map[a] += 1.0 / tot
        feature_map["uAUG"] = codon_str.count("M")  # number of start codon
        feature_map["uORF"] = codon_str.count("*")  # number of stop codon
        return feature_map

    def singleNucleotide_composition(self, seq, three=False):
        dna_str = str(seq).upper()
        N_count = dict()  # add one pseudo count
        N_count["C"] = 1
        N_count["G"] = 1
        N_count["A"] = 1
        N_count["T"] = 1
        for a in dna_str:
            if a not in N_count:
                N_count[a] = 0
            N_count[a] += 1
        feature_map = dict()
        feature_map["CGperc"] = float(N_count["C"] + N_count["G"]) / len(dna_str)
        feature_map["CGratio"] = abs(float(N_count["C"]) / N_count["G"] - 1)
        feature_map["ATratio"] = abs(float(N_count["A"]) / N_count["T"] - 1)

        feature_map["length"] = len(seq)

        return feature_map

    def convert(self, seq: str):
        # feature_map = list(self.codonFreq(seq))
        feature_map = list(self.singleNucleotide_composition(seq).items())

        return feature_map


def get_go_enst_table() -> dict:
    server = BiomartServer("http://www.ensembl.org/biomart")

    # データセットのリストを取得
    ensembl_genes = server.databases["ENSEMBL_MART_ENSEMBL"]
    ensembl_genes = server.datasets["hsapiens_gene_ensembl"]
    response = ensembl_genes.search({"attributes": ["go_id", "ensembl_transcript_id"]})
    go_enst_dict = {}
    for line in response.iter_lines():
        line = line.decode("utf-8")
        items = line.split("\t")
        if items[0] == "":
            pass
        else:
            go_id_list = go_enst_dict.get(items[0])
            if go_id_list == None:
                go_enst_dict[items[0]] = []
            else:
                go_id_list.append(items[1])

    return go_enst_dict


def create_total_df(
    res_dir: str, seq_df: pd.DataFrame, kfold: int = 10
) -> pd.DataFrame:
    kfold = kfold
    flg = 1
    for i in range(kfold):
        dir = os.path.join(res_dir, str(i))
        pred_path = os.path.join(dir, "pred_results.pkl")
        with open(pred_path, "rb") as f:
            pred_results = pickle.load(f)  # (out_cos_sim,out_logits,pair_idx_list)
            pair_idx = reconst_pair_idx(pred_results[-1])
            logits = pred_results[1].reshape(-1)
            cos_sim = pred_results[0].reshape(-1)

        df_pred_res = pd.DataFrame(pair_idx, columns=["utr5", "utr3", "label"])
        df_pred_res["pred"] = list(map(discretize, pred_results[1]))
        df_pred_res["correct"] = (df_pred_res["label"] == df_pred_res["pred"]).values
        df_pred_res["logits"] = logits
        df_pred_res["cos_sim"] = cos_sim
        df_pred_res["ENST_ID"] = seq_df.iloc[df_pred_res.utr5.values]["ENST_ID"].values
        df_pred_res["ENST_ID_PRE"] = list(
            map(lambda enst_id: enst_id.split(".")[0], df_pred_res["ENST_ID"].values)
        )
        df_pred_res["GENE"] = seq_df.iloc[df_pred_res.utr5.values]["GENE"].values
        df_pred_res = pd.concat(
            [
                df_pred_res,
                seq_df.iloc[:, 3:].iloc[df_pred_res.utr5.values].reset_index(drop=True),
            ],
            axis=1,
        )
        df_pred_res.sort_values("cos_sim", ascending=False, inplace=True)
        if flg:
            total_df = df_pred_res
            flg = 0
        else:
            total_df = pd.concat([total_df, df_pred_res])

        total_df.sort_values("cos_sim", ascending=False, inplace=True)
        total_df = total_df[(total_df["label"] == 1)]

    return total_df


def create_total_df_sv(
    res_dir: str, seq_df: pd.DataFrame, kfold: int = 10
) -> pd.DataFrame:
    kfold = kfold
    flg = 1
    for i in range(kfold):
        dir = os.path.join(res_dir, str(i))
        pred_path = os.path.join(dir, "pred_results.pkl")
        with open(pred_path, "rb") as f:
            pred_results = pickle.load(
                f
            )  # [pair_idx_list, preds, out_labels, out_logits]
            pair_idx = reconst_pair_idx(pred_results[0])
            logits = pred_results[-1].reshape(-1)

        df_pred_res = pd.DataFrame(pair_idx, columns=["utr5", "utr3", "label"])
        df_pred_res["pred"] = pred_results[1]
        df_pred_res["correct"] = (df_pred_res["label"] == df_pred_res["pred"]).values
        df_pred_res["logits"] = logits
        df_pred_res["ENST_ID"] = seq_df.iloc[df_pred_res.utr5.values]["ENST_ID"].values
        df_pred_res["ENST_ID_PRE"] = list(
            map(lambda enst_id: enst_id.split(".")[0], df_pred_res["ENST_ID"].values)
        )
        df_pred_res["GENE"] = seq_df.iloc[df_pred_res.utr5.values]["GENE"].values
        df_pred_res = pd.concat(
            [
                df_pred_res,
                seq_df.iloc[:, -4:]
                .iloc[df_pred_res.utr5.values]
                .reset_index(drop=True),
            ],
            axis=1,
        )
        df_pred_res.sort_values("logits", ascending=False, inplace=True)
        if flg:
            total_df = df_pred_res
            flg = 0
        else:
            total_df = pd.concat([total_df, df_pred_res])

        total_df.sort_values("logits", ascending=False, inplace=True)
        total_df = total_df[(total_df["label"] == 1)]

    return total_df


def boxplot(
    df: pd.DataFrame,
    target: str,
    violin_data_dic: dict,
    tick_names: list,
    save_name=False,
):
    figsize = (6, 4)
    xlabel = "CosSim bin"
    data_list = violin_data_dic[target]
    all_data = df[~((df["cos_sim"] > 0.8) & (df["cos_sim"] < 1))][target].values

    _, ax = plt.subplots(1, 1, figsize=figsize)
    pairs = [(tick_names[-2], tick_names[-1])]

    pvalues = [stats.mannwhitneyu(data_list[-1], all_data)[-1]]

    target_val = np.append(
        np.concatenate([lis for lis in violin_data_dic[target]]), df[target].values
    )
    labels = np.append(
        np.concatenate(
            [
                np.repeat(f"{label}", len(lis))
                for label, lis in zip(tick_names, violin_data_dic[target])
            ]
        ),
        np.repeat("All", len(df)),
    )

    data = {target: target_val, xlabel: labels, "cell": None}

    data_df = pd.DataFrame(data)
    # Seabornのboxplotを使用
    plot_params = {
        "data": data_df,
        "x": xlabel,
        "y": target,
        "showfliers": False,
    }
    sns.boxplot(**plot_params)
    annotator = Annotator(ax, pairs, **plot_params)
    annotator.set_pvalues(pvalues)
    annotator.annotate()
    plt.tight_layout()
    if save_name:
        plt.savefig(f"./results/imgs/{save_name}")
