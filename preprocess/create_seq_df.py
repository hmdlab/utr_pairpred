import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import argparse


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, help="path to pc_transcripts.fasta file")
    args.add_argument("--output", type=str, help="path to output file")
    args.add_argument("--max_seq_len", type=int, default=None)
    opt = args.parse_args()
    return opt


def ext_longest_seq(seq_df: pd.DataFrame) -> pd.DataFrame:
    ## Extract longest sequence for each GENE symbol.
    abs_idx_list = []
    unique_genes = seq_df.GENE.value_counts().index.values
    for gene in tqdm(unique_genes):
        tmp_df = seq_df.query("GENE==@gene")
        max_idx = tmp_df["total_len"].idxmax()
        abs_idx_list.append(max_idx)

    max_len_trans_df = seq_df.iloc[abs_idx_list]

    return max_len_trans_df


def convert_TtoU(seq: str) -> str:
    return seq.replace("T", "U")


def main(opt):
    # Create seq df
    fasta = opt.file_path
    UTR5_start, UTR5_end, CDS_start, CDS_end, UTR3_start, UTR3_end = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    gene_symbols = []
    total_len = []
    seq_dict = {}

    # loading fasta file.
    for rec in SeqIO.parse(fasta, format="fasta"):
        desc = rec.description
        seq = str(rec.seq)
        enst_id = desc.split("|")[0]
        regions = desc.split("|")[7:-1]
        flag_5 = 0
        flag_3 = 0
        for reg in regions:
            if "UTR5" in reg:
                flag_5 = True
            elif "UTR3" in reg:
                flag_3 = True
        if flag_5 and flag_3:
            gene_symbols.append(desc.split("|")[5])
            total_len.append(int(desc.split("|")[6]))
            UTR5, CDS, UTR3 = regions[0], regions[1], regions[2]  # UTRおよびCDSの位置を取得
            UTR5_start.append(int(UTR5.split(":")[1].split("-")[0]))  # UTR5の開始位置をリストに追加
            UTR5_end.append(int(UTR5.split(":")[1].split("-")[1]))  # UTR5の終了位置をリストに追加
            CDS_start.append(int(CDS.split(":")[1].split("-")[0]))  # CDSの開始位置をリストに追加
            CDS_end.append(int(CDS.split(":")[1].split("-")[1]))  # CDSの終了位置をリストに追加
            UTR3_start.append(int(UTR3.split(":")[1].split("-")[0]))  # UTR3の開始位置をリストに追加
            UTR3_end.append(int(UTR3.split(":")[1].split("-")[1]))  # UTR3の終了位置をリストに追加
            seq_dict[enst_id] = seq

    seq_df = pd.DataFrame(
        {
            "ENST_ID": list(seq_dict.keys()),
            "GENE": gene_symbols,
            "5UTR": [
                seq_dict[k][s - 1 : e].replace("T", "U")
                for k, s, e in zip(seq_dict.keys(), UTR5_start, UTR5_end)
            ],
            "CDS": [
                seq_dict[k][s - 1 : e].replace("T", "U")
                for k, s, e in zip(seq_dict.keys(), CDS_start, CDS_end)
            ],
            "3UTR": [
                seq_dict[k][s - 1 :].replace("T", "U")
                for k, s, e in zip(seq_dict.keys(), UTR3_start, UTR3_end)
            ],
            "total_len": total_len,
            "5UTR_len": [(e - s) + 1 for s, e in zip(UTR5_start, UTR5_end)],
            "CDS_len": [(e - s) + 1 for s, e in zip(CDS_start, CDS_end)],
            "3UTR_len": [(e - s) + 1 for s, e in zip(UTR3_start, UTR3_end)],
        }
    )
    max_len_trans_df = ext_longest_seq(seq_df)
    if opt.max_seq_len == None:
        max_len_trans_df = max_len_trans_df[
            (max_len_trans_df["5UTR_len"] >= 10) & (max_len_trans_df["3UTR_len"] >= 10)
        ]  ## filtering by minimum seq length

    else:
        max_len_trans_df = max_len_trans_df[
            (max_len_trans_df["5UTR_len"] >= 10)
            & (max_len_trans_df["5UTR_len"] <= opt.max_seq_len)
            & (max_len_trans_df["3UTR_len"] >= 10)
            & (max_len_trans_df["3UTR_len"] <= opt.max_seq_len)
        ]
    print(f"df_shape:{max_len_trans_df.shape}")
    max_len_trans_df.to_csv(opt.output)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
