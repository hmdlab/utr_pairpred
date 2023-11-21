import re
import numpy as np
import pandas as pd


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


def extract_starred_enst_ids_from_cluster_file(file_path):
    enst_id_list = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        current_cluster = None

        for line in lines:
            if line.startswith(">Cluster"):
                # 新しいクラスタが始まる行を検出した場合
                current_cluster = line.strip()
                continue

            match = re.search(r"ENST([0-9A-Za-z]+\.\d+)\.\.\. \*", line)
            if match:
                enst_id = match.group(0)
                enst_id = enst_id.replace("... *", "")
                enst_id_list.append(f"{enst_id}")

    return enst_id_list
