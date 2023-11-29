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


def extract_enst_ids_from_file(file_path) -> np.array:
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
    pattern = re.compile(r"(ENST\d+\.\d+)")

    # テキストデータからENST idを抽出
    enst_ids = re.findall(pattern, text_data)
    enst_ids = np.array(enst_ids)

    return enst_ids


def create_represent_seq_df(
    csv_data_path: str, cdhit_prefix: str, output_path: str
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

    rep_id_list_5utr = extract_enst_ids_from_file(cdhit_path_5utr)
    rep_id_list_3utr = extract_enst_ids_from_file(cdhit_path_3utr)

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
