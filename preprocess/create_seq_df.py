import argparse
import os

import pandas as pd
from Bio import SeqIO
from pyensembl import Genome
from tqdm import tqdm


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, help="path to pc_transcripts.fasta file")
    args.add_argument("--output", type=str, help="path to output file")
    args.add_argument("--max_seq_len", type=int, default=None)
    ## args related to pyensembl.
    args.add_argument(
        "--ensembl_dir", type=str, default=None, help="path to ensembl data dir"
    )
    args.add_argument(
        "--ref_name",
        type=str,
        default=None,
        help="reference genome name when building pyensembl db",
    )
    args.add_argument(
        "--gtf",
        type=str,
        default=None,
        help="gtf file name using when building pyensembl db",
    )
    args.add_argument(
        "--fasta",
        type=str,
        default=None,
        help="fasta file name using when building pyensembl db",
    )
    opt = args.parse_args()
    return opt


def ext_longest_seq(seq_df: pd.DataFrame) -> pd.DataFrame:
    ## Extract longest sequence for each GENE symbol.
    abs_idx_list = []
    unique_genes = seq_df.GENE.value_counts().index.values
    for gene in tqdm(unique_genes):  # noqa: B007
        tmp_df = seq_df.query("GENE==@gene")
        max_idx = tmp_df["total_len"].idxmax()
        abs_idx_list.append(max_idx)

    max_len_trans_df = seq_df.iloc[abs_idx_list]

    return max_len_trans_df


def convert_TtoU(seq: str) -> str:
    return seq.replace("T", "U")


def create_fasta(df: pd.DataFrame, output_path: str) -> None:
    # Create 5UTR fasta file
    utr5_savepath = os.path.basename(output_path).replace(".csv", "_5utr.fa")
    with open(utr5_savepath, "w") as utr5_file:
        for _, row in df.iterrows():
            utr5_file.write(f">{row['ENST_ID']} {row['GENE']}\n{row['5UTR']}\n")

    # Create 3UTR fasta file
    utr3_savepath = os.path.basename(output_path).replace(".csv", "_3utr.fa")
    with open(utr3_savepath, "w") as utr3_file:
        for _, row in df.iterrows():
            utr3_file.write(f">{row['ENST_ID']} {row['GENE']}\n{row['3UTR']}\n")


def main_ensembl(opt: argparse.Namespace):
    ## building pyensembl db
    embl_data = Genome(
        reference_name=os.path.join(opt.ensembl_dir, opt.ref_name),
        annotation_name=f"my_{opt.ref_name}",
        gtf_path_or_url=os.path.join(opt.ensembl_dir, opt.gtf),
        transcript_fasta_paths_or_urls=os.path.join(opt.ensembl_dir, opt.fasta),
    )
    embl_data.index()

    ## Filtering sequence
    (
        enst_ids,
        ensg_id,
        valid_5utr,
        valid_3utr,
        valid_cds,
        len_total,
        len_5utr,
        len_3utr,
        len_cds,
    ) = ([], [], [], [], [], [], [], [], [])

    min_len = 9

    for rec in SeqIO.parse(os.path.join(opt.ensembl_dir, opt.fasta), "fasta"):
        tmp_data = rec.description.split(" ")
        enst_id = tmp_data[0].split(".")[0]
        try:
            transcript = embl_data.transcript_by_id(enst_id)
            utr5 = (
                transcript.five_prime_utr_sequence
                if len(transcript.five_prime_utr_sequence) > min_len
                else None
            )
            utr3 = (
                transcript.three_prime_utr_sequence
                if len(transcript.three_prime_utr_sequence) > min_len
                else None
            )
            cds = (
                transcript.coding_sequence
                if len(transcript.coding_sequence) > min_len
                else None
            )

            if (utr5 is not None) and (utr3 is not None) and (cds is not None):
                enst_ids.append(enst_id)
                ensg_id.append(transcript.gene_id)
                valid_cds.append(cds)
                valid_5utr.append(utr5)
                valid_3utr.append(utr3)

                len_total.append(len(cds) + len(utr5) + len(utr3))
                len_cds.append(len(cds))
                len_5utr.append(len(utr5))
                len_3utr.append(len(utr3))

        except ValueError:
            print(f"Error: {enst_id} not found in Ensembl data.")
            continue

    seq_df = pd.DataFrame(
        {
            "ENST_ID": enst_ids,
            "GENE": ensg_id,
            "5UTR": valid_5utr,
            "CDS": valid_cds,
            "3UTR": valid_3utr,
            "total_len": len_total,
            "5UTR_len": len_5utr,
            "CDS_len": len_cds,
            "3UTR_len": len_3utr,
        }
    )
    max_len_trans_df = ext_longest_seq(seq_df)
    if opt.max_seq_len is not None:
        max_len_trans_df = max_len_trans_df[
            (max_len_trans_df["5UTR_len"] >= 10)
            & (max_len_trans_df["5UTR_len"] <= opt.max_seq_len)
            & (max_len_trans_df["3UTR_len"] >= 10)
            & (max_len_trans_df["3UTR_len"] <= opt.max_seq_len)
        ]
    print(f"df_shape:{max_len_trans_df.shape}")
    max_len_trans_df.to_csv(opt.output)


def main(opt: argparse.Namespace):
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
            UTR5, CDS, UTR3 = (
                regions[0],
                regions[1],
                regions[2],
            )
            UTR5_start.append(int(UTR5.split(":")[1].split("-")[0]))
            UTR5_end.append(int(UTR5.split(":")[1].split("-")[1]))
            CDS_start.append(int(CDS.split(":")[1].split("-")[0]))
            CDS_end.append(int(CDS.split(":")[1].split("-")[1]))
            UTR3_start.append(int(UTR3.split(":")[1].split("-")[0]))
            UTR3_end.append(int(UTR3.split(":")[1].split("-")[1]))
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
    if opt.max_seq_len is None:
        max_len_trans_df = max_len_trans_df[
            (max_len_trans_df["5UTR_len"] >= 10) & (max_len_trans_df["3UTR_len"] >= 10)
        ]  # filtering by minimum seq length

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
    os.makedirs("../data/human", exist_ok=True)
    os.makedirs("../data/mouse", exist_ok=True)

    if opt.ensembl_dir is None:
        main(opt)
    else:
        main_ensembl(opt)
