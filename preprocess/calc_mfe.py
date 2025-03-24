import argparse

import numpy as np
import pandas as pd
import RNA
from tqdm import tqdm


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--seq_df", required=True, help="path to sequence df.")
    args.add_argument("--out", required=True, help="path to output results.")
    opt = args.parse_args()
    return opt


def calc_mfe(seq: str):
    fc = RNA.fold_compound(seq)
    # MFEを計算
    (_, mfe) = fc.mfe()
    return mfe


def utr5_process(
    seq_list: np.ndarray, head_len=100, tail_len=100, cds_len=100
) -> pd.DataFrame:
    """

    Args:
        seq_list (np.ndarray): seq_num, 2 (5utr,CDS)
        head_len (int, optional): _description_. Defaults to 100.
        tail_len (int, optional): _description_. Defaults to 100.
        cds_len (int, optional): _description_. Defaults to 100.

    Returns:
        pd.DataFrame: _description_
    """
    print("Calculating 5UTRs ...")
    # whole_mfe = []
    tail_mfes = []
    tail_cds_mfes = []
    head_mfes = []
    for utr5, cds in tqdm(seq_list):
        if len(utr5) < tail_len:
            tail_seq = utr5
            head_seq = utr5
            if len(cds) < cds_len:
                tail_cds_seq = utr5 + cds
            else:
                tail_cds_seq = utr5 + cds[:cds_len]
        else:
            tail_seq = utr5[-tail_len:]
            head_seq = utr5[:head_len]
            if len(cds) < cds_len:
                tail_cds_seq = tail_seq + cds
            else:
                tail_cds_seq = tail_seq + cds[:cds_len]

        tail_mfes.append(calc_mfe(tail_seq))
        tail_cds_mfes.append(calc_mfe(tail_cds_seq))
        head_mfes.append(calc_mfe(head_seq))

    utr5_mfe_df = pd.DataFrame(
        {
            "5UTR_head_mfe": head_mfes,
            "5UTR_tail_mfe": tail_mfes,
            "5UTR_tail_cds_mfe": tail_cds_mfes,
        }
    )
    return utr5_mfe_df


def utr3_process(
    seq_list: np.ndarray, head_len=500, tail_len=500, cds_len=100
) -> pd.DataFrame:
    """

    Args:
        seq_list (np.ndarray): dim=(seq_num,2) (3utr,CDS)
        head_len (int, optional): _description_. Defaults to 500.
        tail_len (int, optional): _description_. Defaults to 500.
        cds_len (int, optional): _description_. Defaults to 100.

    Returns:
        pd.DataFrame: _description_
    """
    print("Calculating 3UTRs ...")
    # whole_mfe = []
    tail_mfes = []
    head_mfes = []
    head_cds_mfes = []
    for utr3, cds in tqdm(seq_list):
        if len(utr3) < tail_len:
            tail_seq = utr3
            head_seq = utr3
            if len(cds) < cds_len:
                head_cds_seq = cds + utr3
            else:
                head_cds_seq = cds[-cds_len:] + utr3
        else:
            tail_seq = utr3[-tail_len:]
            head_seq = utr3[:head_len]
            if len(cds) < cds_len:
                head_cds_seq = cds + head_seq
            else:
                head_cds_seq = cds[-cds_len:] + head_seq

        tail_mfes.append(calc_mfe(tail_seq))
        head_mfes.append(calc_mfe(head_seq))
        head_cds_mfes.append(calc_mfe(head_cds_seq))

    utr3_mfe_df = pd.DataFrame(
        {
            "3UTR_head_cds_mfe": head_cds_mfes,
            "3UTR_head_mfe": head_mfes,
            "3UTR_tail_mfe": tail_mfes,
        }
    )
    return utr3_mfe_df


def main(opt: argparse.Namespace):
    seq_df = pd.read_csv(opt.seq_df, index_col=0)
    utr5_seqs = seq_df[["5UTR", "CDS"]].values
    utr3_seqs = seq_df[["3UTR", "CDS"]].values

    utr3_mfe_res = utr3_process(utr3_seqs)
    utr5_mfe_res = utr5_process(utr5_seqs)

    total_df = pd.concat([seq_df, utr5_mfe_res, utr3_mfe_res], axis=1)
    total_df.to_csv(opt.out)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
