"""Getting sequence embedding code"""
import argparse
import pickle

import torch
import fm
from tqdm import tqdm
import pandas as pd


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--i", type=str, help="path to input file")
    args.add_argument("--o", type=str, help="path to output file")
    args.add_argument(
        "--over_length",
        type=str,
        choices=["trancate_forward", "trancate_back", "average"],
        help="processing method when the input sequence length over the maximum input len,1022.",
    )
    args.add_argument(
        "--RNAFM_path",
        type=str,
        help="path to pretrained params of RNA-FM",
        default="/home/ksuga/whole_mrna_predictor/RNA-FM/pretrained/RNA-FM_pretrained.pth",
    )
    opt = args.parse_args()
    return opt


class GetEmbedding:
    """Get Embedding class"""

    def __init__(self, opt: argparse.Namespace):
        self.max_seq_len = 1022  #

        self.over_length = opt.over_length  # "trancate" or "average"
        self.model, self.alphabet = fm.pretrained.rna_fm_t12(
            model_location=opt.RNAFM_path
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _calc_embedding(self, seq: str, seq_name: str) -> torch.Tensor:
        data = [(f"{seq_name}", f"{seq}")]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][
            12
        ]  # dim=(1,seq_len+2,emb_dim=640)
        token_embeddings = token_embeddings.detach().cpu()
        return token_embeddings[0][0]  # return embedding of [CLS] token.

    def get(self, seq: str, seq_name="RNA1") -> torch.Tensor:
        """getting embedding for each sequence

        Args:
            seq (str): sequence string
            seq_name (str, optional): seqence name if needed. Defaults to "RNA1".

        Raises:
            NotImplementedError: When over_length process is not ["trancate","average]
        Returns:
            torch.Tensor: embedding tensor
        """
        seq_len = len(seq)
        if seq_len > self.max_seq_len:
            if self.over_length == "trancate_forward":
                seq = seq[: self.max_seq_len]
                embedding = self._calc_embedding(seq, seq_name)
                return embedding
            elif self.over_length == "trancate_back":
                seq = seq[-self.max_seq_len :]
                embedding = self._calc_embedding(seq, seq_name)
                return embedding

            elif self.over_length == "average":
                seq_fragments = [
                    seq[i : i + self.max_seq_len]
                    for i in range(0, len(seq), self.max_seq_len)
                ]
                frag_embs = [
                    self._calc_embedding(seq_frag, seq_name)
                    for seq_frag in seq_fragments
                ]  # list[torch.Tensor]
                frag_embs = torch.stack(frag_embs)
                ave_embedding = torch.mean(frag_embs, 0)  # calc mean along with dim=1
                return ave_embedding
            else:
                raise NotImplementedError()

        else:
            return self._calc_embedding(seq, seq_name)


def main(opt: argparse.Namespace):
    """main"""
    seq_df = pd.read_csv(opt.i, index_col=0)
    embedder = GetEmbedding(opt)
    emb_array = []
    seq5utr, seq3utr = seq_df["5UTR"].values, seq_df["3UTR"].values

    for utr5, utr3 in tqdm(zip(seq5utr, seq3utr)):
        emb5, emb3 = embedder.get(utr5), embedder.get(utr3)
        emb_array.append([emb5, emb3])

    with open(opt.o, "wb") as f:
        pickle.dump(emb_array, f)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
