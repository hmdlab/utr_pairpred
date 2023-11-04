import argparse
import numpy as np
import pandas as pd
import torch
import fm


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--file_path", type=str, help="path to input file")
    args.add_argument(
        "--over_length",
        type=str,
        choices=["trancate", "average"],
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
    def __init__(self, opt: argparse.Namespace):
        self.max_seq_len = 1022  #

        self.over_length = opt.over_length  # "trancate" or "average"
        self.model, self.alphabet = fm.pretrained.rna_fm_t12(
            model_location=opt.RNAFM_path
        )
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
    
    def _calc_embedding(self,seq:str,seq_name:str)->torch.Tensor:
        data = [(f"{seq_name}", f"{seq}")]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][
            12
        ]  # dim=(1,seq_len+2,emb_dim=640)
        return token_embeddings[0][0]  # return embedding of [CLS] token.

    def get(self, seq: str, seq_name="RNA1")->torch.Tensor:
        seq_len=len(seq)
        if seq_len>self.max_seq_len:
            if self.over_length=="trancate":
                seq=seq[:self.max_seq_len]   
                embedding=self._calc_embedding(seq,seq_name)
                return embedding
            elif self.over_length=="average":
                seq_fragments=[seq[i:i+self.max_seq_len] for i in range(0, len(seq), self.max_seq_len)]
                frag_embs=[self._calc_embedding(seq_frag,seq_name) for seq_frag in seq_fragments ] # list[torch.Tensor]
                frag_embs=torch.tensor(frag_embs)
                ave_embedding=torch.mean(frag_embs,1) # calc mean along with dim=1
                return ave_embedding

def main(opt: argparse.Namespace):
    pass


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
