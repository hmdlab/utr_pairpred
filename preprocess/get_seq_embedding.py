"""Getting sequence embedding code"""

import argparse
import os
import pickle
import subprocess
import sys
import time
from itertools import product
from multiprocessing import Pool, cpu_count

import fm
import pandas as pd
import torch
from huggingface_hyena import HyenaDNAPreTrainedModel
from rinalmo.pretrained import get_pretrained_model
from standalone_hyenadna import CharacterTokenizer
from tqdm import tqdm


def _argparse():
    args = argparse.ArgumentParser()
    args.add_argument("--i", type=str, help="path to input file")
    args.add_argument("--o", type=str, help="path to output file")
    args.add_argument(
        "--over_length",
        type=str,
        choices=["trancate_forward", "trancate_back", "average", "edge"],
        help="processing method when the input sequence length over the maximum input len,1022.",
    )
    args.add_argument(
        "--hyenadna",
        type=str,
        default=None,
        choices=[
            "hyenadna-tiny-1k-seqlen",
            "hyenadna-small-32k-seqlen",
            "hyenadna-medium-160k-seqlen",
            "hyenadna-medium-450k-seqlen",
            "hyenadna-large-1m-seqlen",
        ],
        help="hyenadna model name",
    )
    args.add_argument("--rinalmo", action="store_true", default=False)
    args.add_argument(
        "--rinalmo_method", type=str, choices=["whole", "average", "whole_ave"]
    )
    args.add_argument("--feature_craft", action="store_true")
    args.add_argument("--with_pad", action="store_true")
    args.add_argument(
        "--RNAFM_path",
        type=str,
        help="path to pretrained params of RNA-FM",
        default="/home/ksuga/whole_mrna_predictor/RNA-FM/pretrained/RNA-FM_pretrained.pth",
    )
    args.add_argument(
        "--format",
        type=str,
        choices=["pkl", "pt"],
        help="save file format. pickle or pt",
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

        self.model = self.model.to(self.device)

    def _calc_embedding(self, seq: str, seq_name: str) -> torch.Tensor:
        data = [(f"{seq_name}", f"{seq}")]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][
            12
        ]  # dim=(1,seq_len+2,emb_dim=640)
        batch_tokens = batch_tokens.detach().cpu()
        token_embeddings = token_embeddings.detach().cpu()
        return token_embeddings[0][0]  # return embedding of [CLS] token.

    def get(self, seq: str, seq_name="RNA1", region="utr5") -> torch.Tensor:
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
                frag_embs = torch.stack(frag_embs)  # fragments of embeddings
                ave_embedding = torch.mean(frag_embs, 0)  # calc mean along with dim=1
                return ave_embedding

            elif self.over_length == "edge":
                if region == "utr5":
                    seq = seq[: self.max_seq_len]
                elif region == "utr3":
                    seq = seq[-self.max_seq_len :]

                embedding = self._calc_embedding(seq, seq_name)
                return embedding

            else:
                raise NotImplementedError()

        else:
            return self._calc_embedding(seq, seq_name)


class GetEmbeddingRinalMo:
    """Get Embedding class"""

    def __init__(self, opt: argparse.Namespace):
        self.max_seq_len = 1022
        self.method = opt.rinalmo_method

        self.model, self.alphabet = get_pretrained_model(model_name="giga-v1")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = self.model.to(self.device)

    def _calc_embedding(self, seq: str) -> torch.Tensor:
        data = [seq]
        tokens = torch.tensor(
            self.alphabet.batch_tokenize(data), dtype=torch.int64, device=self.device
        )
        tokens = tokens.to(self.device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = self.model(tokens)
        token_embeddings = output["representation"]  # dim=(1,seq_len+2,emb_dim=1280)
        tokens = tokens.detach().cpu()
        token_embeddings = token_embeddings.detach().cpu()
        return token_embeddings[0]  # return embedding dim of [seq_len,emb_dim]

    def get(self, seq: str) -> torch.Tensor:
        """getting embedding for each sequence

        Args:
            seq (str): sequence string

        Returns:
            torch.Tensor: embedding tensor
        """
        if self.method == "average":
            # embedding = self._calc_embedding(seq)
            seq_fragments = [
                seq[i : i + self.max_seq_len]
                for i in range(0, len(seq), self.max_seq_len)
            ]
            frag_embs = [
                self._calc_embedding(seq_frag) for seq_frag in seq_fragments
            ]  # list[torch.Tensor]
            frag_embs = torch.stack(frag_embs)  # fragments of embeddings
            ave_embedding = torch.mean(frag_embs, 0)  # calc mean along with dim=1
            return ave_embedding

        elif self.method == "whole":
            embedding = self._calc_embedding(seq)
            return embedding[0]  # return only [CLS] token

        elif self.method == "whole_ave":
            embedding = self._calc_embedding(seq)
            return embedding.mean(dim=0)  # use average of whole embedding.


class GetEmbeddingWithPad(GetEmbedding):
    """Get Embedding class"""

    def __init__(self, opt: argparse.Namespace):
        super().__init__(opt)

    def _calc_embedding(self, seq: str, seq_name: str) -> torch.Tensor:
        data = [(f"{seq_name}", f"{seq}")]
        _, _, batch_tokens = self.batch_converter(data)
        pad_tokens = torch.ones(
            [(self.max_seq_len + 2) - batch_tokens.size()[-1]], dtype=torch.long
        )
        batch_tokens = torch.concat([batch_tokens.squeeze(), pad_tokens]).unsqueeze(0)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[12])
        token_embeddings = results["representations"][
            12
        ]  # dim=(1,max_seq_len,emb_dim=640)
        batch_tokens = batch_tokens.detach().cpu()
        token_embeddings = token_embeddings.detach().cpu()
        return token_embeddings  # return all

    def get(self, seq: str, seq_name="RNA1", region="utr5") -> torch.Tensor:
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

            else:
                raise NotImplementedError()

        else:
            return self._calc_embedding(seq, seq_name)


class GetEmbeddingHyenaDNA:
    def __init__(self, pretrained_model_name="hyenadna-small-32k-seqlen"):
        use_head = False
        n_classes = 2
        backbone_cfg = None
        max_lengths = {
            "hyenadna-tiny-1k-seqlen": 1024,
            "hyenadna-small-32k-seqlen": 32768,
            "hyenadna-medium-160k-seqlen": 160000,
            "hyenadna-medium-450k-seqlen": 450000,  # T4 up to here
            "hyenadna-large-1m-seqlen": 1_000_000,  # only A100 (paid tier)
        }
        max_length = max_lengths[pretrained_model_name]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)
        self.model = HyenaDNAPreTrainedModel.from_pretrained(
            "./checkpoints",
            pretrained_model_name,
            download=True,
            config=backbone_cfg,
            device=self.device,
            use_head=use_head,
            n_classes=n_classes,
        )
        self.model.to(self.device)

        self.tokenizer = CharacterTokenizer(
            characters=["A", "C", "G", "T", "N"],  # add DNA characters, N is uncertain
            model_max_length=max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side="left",  # since HyenaDNA is causal, we pad on the left
        )

    def get(self, seq: str, seq_name=None) -> torch.Tensor:
        tok_seq = self.tokenizer(seq)
        tok_seq = tok_seq["input_ids"]  # grab ids

        # place on device, convert to tensor
        tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
        tok_seq = tok_seq.to(self.device)
        self.model.eval()
        with torch.inference_mode():
            embeddings = self.model(tok_seq)  # dim=[bs,seq_len,hidden_dim]

        tok_seq = tok_seq.detach().cpu()

        return embeddings[0].mean(dim=0)  # embeddings[0][-1]  only use CLS token


class GetFeature:
    def __init__(self):
        pass

    def singleNucleotide_composition(self, seq):
        base_dic = {"A": 0, "T": 0, "G": 0, "C": 0}
        for base in base_dic:
            base_dic[base] = seq.count(base)

        for k, v in base_dic.items():  ## avoid zero divide
            if v == 0:
                base_dic[k] = 1

        feature_map = dict()
        feature_map["CGperc"] = (base_dic["C"] + base_dic["G"]) / len(seq)
        feature_map["CGratio"] = base_dic["C"] / base_dic["G"]
        feature_map["ATratio"] = base_dic["A"] / base_dic["T"]
        feature_map["Length"] = len(seq)

        feature_map = dict(**base_dic, **feature_map)

        return feature_map

    def _RNAfold_energy(self, seq: str, *args)->float:
        rnaf = subprocess.Popen(
            ["RNAfold", "--noPS"] + list(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Universal Newlines effectively allows string IO.
            universal_newlines=True,
        )
        rnafold_output, _ = rnaf.communicate(seq)
        output_lines = (
            rnafold_output.strip().splitlines()
        )  ## output_lines:list=[original_seq,dot-bracket (energy)]
        try:
            energy = float(output_lines[1].rsplit("(", 1)[1].strip("()").strip())
        except IndexError:
            print(f'Error output:{output_lines}')
            energy = -100
        return energy

    def foldenergy_feature(self, seq) -> dict:
        dna_str = str(seq)
        feature_map = dict()
        # feature_map["energy_head"] = self._RNAfold_energy(dna_str[:100])
        feature_map["energy_whole"] = self._RNAfold_energy(dna_str)
        # feature_map["energy_tail"] = self._RNAfold_energy(dna_str[-100:])
        return feature_map

    def _count_kmers(self, seq: str, k):
        kmer_counts = {}
        sequence_length = len(seq)

        for i in range(sequence_length - k + 1):
            kmer = seq[i : i + k]

            if kmer in kmer_counts:
                kmer_counts[kmer] += 1
            else:
                kmer_counts[kmer] = 1

        return kmer_counts

    def _generate_all_patterns(self, k):
        # 全てのk-merパターンを生成
        nucleotides = ["A", "C", "G", "T"]
        patterns = ["".join(p) for p in product(nucleotides, repeat=k)]
        return patterns

    def Kmer_feature(self, seq: str) -> dict:
        result_dict = dict()
        for k in range(3, 7):
            kmer_counts = self._count_kmers(seq, k)

            # 存在しないパターンに対してもカウント0として辞書に含める
            for pattern in self._generate_all_patterns(k):
                result_dict[pattern] = kmer_counts.get(pattern, 0)
        return result_dict

    def oss(self, cmd):
        print(cmd)
        os.system(cmd)

    def get(self, seq):
        ##codon
        feature_dic: dict = self.singleNucleotide_composition(seq)
        feature_dic = dict(**feature_dic, **self.foldenergy_feature(seq))
        feature_dic = dict(**feature_dic, **self.Kmer_feature(seq))
        return feature_dic


def main(opt: argparse.Namespace):
    """main"""
    seq_df = pd.read_csv(opt.i, index_col=0)
    emb_array = []

    if opt.hyenadna:
        print(f"Using '{opt.hyenadna}' model for embedding !!!")
        embedder = GetEmbeddingHyenaDNA(pretrained_model_name=opt.hyenadna)
        seq5utr, seq3utr = seq_df["5UTR"].values, seq_df["3UTR"].values

        for utr5, utr3 in tqdm(zip(seq5utr, seq3utr)):
            utr5, utr3 = utr5.replace("U", "T"), utr3.replace("U", "T")
            emb5, emb3 = embedder.get(utr5), embedder.get(utr3)
            emb5, emb3 = emb5.detach().cpu(), emb3.detach().cpu()
            emb_array.append([emb5, emb3])

    elif opt.feature_craft:
        print("Creating features !!!")
        converter = GetFeature()
        pool = Pool(cpu_count())
        seq5utr, seq3utr = seq_df["5UTR"].values, seq_df["3UTR"].values
        seq5utr = [seq.replace("U", "T") for seq in seq5utr]
        seq3utr = [seq.replace("U", "T") for seq in seq3utr]
        """
        print("Creating 5utr features ...")
        time1 = time.time()
        feature_5utr = pool.map(converter.get, seq5utr)
        df_5utr = pd.DataFrame(feature_5utr, index=seq_df["ENST_ID"].values)
        df_5utr.to_csv(
            os.path.join(
                opt.o, os.path.basename(opt.i).replace(".csv", "_feature_5utr.csv")
            )
        )
        """
        time2 = time.time()
        #print(f"elaplsed time 5utr:{time2-time1}s")
        print("Creating 3utr features ...")
        feature_3utr = pool.map(converter.get, seq3utr)
        df_3utr = pd.DataFrame(feature_3utr, index=seq_df["ENST_ID"].values)
        df_3utr.to_csv(
            os.path.join(
                opt.o, os.path.basename(opt.i).replace(".csv", "_feature_3utr.csv")
            )
        )
        time3 = time.time()
        print(f"elaplsed time 3utr:{time3-time2}s")

        sys.exit(0)

    elif opt.with_pad:
        print("RNA-FM batch processing")
        all_emb5, all_emb3 = None, None
        embedder = GetEmbeddingWithPad(opt)
        seq5utr, seq3utr = seq_df["5UTR"].values, seq_df["3UTR"].values
        for utr5, utr3 in tqdm(zip(seq5utr, seq3utr)):
            emb5, emb3 = embedder.get(utr5), embedder.get(utr3)
            if all_emb5 is None:
                all_emb5, all_emb3 = emb5, emb3
            else:
                all_emb5 = torch.concat([all_emb5, emb5])
                all_emb3 = torch.concat([all_emb3, emb3])

        emb_array = (all_emb5, all_emb3)

    elif opt.rinalmo:
        print("Using RiNARMo for Embedding !!!")
        embedder = GetEmbeddingRinalMo(opt)
        seq5utr, seq3utr = seq_df["5UTR"].values, seq_df["3UTR"].values

        for utr5, utr3 in tqdm(zip(seq5utr, seq3utr)):
            emb5, emb3 = embedder.get(utr5), embedder.get(utr3)
            emb_array.append([emb5, emb3])

        emb5_all = torch.stack([e[0] for e in emb_array])
        emb3_all = torch.stack([e[1] for e in emb_array])

        emb_pt = torch.concat([emb5_all, emb3_all])

    else:
        print("Using 'RNA-FM' for embedding !!! ")
        embedder = GetEmbedding(opt)
        seq5utr, seq3utr = seq_df["5UTR"].values, seq_df["3UTR"].values

        for utr5, utr3 in tqdm(zip(seq5utr, seq3utr)):
            if opt.over_length == "edge":
                emb5, emb3 = embedder.get(utr5, region="utr5"), embedder.get(
                    utr3, region="utr3"
                )

            else:
                emb5, emb3 = embedder.get(utr5), embedder.get(utr3)
            emb_array.append([emb5, emb3])

    print("Writing down embedding results ...")
    if opt.format == "pkl":
        with open(opt.o, "wb") as f:
            pickle.dump(emb_array, f)

    elif opt.format == "pt":
        torch.save(emb_pt, opt.o)


if __name__ == "__main__":
    opt = _argparse()
    main(opt)
