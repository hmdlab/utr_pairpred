# UTR_PairPred

## Installation
requirements
```sh
python>=3.9.0
CUDA=11.8
torch=2.2.0
```
- Install required libraries  with `poetry install`


## Data preprocess
1. Download GENCODE, `Protein-coding transcript sequences` fasta file from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.pc_transcripts.fa.gz)

2. Createing sequence df from GENCODE raw fasta file.
```linux
cd scripts
sh create_seq_df.sh
```

3. Getting sequence embeddings (model inputs).
- With `RNA-FM`: `sh get_emb_rnafm.sh`
- With `RiNALMo`: `sh get_emb_rinalmo.sh`
- For random forest feature: `sh get_rf_feature.sh`


## Training prediction models
- Use `src/run_train_XX.py` code for training (replace XX from the below learning method abb table as you want).
- Config also has name rule `config/<SPECIES>_<LEANING_METHOD>.yaml`

| abb | full |
| ---- | ---- |
| cl | contrastive learning |
| sv | supervised learning |
| rf | random forest |

- Run example
```sh
poetry run python run_train_cl.py --cfg ../config/human_cl.yaml
```

## Downstream analysis
- 

## Utils
**cd-hit**
- To eliminate similar sequences, use script `script/cd-hit.sh`
- After running cd-hit, it's need to create new seq_df and embedding along with the result.
	- You can use `create_represent_seq_df()` func in `utils.py` file.
\\
