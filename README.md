# UTR_PairPred

## Installation
- Install required python libraries  with `poetry install

requirements
```sh
python>=3.9.0
CUDA=11.8
torch=2.2.0
```

- If you want to preprocess by yourself, please also install tools as following instructions.
	- `cd-hit`: https://github.com/weizhongli/cdhit
	- `ViennaRNA`: https://github.com/ViennaRNA/ViennaRNA


## Data preprocess
1. Download GENCODE, `Protein-coding transcript sequences` fasta file from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.pc_transcripts.fa.gz)

2. Createing sequence df from GENCODE raw fasta file.
```linux
cd scripts
sh create_seq_df.sh
```
- Then, `gencode_v44(vM33)_utr_gene_unique.csv` and `gencode_v44(vM33)_utr_gene_unique_5utr(3utr).fa` file will generate.
- If you want to remove similar sequences, please run `scripts/cd_hit.sh` with those fasta files.

3. Getting sequence embeddings (model inputs).
- With `RNA-FM`: `sh get_emb_rnafm.sh`
- With `RiNALMo`: `sh get_emb_rinalmo.sh`
- For random forest feature: `sh get_rf_feature.sh`


## Training prediction models
- Use `src/run_train_XX.py` code for training (replace XX from the below learning method abb table as you want).
- Config also has name rule `config/<SPECIES>_<LEARNING_METHOD>.yaml`

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
## Downstream analysis
- [`crossval_analysis.ipynb`](./notebooks/crossval_analysis.ipynb):  
  Performs cross-validation analysis to evaluate the consistency of results across experiments. Visualizes the distribution of cosine similarity and correlations between different experiments. 

- [`sequential_analysis.ipynb`](./notebooks/sequential_analysis.ipynb): Analyzes basic sequence features (e.g., lengths of 5'UTR, 3'UTR, CDS, and MFE) 

- [`expression_analysis.ipynb`](./notebooks/expression_analysis.ipynb): Analyzes translation efficiency (TE) using RNA-seq and Ribo-seq data for each cell line.

## Utils
**cd-hit**
- To eliminate similar sequences, use script `script/cd-hit.sh`
- After running cd-hit, it's need to create new seq_df and embedding along with the result.
	- You can use `create_represent_seq_df()` func in `utils.py` file.

