# UTR_PairPred

## Installation
- Install required python libraries with `poetry install`

Basic requirements
```sh
python>=3.9.0
CUDA=11.8
torch=2.2.0
```

- If you want to preprocess by yourself, please also install tools as following instructions.
	- `cd-hit`: https://github.com/weizhongli/cdhit
	- `ViennaRNA`: https://github.com/ViennaRNA/ViennaRNA


## Data preprocess
**Processed sequence embedding & sequence csv files can be downloaded from [here](https://waseda.box.com/v/utr-pairpred-data.)**

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

* Processed sequence embedding & sequence csv files can be downloaded from [here](https://waseda.box.com/v/utr-pairpred-data.)


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

## Cross-species evaluation
Train models on one species from scratch and evaluate on another species' data (e.g., train on human, evaluate on mouse, or vice versa). This workflow does not use cross-validation (kfold=1).

**Important**: This will train models from scratch, not use pre-trained weights.

- For contrastive learning models:
```sh
cd scripts
sh run_cross_species_eval_cl.sh
```
This will:
1. Train a CL model on human data (60 epochs)
2. Evaluate the trained model on mouse data
3. Train a CL model on mouse data (60 epochs)
4. Evaluate the trained model on human data

- For supervised learning models:
```sh
cd scripts
sh run_cross_species_eval_sv.sh
```
This will:
1. Train an SV model on human data (100 epochs)
2. Evaluate the trained model on mouse data
3. Train an SV model on mouse data (100 epochs)
4. Evaluate the trained model on human data

- You can also run training and evaluation separately:
```sh
# Train on human
poetry run python ../src/run_train_cl.py --cfg ../config/human_cl_cross_species.yaml

# Evaluate on mouse
poetry run python ../src/run_cross_species_eval.py \
    --cfg ../config/human_cl_cross_species.yaml \
    --model_path ../results/runs/cross_species_human_cl/best_model.pth \
    --eval_species mouse \
    --method cl
```

### Evaluation Parameters:
- `--cfg`: Config file for the training species
- `--model_path`: Path to the trained model (.pth file)
- `--eval_species`: Species to evaluate on (human or mouse)
- `--method`: Learning method (cl or sv)

### Results:
- Training results: `results/runs/cross_species_{species}_{method}/`
- Cross-species evaluation results: `results/runs/cross_species_{species}_{method}/cross_species_*_eval_{eval_species}/`

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

