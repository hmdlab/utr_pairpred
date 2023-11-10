# UTR_PairPred
## Data preprocess
1. Download gencode `Protein-coding transcript sequences` fasta file from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.pc_transcripts.fa.gz)
2. Createing sequence df with filtering.
- run code below in `scripts` dir.
- you need to change some arguments as you want.
```linux
cd scripts
sh create_seq_df.sh
```
3. Getting sequence embedding
```linux
sh get_emb.sh
```

## Training prediction models
1. Create and put config in `config` dir.

2. Training with this code
```linux
python trainer.py --cfg <YOUR_CONFIG_PATH>
```
- Results will be saved in wandb.
