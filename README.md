# UTR_PairPred
## Data preprocess
1. Download gencode `Protein-coding transcript sequences` fasta file from [here](https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.pc_transcripts.fa.gz)
2. run code like below
```python
python seq_preprocess.py --file_path <path to FASTA> --output <path to save result .csv>
```
