# Create sequence df for easy analysis from GENCODE raw fasta files

# For human
python ../preprocess/create_seq_df.py \
		--file_path ../data/human/gencode.v44.pc_transcripts.fa \
		--output ../data/human/gencode_v44_utr_gene_unique.csv \

# For mouse
python ../preprocess/create_seq_df.py \
		--file_path ../data/mouse/gencode.vM33.pc_transcripts.fa \
		--output ../data/mouse/gencode_vM33_utr_gene_unique.csv \
