# Create sequence df for easy analysis from GENCODE raw fasta files

python ../preprocess/create_seq_df.py \
		--file_path ../data/gencode.v44.pc_transcripts.fa \
		--output ../data/gencode44_utr_gene_unique.csv \


python ../preprocess/create_seq_df.py \
		--file_path ../data/gencode.vM33.pc_transcripts.fa \
		--output ../data/gencode_vM33_utr_gene_unique.csv \
