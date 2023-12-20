export CUDA_VISIBLE_DEVICES=3

python ../preprocess/get_seq_embedding.py \
		--i ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
		--o ../data/human/gencode44_utr_gene_unique_hyenadna_cdhit09.pkl \
		--hyenadna hyenadna-small-32k-seqlen
