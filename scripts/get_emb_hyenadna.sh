export CUDA_VISIBLE_DEVICES=3

python ../preprocess/get_seq_embedding.py \
		--i ../data/gencode44_utr_gene_unique_max32k.csv \
		--o ../data/gencode44_embedding_hyenadna_32k.pkl \
		--hyenadna hyenadna-small-32k-seqlen
