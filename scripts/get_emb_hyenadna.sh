export CUDA_VISIBLE_DEVICES=3

python ../preprocess/get_seq_embedding.py \
		--i ../data/gencode_vM33_utr_gene_unique_max32k.csv \
		--o ../data/gencode_vM33_embedding_hyenadna_ave_32k.pkl \
		--hyenadna hyenadna-small-32k-seqlen
