export CUDA_VISIBLE_DEVICES=1

python ../preprocess/get_seq_embedding.py \
		--i ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
		--o ../data/human/gencode44_embedding_rinalmo_cdhit09.pkl \
		--rinalmo
