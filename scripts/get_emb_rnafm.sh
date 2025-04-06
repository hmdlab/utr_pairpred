# Get the embedding of the RNA-FM model

poetry run python ../preprocess/get_seq_embedding.py \
		--i ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
		--o ../data/human/_test_gencode44_embedding_forward_cdhit09.pkl \
		--over_length average \


poetry run python ../preprocess/get_seq_embedding.py \
		--i ../data/mouse/gencode_vM33_utr_gene_unique_cdhit09.csv \
		--o ../data/mouse/_test_gencode_vM33_embedding_ave_cdhit09.pkl \
		--over_length average \
