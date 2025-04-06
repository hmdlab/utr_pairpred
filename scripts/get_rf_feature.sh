# Get features for random forest model

poetry run python ../preprocess/get_seq_embedding.py \
		--i ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
		--o ../data/human/\
		--feature_craft 

poetry run python ../preprocess/get_seq_embedding.py \
		--i ../data/mouse/gencode_vM33_utr_gene_unique_cdhit09.csv \
		--o ../data/mouse\
		--feature_craft 

