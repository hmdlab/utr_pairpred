export CUDA_VISIBLE_DEVICES=1
"""
python ../preprocess/get_seq_embedding.py \
		--i ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
		--o ../data/human/gencode44_embedding_rinalmo_ave_cdhit09.pt \
		--rinalmo \
		--rinalmo_method average \
		--format pt \

python ../preprocess/get_seq_embedding.py \
		--i ../data/human/gencode44_utr_gene_unique_cdhit09.csv \
		--o ../data/human/gencode44_embedding_rinalmo_whole_ave_cdhit09.pt \
		--rinalmo \
		--rinalmo_method whole_ave \
		--format pt \
"""
python ../preprocess/get_seq_embedding.py \
		--i ../data/mouse/gencode_vM33_utr_gene_unique_cdhit09.csv \
		--o ../data/mouse/gencode_vM33_embedding_rinalmo_whole_ave_cdhit09.csv \
		--rinalmo \
		--rinalmo_method whole_ave \
		--format pt \
