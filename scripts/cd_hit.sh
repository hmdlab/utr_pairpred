IDENTITY=0.9

# For human

cd-hit   -i ../data/human/gencode_v44_gene_unique_5utr.fa \
		 -o ../results/cdhit/cdhit_gencode44_thresh09_5utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 2000 \
		 -c $IDENTITY \

cd-hit   -i ../data/human/gencode_v44_gene_unique_3utr.fa \
		 -o ../results/cdhit/cdhit_gencode44_thresh09_3utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 2000 \
		 -c $IDENTITY

# For mouse

cd-hit   -i ../data/mouse/gencode_vM33_utr_gene_unique_5utr.fa \
		 -o ../results/cdhit/cdhit_gencode_vM33_thresh09_5utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 3000 \
		 -c $IDENTITY \

cd-hit   -i ../data/mouse/gencode_vM33_utr_gene_unique_3utr.fa \
		 -o ../results/cdhit/cdhit_gencode_vM33_thresh09_3utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 3000 \
		 -c $IDENTITY
