IDENTITY=0.9

cd-hit   -i ../data/mouse/gencode_vM33_utr_gene_unique_5utr.fa \
		 -o ../results/cdhit/cdhit_gencode_vM33_thresh08_5utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 3000 \
		 -c $IDENTITY \

cd-hit   -i ../data/mouse/gencode_vM33_utr_gene_unique_3utr.fa \
		 -o ../results/cdhit/cdhit_gencode_vM33_thresh08_3utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 3000 \
		 -c $IDENTITY
