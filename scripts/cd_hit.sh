IDENTITY=0.8
cd-hit   -i ../data/human/gencode44_gene_unique_5utr.fa \
		 -o ../results/cdhit/cdhit_gencode44_thresh08_5utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 2000 \
		 -c $IDENTITY \

cd-hit   -i ../data/human/gencode44_gene_unique_3utr.fa \
		 -o ../results/cdhit/cdhit_gencode44_thresh08_3utr.fa \
		 -n 5 \
		 -T 0 \
		 -M 2000 \
		 -c $IDENTITY
