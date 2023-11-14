SPECIES=Danio_rerio
RELEASE=110
REF_NAME=GRCz11


python ../preprocess/create_seq_df.py \
		--ensembl_dir ../data/$SPECIES \
		--ref_name $REF_NAME \
		--gtf $SPECIES.$REF_NAME.$RELEASE.gtf \
		--fasta $SPECIES.$REF_NAME.cdna.all.fa \
		--output ../data/$SPECIES/ensembl.$SPECIES.$RELEASE.utr_gene_unique.csv
