DIR=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE91680_PCBP2
INT_FILE=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE91680_PCBP2/PCBP2_regions
NAME=Regions

computeMatrix scale-regions \
				-S /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE91680_PCBP2/GSM2423343_GRCh38_plus.bigWig \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE91680_PCBP2/GSM2423343_GRCh38_minus.bigWig \
				-R  /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.8-1.0_5utr.bed \
					/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.8-1.0_3utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.6-0.8_5utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.6-0.8_3utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim_all_5utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim_all_3utr.bed \
				-o ${INT_FILE}.gz\
				-a 500 \
				-b 500 \
				--sortRegions no \
				--missingDataAsZero \
				--verbose \
				-p max

plotHeatmap -m ${INT_FILE}.gz \
			-o ${DIR}/PCBP2_heat_${NAME}.png


plotProfile -m ${INT_FILE}.gz -out ${DIR}/PCBP2_profile_${NAME}.png --plotType fill \
		   --perGroup --numPlotsPerRow 2 --plotTitle "PCBP2_profile"
