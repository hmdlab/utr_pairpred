DIR=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE91680_PCBP2
INT_FILE=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE91680_PCBP2/PCBP2_abs_0608
NAME=abs_0608

plotHeatmap -m ${INT_FILE}.gz \
			-o ${DIR}/PCBP2_heat_${NAME}.png


plotProfile -m ${INT_FILE}.gz -out ${DIR}/PCBP2_profile_${NAME}.png --plotType fill \
		   --perGroup --numPlotsPerRow 2 --plotTitle "PCBP2_profile"
