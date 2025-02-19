DIR=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE177941_PABPC4
INT_FILE=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE177941_PABPC4/PABPC4_peak
OUTPUT=/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE177941_PABPC4/PCBP4_heat_cossim08_ref.png
NAME=Regions

computeMatrix scale-regions \
				-S /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE177941_PABPC4/GSM5379302_GRCh38_plus.bigWig \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/GSE177941_PABPC4/GSM5379302_GRCh38_minus.bigWig \
				-R /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.8-1.0_5utr.bed \
					/home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.8-1.0_3utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.6-0.8_5utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim0.6-0.8_3utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim_all_5utr.bed \
				   /home/ksuga/whole_mrna_predictor/UTR_PairPred/data/human/CLIP/CosSim_all_3utr.bed \
				-o ${INT_FILE}.gz\
				-a 1000 \
				-b 1000 \
				--sortRegions no \
				--missingDataAsZero \
				--verbose \
				-p max

plotHeatmap -m ${INT_FILE}.gz \
			-o ${DIR}/PABPC4_heat_${NAME}.png

plotProfile  -m ${INT_FILE}.gz -out ${DIR}/PABPC4_profile_${NAME}.png --plotType=fill \
			--perGroup --numPlotsPerRow 2 --plotTitle "PABPC4 profile"
