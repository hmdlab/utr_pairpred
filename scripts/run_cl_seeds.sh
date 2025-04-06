CONFIG=/home/ksuga/whole_mrna_predictor/UTR_PairPred/config/human/contrastive_learning_10foldval_rinalmo.yaml
CONFIG1=/home/ksuga/whole_mrna_predictor/UTR_PairPred/config/human/contrastive_learning_10foldval.yaml
CONFIG2=/home/ksuga/whole_mrna_predictor/UTR_PairPred/config/human/mlp_split_large_ave_10fold.yaml
CONFIG3=/home/ksuga/whole_mrna_predictor/UTR_PairPred/config/human/mlp_split_large_rinalmo_10fold.yaml

for CFG in ${CONFIG1} ${CONFIG2} ${CONFIG3}
do
	for i in 1 2 3 4 5
	do
	python run_train_CL.py --cfg ${CFG} --seed ${i}
	done
done
