from model import PairPredMLP, PairPredMLP_split, PairPredCNN, PairPredCNN_small

MODEL_DICT = {
    "mlp": PairPredMLP,
    "mlp_split": PairPredMLP_split,
    "cnn": PairPredCNN,
    "cnn_small": PairPredCNN_small,
}
