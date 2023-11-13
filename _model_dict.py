from model import (
    PairPredMLP,
    PairPredMLP_split,
    PairPredCNN,
    PairPredCNN_small,
    PairPredMLP_split_large,
)

MODEL_DICT = {
    "mlp": PairPredMLP,
    "mlp_split": PairPredMLP_split,
    "mlp_split_large": PairPredMLP_split_large,
    "cnn": PairPredCNN,
    "cnn_small": PairPredCNN_small,
}
