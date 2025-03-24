from model import (
    PairPredCNN,
    PairPredCNN_small,
    PairPredConcatSelfAttn,
    PairPredCR,
    PairPredCrossAttn,
    PairPredMLP,
    PairPredMLP_split,
    PairPredMLP_split_large,
    PairPredMLP_split_skip,
)

MODEL_DICT = {
    "mlp": PairPredMLP,
    "mlp_split": PairPredMLP_split,
    "mlp_split_large": PairPredMLP_split_large,
    "mlp_split_skip": PairPredMLP_split_skip,
    "cnn": PairPredCNN,
    "cnn_small": PairPredCNN_small,
    "contrastive": PairPredCR,
    "crossattn": PairPredCrossAttn,
    "concatselfattn": PairPredConcatSelfAttn,
}
