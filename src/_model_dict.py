from model import (
    PairPredMLP,
    PairPredMLP_split,
    PairPredCNN,
    PairPredCNN_small,
    PairPredMLP_split_large,
    PairPredMLP_split_skip,
    PairPredCR,
    PairPredCrossAttn,
    PairPredConcatSelfAttn,
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
