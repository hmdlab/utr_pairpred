from model import PairPredMLP, PairPredMLP_split, PairPredCNN

MODEL_DICT = {"mlp": PairPredMLP, "mlp_split": PairPredMLP_split, "cnn": PairPredCNN}
