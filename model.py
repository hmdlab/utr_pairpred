"""Model class for simple MLP"""
import torch.nn as nn


class PairPredMLP(nn.Module):
    def __init__(self, cfg):
        super.__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc1_in, out_features=self.cfg.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.layer1_in),
            nn.Dropout(p=0.02),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc2_in, out_features=self.cfg.fc2_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.layer2_in),
            nn.Dropout(p=0.02),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc3_in, out_features=self.cfg.fc3_out),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
