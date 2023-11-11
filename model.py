"""Model class for simple MLP"""
import torch
import torch.nn as nn
from attrdict import AttrDict


class PairPredMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc1_in, out_features=self.cfg.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc1_out),
            nn.Dropout(cfg.dropout),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc2_in, out_features=self.cfg.fc2_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc2_out),
            nn.Dropout(cfg.dropout),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc3_in, out_features=self.cfg.fc3_out),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class PairPredMLP_split(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.cfg = cfg
        self.fc5utr = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc5utr_in, out_features=self.cfg.fc5utr_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc5utr_out),
            nn.Dropout(cfg.dropout_5utr),
        )

        self.fc3utr = nn.Sequential(
            nn.Linear(in_features=self.cfg.fc3utr_in, out_features=self.cfg.fc3utr_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc3utr_out),
            nn.Dropout(cfg.dropout_3utr),
        )

        self.fc_common1 = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.fc_common1_in, out_features=self.cfg.fc_common1_out
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc_common1_out),
            nn.Dropout(cfg.dropout_fc_common1),
        )

        self.fc_common2 = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.fc_common2_in, out_features=self.cfg.fc_common2_out
            ),
        )

    def forward(self, inputs):
        x_5utr, x_3utr = inputs[0], inputs[1]
        x_5utr = self.fc5utr(x_5utr)
        x_3utr = self.fc3utr(x_3utr)
        x = torch.cat([x_5utr, x_3utr], dim=1)
        x = self.fc_common1(x)
        x = self.fc_common2(x)
        return x


class PairPredCNN(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.in_planes = cfg.in_planes
        self.out_planes = cfg.out_planes
        self.main_planes = cfg.main_planes  # dim. int
        dropout = cfg.dropout  # float
        self.emb_cnn = nn.Sequential(
            nn.Conv1d(self.in_planes, self.main_planes, kernel_size=3, padding=1),  ## 3
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=2,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=1,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=2,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=1,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=2,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=1,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.main_planes * 1, self.out_planes),
        )

    def forward(self, x):
        """forward function"""
        output = self.emb_cnn(x)
        return output


class PairPredCNN_small(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.in_planes = cfg.in_planes
        self.out_planes = cfg.out_planes
        self.main_planes = cfg.main_planes  # dim. int
        dropout = cfg.dropout  # float
        self.emb_cnn = nn.Sequential(
            nn.Conv1d(self.in_planes, self.main_planes, kernel_size=3, padding=1),  ## 3
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=2,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            ResBlock(
                self.main_planes * 1,
                self.main_planes * 1,
                stride=1,
                dilation=1,
                conv_layer=nn.Conv1d,
                norm_layer=nn.BatchNorm1d,
            ),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.main_planes * 1, self.out_planes),
        )

    def forward(self, x):
        """forward function"""
        output = self.emb_cnn(x)
        return output


class ResBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        dilation=1,
        conv_layer=nn.Conv2d,
        norm_layer=nn.BatchNorm2d,
    ):
        super(ResBlock, self).__init__()
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(
            out_planes, out_planes, kernel_size=3, padding=dilation, bias=False
        )

        if stride > 1 or out_planes != in_planes:
            self.downsample = nn.Sequential(
                conv_layer(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_planes),
            )
        else:
            self.downsample = None

        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
