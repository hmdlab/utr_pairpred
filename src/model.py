"""Model class for simple MLP"""
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
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


class PairPredMLP_split_large(nn.Module):
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
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc_common2_out),
            nn.Dropout(cfg.dropout_fc_common2),
        )

        self.fc_common3 = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.fc_common3_in, out_features=self.cfg.fc_common3_out
            ),
        )

    def forward(self, inputs):
        x_5utr, x_3utr = inputs[0], inputs[1]
        x_5utr = self.fc5utr(x_5utr)
        x_3utr = self.fc3utr(x_3utr)
        x = torch.cat([x_5utr, x_3utr], dim=1)
        x = self.fc_common1(x)
        x = self.fc_common2(x)
        x = self.fc_common3(x)
        return x


class PairPredMLP_split_skip(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.cfg = cfg
        self.fc5utr = SkipNetwork(cfg)
        self.fc3utr = SkipNetwork(cfg)

        self.fc_common1 = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.fc_common1_in, out_features=self.cfg.fc_common1_out
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.cfg.fc_common1_out),
            nn.Dropout(cfg.dropout_fc_common1),
        )

        self.fc_output = nn.Sequential(
            nn.Linear(
                in_features=self.cfg.fc_output_in, out_features=self.cfg.fc_output_out
            ),
        )

    def forward(self, inputs):
        x_5utr, x_3utr = inputs[0], inputs[1]
        x_5utr = self.fc5utr(x_5utr)
        x_3utr = self.fc3utr(x_3utr)
        x = torch.cat([x_5utr, x_3utr], dim=1)
        x = self.fc_common1(x)
        x = self.fc_output(x)
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


# Belows are extracted from https://gist.github.com/yonedahayato/17b9dac98cdb77ea82fec1ea6516d94c#file-re_training_clip-ipynb
# Used for PairPred Contrastive Learning.
class SkipBlock(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        input_dim = cfg.input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=input_dim),
            nn.Dropout(cfg.dropout2),
        )

    def forward(self, x):
        h = self.fc(x)
        h = torch.add(h, x)
        h = F.relu(h)

        return h


class SkipNetwork(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        input_dim = cfg.input_dim
        output_dim = cfg.output_dim

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=input_dim),
            nn.Dropout(cfg.dropout2),
        )

        self.block1 = SkipBlock(cfg)
        self.block2 = SkipBlock(cfg)

        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=output_dim),
            nn.Dropout(cfg.dropout2),
        )

    def forward(self, x):
        x = self.fc1(x)

        x = self.block1(x)
        x = self.block2(x)

        x = self.fc2(x)

        return x


class PairPredCR(nn.Module):
    """Model for PairPred contrastive learning."""

    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.network_utr5 = SkipNetwork(cfg)
        self.network_utr3 = SkipNetwork(cfg)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, inputs: tuple) -> tuple:
        logit_scale = self.logit_scale.exp()
        utr5_features = self.network_utr5(inputs[0])
        utr3_features = self.network_utr3(inputs[1])

        ## Normalize to calculate cos similarity [-1 to 1]
        utr5_features = utr5_features / utr5_features.norm(dim=1, keepdim=True)
        utr3_features = utr3_features / utr3_features.norm(dim=1, keepdim=True)

        logits_per_utr5 = logit_scale * utr5_features @ utr3_features.T
        logits_per_utr3 = logit_scale * utr3_features @ utr5_features.T

        return (logits_per_utr5, logits_per_utr3)

    def predict(self, inputs: tuple) -> tuple:
        """without scaling"""
        utr5_features_ori = self.network_utr5(inputs[0])
        utr3_features_ori = self.network_utr3(inputs[1])

        ## Normalize to calculate cos similarity [-1 to 1]
        utr5_features = utr5_features_ori / utr5_features_ori.norm(dim=1, keepdim=True)
        utr3_features = utr3_features_ori / utr3_features_ori.norm(dim=1, keepdim=True)

        logits_per_utr5 = utr5_features @ utr3_features.T
        logits_per_utr3 = utr3_features @ utr5_features.T

        return (logits_per_utr5, logits_per_utr3), (
            utr5_features_ori,
            utr3_features_ori,
        )


# Belows are extracted from https://github.com/Hhhzj-7/DeepCoVDR/blob/main/enceoder.py
class FFN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        input_dim = cfg.attn_dim
        hidden_dim = cfg.hidden_dim
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class ConcatSelfAttn(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.multi_head_attn = MultiheadAttention(
            embed_dim=cfg.attn_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout_attn,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(cfg.dropout_attn)
        self.layernorm_attn = nn.LayerNorm(cfg.attn_dim, eps=float(cfg.layer_norm_eps))

        self.ffn = FFN(cfg)
        self.dropout_ffn = nn.Dropout(cfg.dropout_ffn)
        self.layernorm_ffn = nn.LayerNorm(cfg.attn_dim, eps=float(cfg.layer_norm_eps))

    def forward(self, inputs: tuple) -> torch.Tensor:
        ## Multi head attention module
        attn_out, _ = self.multi_head_attn(inputs, inputs, inputs)
        attn_out = self.dropout_attn(attn_out)
        attn_out = self.layernorm_attn(attn_out + inputs)

        ## FeedForward module
        ffn_out = self.ffn(attn_out)
        ffn_out = self.dropout_ffn(ffn_out)
        ffn_out = self.layernorm_ffn(ffn_out + attn_out)  # (bs,seq_len,hidden_dim)

        return ffn_out


class CrossAttn(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.multi_head_attn = MultiheadAttention(
            embed_dim=cfg.attn_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout_attn,
            batch_first=True,
        )
        self.dropout_attn = nn.Dropout(cfg.dropout_attn)
        self.layernorm_attn = nn.LayerNorm(cfg.attn_dim, eps=float(cfg.layer_norm_eps))

        self.ffn = FFN(cfg)
        self.dropout_ffn = nn.Dropout(cfg.dropout_ffn)
        self.layernorm_ffn = nn.LayerNorm(cfg.attn_dim, eps=float(cfg.layer_norm_eps))

    def forward(self, inputs: tuple) -> torch.Tensor:
        input_query = inputs[0]
        input_key = inputs[1]
        ## Multi head attention module
        attn_out, _ = self.multi_head_attn(input_query, input_key, input_key)
        attn_out = self.dropout_attn(attn_out)
        attn_out = self.layernorm_attn(attn_out + input_query)  # [length,bs,hidden_dim]

        ## FeedForward module
        ffn_out = self.ffn(attn_out)
        ffn_out = self.dropout_ffn(ffn_out)
        ffn_out = self.layernorm_ffn(ffn_out + attn_out)  # (bs,seq_len,hidden_dim)

        return ffn_out


class PairPredCrossAttn(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()
        self.crossattn_utr5 = CrossAttn(cfg)
        self.crossattn_utr3 = CrossAttn(cfg)

        self.pooler = nn.MaxPool1d(kernel_size=cfg.attn_dim)

        self.head1 = nn.Sequential(
            nn.Linear(in_features=cfg.fc1_in, out_features=cfg.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=cfg.fc1_out),
            nn.Dropout(cfg.dropout_head),
        )

        self.head2 = nn.Sequential(
            nn.Linear(in_features=cfg.fc2_in, out_features=cfg.fc2_out)
        )

    def forward(self, inputs):
        attn_out_utr5 = self.crossattn_utr5((inputs[0], inputs[1]))
        attn_out_utr3 = self.crossattn_utr3((inputs[1], inputs[0]))

        attn_out_utr5 = self.pooler(attn_out_utr5.transpose(1, 2)).squeeze()
        attn_out_utr3 = self.pooler(attn_out_utr3.transpose(1, 2)).squeeze()

        x = torch.concat([attn_out_utr5, attn_out_utr3], dim=1)
        x = self.head1(x)
        x = self.head2(x)

        return x


class PairPredConcatSelfAttn(nn.Module):
    def __init__(self, cfg: AttrDict):
        super().__init__()

        self.concatattn = nn.ModuleList(
            ConcatSelfAttn(cfg) for _ in range(cfg.num_layers)
        )

        self.head1 = nn.Sequential(
            nn.Linear(in_features=cfg.fc1_in, out_features=cfg.fc1_out),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=cfg.fc1_out),
            nn.Dropout(cfg.dropout_head),
        )

        self.head2 = nn.Sequential(
            nn.Linear(in_features=cfg.fc2_in, out_features=cfg.fc2_out)
        )

    def forward(self, inputs):
        for layer in self.concatattn:
            x = layer(inputs)  # [bs,seq_len*2,hidden_dim]
        x = F.adaptive_avg_pool1d(x.transpose(1, 2), output_size=1).squeeze()
        x = self.head1(x)
        x = self.head2(x)

        return x
