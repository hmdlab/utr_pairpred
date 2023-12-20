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


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads  # multi-heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.query2 = nn.Linear(hidden_size, self.all_head_size)
        self.key2 = nn.Linear(hidden_size, self.all_head_size)
        self.value2 = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # [bs,seq_len,hidden_dim,num_head]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [bs,hidden_dim,seq_len,num_head]

    def forward(self, hidden_states):
        # for cross-attention module
        # Q, K, V for 5utr cross-attention
        query_5utr = self.transpose_for_scores(self.query(hidden_states[0]))
        key_5utr = self.transpose_for_scores(self.key(hidden_states[0]))
        value_5utr = self.transpose_for_scores(self.value(hidden_states[0]))

        # Q, K, V for 3utr in cross-attention
        query_3utr = self.transpose_for_scores(self.query2(hidden_states[0]))
        key_3utr = self.transpose_for_scores(self.key2(hidden_states[0]))
        value_3utr = self.transpose_for_scores(self.value2(hidden_states[0]))

        # attention scores for 5UTR
        attention_scores_5utr = torch.matmul(query_5utr, key_3utr.transpose(-1, -2))
        attention_scores_5utr = attention_scores_5utr / math.sqrt(
            self.attention_head_size
        )
        attention_probs_5utr = nn.Softmax(dim=-1)(attention_scores_5utr)
        attention_probs_5utr = self.dropout(attention_probs_5utr)
        attention_output_5utr = torch.matmul(attention_probs_5utr, value_3utr)
        attention_output_5utr = attention_output_5utr.permute(0, 2, 1, 3).contiguous()

        new_attention_shape = attention_output_5utr.size()[:-2] + (self.all_head_size,)
        attention_output_5utr = attention_output_5utr.view(*new_attention_shape)

        # attention scores for 3UTR
        attention_scores_3utr = torch.matmul(query_3utr, key_5utr.transpose(-1, -2))
        attention_scores_3utr = attention_scores_3utr / math.sqrt(
            self.attention_head_size
        )
        attention_probs_3utr = nn.Softmax(dim=-1)(attention_scores_3utr)
        attention_probs_3utr = self.dropout(attention_probs_3utr)
        attention_output_3utr = torch.matmul(attention_probs_3utr, value_5utr)

        attention_output_3utr = attention_output_3utr.permute(0, 2, 1, 3).contiguous()

        new_attention_shape = attention_output_3utr.size()[:-2] + (self.all_head_size,)
        attention_output_3utr = attention_output_3utr.view(*new_attention_shape)

        return (attention_output_5utr, attention_output_3utr)


# output of self-attention
class CrossAttnOutput(nn.Module):
    """LayerNorm + Skip Connection"""

    def __init__(self, hidden_size, hidden_dropout_prob):
        super(CrossAttnOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(Attention, self).__init__()
        self.selfattn = SelfAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.output_5utr = CrossAttnOutput(hidden_size, hidden_dropout_prob)
        self.output_3utr = CrossAttnOutput(hidden_size, hidden_dropout_prob)

    def forward(self, inputs):
        crossattention_output: tuple = self.selfattn(inputs)
        attention_output_5utr = self.output_5utr(crossattention_output[0], inputs[0])
        attention_output_3utr = self.output_3utr(crossattention_output[1], inputs[1])

        return (attention_output_5utr, attention_output_3utr)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_dropout_prob):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout1 = nn.Dropout(hidden_dropout_prob)
        self.dropout2 = nn.Dropout(hidden_dropout_prob)

    def forward(self, x):
        x = self.linear1(x)  # hidden_size -> intermed_size
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.linear2(x)  # intermed_size -> hidden_size
        x = self.dropout2(x)
        return x


# output
class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        num_attention_heads,
        attention_probs_dropout_prob,
        hidden_dropout_prob,
    ):
        super(TransformerEncoder, self).__init__()
        self.attention = Attention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_prob,
            hidden_dropout_prob,
        )
        self.feedforward = FeedForward(
            hidden_size, intermediate_size, hidden_dropout_prob
        )
        self.LayerNorm = LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask, fusion):
        attention_output = self.attention(hidden_states, attention_mask, fusion)
        feedforward_output = self.feedforward(attention_output)
        layer_output = self.LayerNorm(
            feedforward_output + attention_output
        )  ## skip connection
        return layer_output
