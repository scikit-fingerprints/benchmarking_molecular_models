import torch
import torch.nn as nn
import torch.nn.functional as F
from x_transformers import TransformerWrapper, Encoder


def global_ap(x):
    """
    Global Average Pooling
    Input: [B, max_len, hid_dim]
    Return: [B, hid_dim]
    """
    return torch.mean(x.view(x.size(0), x.size(1), -1), dim=1)


class Xtransformer_Encoder(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(Xtransformer_Encoder, self).__init__()
        self.args = args
        self.encoder = TransformerWrapper(
            num_tokens = args.dic_size,
            max_seq_len = args.max_len,
            emb_dropout = dropout,
            attn_layers = Encoder(
                dim = args.d_model,
                depth = args.nlayers,
                heads = args.nhead
            )
        )
        self.linear = nn.Linear(args.dic_size, args.max_len)
        self.relu = nn.ReLU()

    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len]
        if src_mask is None:
            src_mask = src.gt(0).to(torch.int).bool()

        x = self.encoder(src, mask=src_mask)      # [batch_size, seq_len, dic_size]
        x = global_ap(x)                          # [batch_size, dic_size]

        out = self.linear(x)                      # [batch_size, seq_len]

        return out


class Classifier(nn.Module):
    def __init__(self, args, encoder, num_classes=1, dropout=0.1):
        super(Classifier, self).__init__()
        self.args = args
        self.encoder = encoder
        self.clf = nn.Linear(args.max_len, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        x = self.encoder(src)
        x = self.relu(self.dropout(x))

        out = self.clf(x)

        return out




