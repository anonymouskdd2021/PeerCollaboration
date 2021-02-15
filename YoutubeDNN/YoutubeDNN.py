import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, hidden_sizes, dropout, use_bn=False):
        super().__init__()
        layers = list()
        self.use_bn = use_bn
        self.hidden_size = input_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        if self.use_bn:
            batch_size = x.size(0)
            x = x.view(-1, self.hidden_size)
            out = self.mlp(x).view(batch_size, -1, self.hidden_size)
        else:
            out = self.mlp(x)
        return out


class YouTuBeDNN(nn.Module):
    def __init__(self, hidden_size, item_size, layer_num=1, pad_id=0, dropouts=(0, 0), use_bn=False):
        super(YouTuBeDNN, self).__init__()
        self.embedding = nn.Embedding(item_size, hidden_size, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropouts[0])

        self.layer_num = layer_num

        self.fcs = MultiLayerPerceptron(hidden_size, [hidden_size]*self.layer_num, dropout=dropouts[1], use_bn=use_bn)
        self.final = nn.Linear(hidden_size, item_size)

    def forward(self, inputs):
        x = self.dropout(self.embedding(inputs))
        h = self.fcs(x)
        out = self.final(torch.mean(h, dim=1, keepdim=False))
        return out