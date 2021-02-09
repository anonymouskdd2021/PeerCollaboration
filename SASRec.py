import torch
import torch.nn as nn
import argparse
from utils import normalize
from Modules import multihead_attention, feedforward


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate=0.5):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.SelfAttention = multihead_attention(num_units=self.hidden_size,
                                                 num_heads=self.num_heads, dropout_rate=self.dropout_rate,
                                                 causality=True, with_qk=False, hidden_size=self.hidden_size)
        self.ff = feedforward(num_units=[self.hidden_size, self.hidden_size], dropout_rate=self.dropout_rate)

    def forward(self, input):
        x = self.SelfAttention(queries=input, keys=input)
        out = self.ff(x)
        return out


class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, max_len, padid, num_blocks, num_heads, dropout_rate=0.5, device='gpu'):
        super(SASRec, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.max_len = max_len
        self.padid = padid
        self.device = torch.device(device)
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate=dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=self.item_num,
            embedding_dim=self.hidden_size,
            padding_idx=self.padid
        )
        self.pos_embeddings = nn.Embedding(
            num_embeddings=self.max_len,
            embedding_dim=self.hidden_size,
        )
        
        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        nn.init.normal_(self.pos_embeddings.weight, 0, 0.01)
        
        rb = [TransformerLayer(hidden_size=self.hidden_size,
                               num_heads=self.num_heads,dropout_rate=self.dropout_rate) for _ in range(self.num_blocks)]
        self.transformers = nn.Sequential(*rb)

        #dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        
        #softmax Layer
        self.final = nn.Linear(self.hidden_size, self.item_num)
        
    def forward(self, inputs, onecall=False):
        input_emb = self.item_embeddings(inputs)
        pos_emb_input = torch.cat(inputs.size(0)*[torch.arange(start=0,end=inputs.size(1)).to(self.device).unsqueeze(0)])
        pos_emb_input = pos_emb_input.long()
        pos_emb = self.pos_embeddings(pos_emb_input)
        x = input_emb+pos_emb

        x = self.dropout(x)

        x = self.transformers(x)

        if onecall:
            x = x[:, -1, :].view(-1, self.hidden_size) # [batch_size, hidden_size]
        else:
            x = x.view(-1, self.hidden_size) # [batch_size*seq_len, hidden_size]

        out = self.final(x)
        return out
