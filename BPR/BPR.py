import torch
import torch.nn as nn
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay, layer_num=0, reg_type='part'):
        super().__init__()
        self.W = nn.Embedding(user_size, dim)
        self.W_layer = nn.Sequential(*[nn.Linear(dim, dim) for _ in range(layer_num)])
        self.H = nn.Embedding(item_size, dim)
        self.H_layer = nn.Sequential(*[nn.Linear(dim, dim) for _ in range(layer_num)])
        # nn.init.xavier_normal_(self.W.weight.data)
        # nn.init.xavier_normal_(self.H.weight.data)
        nn.init.normal_(self.W.weight, 0, 0.01)
        nn.init.normal_(self.H.weight, 0, 0.01)
        self.weight_decay = weight_decay
        self.reg_type = reg_type

    def forward(self, u, i, j):
        """Return loss value.
        
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]
        
        Returns:
            torch.FloatTensor
        """
        u_emb = self.W(u)
        u_emb = self.W_layer(u_emb)
        i_emb = self.H(i)
        j_emb = self.H(j)
        i_emb = self.H_layer(i_emb)
        j_emb = self.H_layer(j_emb)
        x_ui = torch.mul(u_emb, i_emb).sum(dim=1)
        x_uj = torch.mul(u_emb, j_emb).sum(dim=1)
        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        out = -log_prob
        if self.reg_type != 'all':
            out += self.weight_decay * (u_emb.norm(dim=1).pow(2).sum() + i_emb.norm(dim=1).pow(2).sum()
                                         + j_emb.norm(dim=1).pow(2).sum())
        return out

    def predict(self, u):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W(u)
        u_rep = self.W_layer(u)
        all_items_rep = self.H_layer(self.H.weight)
        scores = torch.mm(u_rep, all_items_rep.t()) # [batch_size, item_size]
        return scores

    def predict_top(self, u, i):
        """Return recommended item list given users.
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W(u)
        x_ui = torch.mm(u, self.H.t())

        pred = torch.argsort(x_ui, dim=1)
        return pred
