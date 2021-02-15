# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class multihead_attention(nn.Module):
    def __init__(self, hidden_size, num_units=None, num_heads=8, dropout_rate=0, causality=True,
                        with_qk=False):
        super(multihead_attention, self).__init__()
        self.num_units=num_units
        self.num_heads = num_heads
        self.dropout_rate=dropout_rate
        self.causality = causality
        self.with_qk=with_qk
        self.hidden_size=hidden_size
        self.fc1 = nn.Linear(self.hidden_size, num_units)
        self.fc2 = nn.Linear(self.hidden_size, num_units)
        self.fc3 = nn.Linear(self.hidden_size, num_units)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units)
    '''Applies multihead attention.
    
    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''
    # Set the fall back option for num_units
    
    def forward(self, queries, keys):
        if self.num_units is None:
            self.num_units = queries.size(-1)
            # Linear projections
        
        Q = self.fc1(queries) # (N, T_q, C)
        K = self.fc2(keys) # (N, T_k, C)
        V = self.fc3(keys) # (N, T_k, C)
    
        # Split and concat
        q_split = int(Q.size(2)/self.num_heads)
        k_split = int(K.size(2)/self.num_heads)
        v_split = int(V.size(2)/self.num_heads)
        Q_ = torch.cat(torch.split(Q, q_split, dim=2), dim=0) # (h*N, T_q, C/h) 
        K_ = torch.cat(torch.split(K, k_split, dim=2), dim=0) # (h*N, T_k, C/h) 
        V_ = torch.cat(torch.split(V, v_split, dim=2), dim=0) # (h*N, T_k, C/h) 

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute(0, 2, 1)) # (h*N, T_q, T_k)
        
        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)
        
        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, -1))) # (N, T_k)
        key_masks = torch.cat(self.num_heads * [key_masks]) # (h*N, T_k)
        key_masks = torch.cat(queries.size(1) * [key_masks.unsqueeze(1)], dim=1) # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs)*(-2**32+1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :]) # (T_q, T_k)
            tril = torch.tril(diag_vals) # (T_q, T_k)
            masks = torch.cat(outputs.size(0)*[tril.unsqueeze(0)]) # (h*N, T_q, T_k)

            paddings = torch.ones_like(masks)*(-2**32+1)
            outputs = torch.where(torch.eq(masks, 0), paddings, outputs) # (h*N, T_q, T_k)
  
        # Activation
        outputs = self.softmax(outputs) # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries,-1))) # (N, T_q)
        query_masks = torch.cat(self.num_heads*[query_masks]) # (h*N, T_q)
        query_masks = torch.cat(keys.size(1)*[query_masks.unsqueeze(-1)], dim=2) # (h*N, T_q, T_k)
        outputs = outputs * query_masks # broadcasting. (N, T_q, C)
    
        # Dropouts
        
        outputs = self.dropout(outputs)
               
        # Weighted sum
        outputs = torch.matmul(outputs, V_) # ( h*N, T_q, C/h)
        
        # Restore shape
        o_split = int(outputs.size(0)/self.num_heads)
        outputs = torch.cat(torch.split(outputs, o_split, dim=0), dim=2) # (N, T_q, C)
              
        # Residual connection
        outputs += queries

        # Normalize
        outputs = self.layer_norm(outputs) # (N, T_q, C)
 
        if self.with_qk: return Q,K
        else: return outputs
        
        
class feedforward(nn.Module):

    def __init__(self, num_units, dropout_rate=0.5):
        super(feedforward,self).__init__()
        self.inner_cnn = nn.Conv1d(num_units[0], num_units[0],1)
        self.readout_cnn = nn.Conv1d(num_units[0],num_units[1],1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units[1])
        
    def forward (self,inputs):
        residual = inputs
        x = inputs.transpose(1, 2) # [N, C, T_q]
        x = F.relu(self.inner_cnn(x))
        x = self.dropout(x)
        x = self.readout_cnn(x)
        x = x.transpose(1, 2)  # [N, C, T_q]
        x = self.dropout(x)
        x += residual
        outputs = self.layer_norm(x)
        return outputs


class MaskedEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super(MaskedEmbedding, self).__init__(num_embeddings, embedding_dim, padding_idx)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.embedding(x, weight)
        else:
            return F.embedding(x, self.weight)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.conv1d(x, weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv1d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class mask_multihead_attention(nn.Module):
    def __init__(self, hidden_size, num_units=None, num_heads=8, dropout_rate=0, causality=True,
                 with_qk=False):
        super(mask_multihead_attention, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.with_qk = with_qk
        self.hidden_size = hidden_size
        self.fc1 = MaskedLinear(self.hidden_size, num_units)
        self.fc2 = MaskedLinear(self.hidden_size, num_units)
        self.fc3 = MaskedLinear(self.hidden_size, num_units)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units)

    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    # Set the fall back option for num_units

    def forward(self, queries, keys):
        if self.num_units is None:
            self.num_units = queries.size(-1)
            # Linear projections

        Q = self.fc1(queries)  # (N, T_q, C)
        K = self.fc2(keys)  # (N, T_k, C)
        V = self.fc3(keys)  # (N, T_k, C)

        # Split and concat
        q_split = int(Q.size(2) / self.num_heads)
        k_split = int(K.size(2) / self.num_heads)
        v_split = int(V.size(2) / self.num_heads)
        Q_ = torch.cat(torch.split(Q, q_split, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, k_split, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, v_split, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, -1)))  # (N, T_k)
        key_masks = torch.cat(self.num_heads * [key_masks])  # (h*N, T_k)
        key_masks = torch.cat(queries.size(1) * [key_masks.unsqueeze(1)], dim=1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            masks = torch.cat(outputs.size(0) * [tril.unsqueeze(0)])  # (h*N, T_q, T_k)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.eq(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = self.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = torch.sign(torch.abs(torch.sum(queries, -1)))  # (N, T_q)
        query_masks = torch.cat(self.num_heads * [query_masks])  # (h*N, T_q)
        query_masks = torch.cat(keys.size(1) * [query_masks.unsqueeze(-1)], dim=2)  # (h*N, T_q, T_k)
        outputs = outputs * query_masks  # broadcasting. (N, T_q, C)

        # Dropouts

        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        o_split = int(outputs.size(0) / self.num_heads)
        outputs = torch.cat(torch.split(outputs, o_split, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = self.layer_norm(outputs)  # (N, T_q, C)

        if self.with_qk:
            return Q, K
        else:
            return outputs

    def set_masks(self, masks):
        self.fc1.set_mask(masks[0])
        self.fc2.set_mask(masks[1])
        self.fc3.set_mask(masks[2])


class mask_feedforward(nn.Module):

    def __init__(self, num_units, dropout_rate=0.5):
        super(mask_feedforward, self).__init__()
        self.inner_cnn = MaskedConv1d(num_units[0], num_units[0], 1)
        self.readout_cnn = MaskedConv1d(num_units[0], num_units[1], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units[1])

    def forward(self, inputs):
        residual = inputs
        x = inputs.transpose(1, 2)  # [N, C, T_q]
        x = F.relu(self.inner_cnn(x))
        x = self.dropout(x)
        x = self.readout_cnn(x)
        x = x.transpose(1, 2)  # [N, C, T_q]
        x = self.dropout(x)
        x += residual
        outputs = self.layer_norm(x)
        return outputs

    def set_masks(self, masks):
        self.inner_cnn.set_mask(masks[0])
        self.readout_cnn.set_mask(masks[1])


class multihead_attention_rezero(nn.Module):
    def __init__(self, hidden_size, num_units=None, num_heads=8, dropout_rate=0, causality=True,
                 with_qk=False):
        super(multihead_attention_rezero, self).__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.causality = causality
        self.with_qk = with_qk
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(self.hidden_size, num_units)
        self.fc2 = nn.Linear(self.hidden_size, num_units)
        self.fc3 = nn.Linear(self.hidden_size, num_units)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units)
        self.rez = nn.Parameter(torch.zeros(1))

    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    # Set the fall back option for num_units

    def forward(self, queries, keys):
        if self.num_units is None:
            self.num_units = queries.size(-1)
            # Linear projections

        Q = self.fc1(queries)  # (N, T_q, C)
        K = self.fc2(keys)  # (N, T_k, C)
        V = self.fc3(keys)  # (N, T_k, C)

        # Split and concat
        q_split = int(Q.size(2) / self.num_heads)
        k_split = int(K.size(2) / self.num_heads)
        v_split = int(V.size(2) / self.num_heads)
        Q_ = torch.cat(torch.split(Q, q_split, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, k_split, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, v_split, dim=2), dim=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = torch.matmul(Q_, K_.permute(0, 2, 1))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.abs(torch.sum(keys, -1)))  # (N, T_k)
        key_masks = torch.cat(self.num_heads * [key_masks])  # (h*N, T_k)
        key_masks = torch.cat(queries.size(1) * [key_masks.unsqueeze(1)], dim=1)  # (h*N, T_q, T_k)

        paddings = torch.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = torch.where(torch.eq(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if self.causality:
            diag_vals = torch.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = torch.tril(diag_vals)  # (T_q, T_k)
            masks = torch.cat(outputs.size(0) * [tril.unsqueeze(0)])  # (h*N, T_q, T_k)

            paddings = torch.ones_like(masks) * (-2 ** 32 + 1)
            outputs = torch.where(torch.eq(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = self.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        # query_masks = torch.sign(torch.abs(torch.sum(queries,-1))) # (N, T_q)
        # query_masks = torch.cat(self.num_heads*[query_masks]) # (h*N, T_q)
        # query_masks = torch.cat(keys.size(1)*[query_masks.unsqueeze(-1)], dim=2) # (h*N, T_q, T_k)
        # outputs *= query_masks # broadcasting. (N, T_q, C)

        # Dropouts

        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        o_split = int(outputs.size(0) / self.num_heads)
        outputs = torch.cat(torch.split(outputs, o_split, dim=0), dim=2)  # (N, T_q, C)

        # Residual connection
        outputs = queries + outputs * self.rez

        # Normalize
        outputs = self.layer_norm(outputs)  # (N, T_q, C)

        if self.with_qk:
            return Q, K
        else:
            return outputs


class feedforward_rezero(nn.Module):

    def __init__(self, num_units, dropout_rate=0.5):
        super(feedforward_rezero, self).__init__()
        self.inner_cnn = nn.Conv1d(num_units[0], num_units[0], 1)
        self.readout_cnn = nn.Conv1d(num_units[0], num_units[1], 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(num_units[1])
        self.rez = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        residual = inputs
        x = inputs.transpose(1, 2)  # [N, C, T_q]
        x = F.relu(self.inner_cnn(x))
        x = self.dropout(x)
        x = self.readout_cnn(x)
        x = x.transpose(1, 2)  # [N, C, T_q]
        x = self.dropout(x)
        x = residual + x * self.rez
        outputs = self.layer_norm(x)
        return outputs


def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)
