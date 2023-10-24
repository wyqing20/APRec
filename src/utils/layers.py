# -*- coding: UTF-8 -*-

from audioop import bias
import math
from sqlite3 import adapters
from typing import ForwardRef
from numpy.testing._private.utils import print_assert_equal
import torch
import torch.nn as nn
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, kq_same=False, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention.
        """
        self.d_model = d_model
        self.h = n_heads
        self.d_k = self.d_model // self.h
        self.kq_same = kq_same

        if not kq_same:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

    def head_split(self, x):  # get dimensions bs * h * seq_len * d_k
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        return x.view(*new_x_shape).transpose(-2, -3)

    def forward(self, q, k, v, mask=None):
        origin_shape = q.size()
        
        # perform linear operation and split into h heads
        if not self.kq_same:
            
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
       
        output,scores = self.scaled_dot_product_attention(q, k, v, self.d_k, mask)
        
        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        return output,scores
    def forward_hypter_prefix(self, q, k, v, prefix_k,prefix_q,mask=None):
        origin_shape = v.size()
        
        # perform linear operation and split into h heads
        if not self.kq_same:
            q=self.q_linear(q)
         
            q=torch.cat((prefix_q[:,:,0,:],q),dim=1)
           
            q = self.head_split(self.q_linear(q))
        else:
            q = self.head_split(self.k_linear(q))
        k=self.k_linear(k)
        k=torch.cat((prefix_k[:,:,0,:],k),dim=1)

        k = self.head_split(self.k_linear(k))
        v = self.head_split(self.v_linear(v))

        # calculate attention using function we will define next
       
        output,scores = self.scaled_dot_product_attention(q, k, v, self.d_k, mask,prefix_len=prefix_k.shape[1])
       
        # concatenate heads and put through final linear layer
        output = output.transpose(-2, -3).reshape(origin_shape)
        
        return output,scores

    @staticmethod
    def scaled_dot_product_attention(q, k, v, d_k, mask=None,prefix_len=None):
        """
        This is called by Multi-head attention object to find the values.
        """
       
        scores = torch.matmul(q, k.transpose(-2, -1)) / d_k ** 0.5  # bs, head, q_len, k_len
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -np.inf)
        
        scores = (scores - scores.max()).softmax(dim=-1)
        scores = scores.masked_fill(torch.isnan(scores), 0)

        if prefix_len is not None:
            scores=scores[:,:,prefix_len:,prefix_len:]
        
        output = torch.matmul(scores, v)  # bs, head, q_len, d_k
      
        
        return output,scores



class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, dropout=0, kq_same=False):
        super().__init__()
        """
        This is a Basic Block of Transformer. It contains one Multi-head attention object. 
        Followed by layer norm and position wise feedforward net and dropout layer.
        """
        # Multi-Head Attention Block
       
        self.masked_attn_head = MultiHeadAttention(d_model, n_heads, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
      
        context,scores = self.masked_attn_head(seq, seq, seq, mask)
        
        context = self.layer_norm1(self.dropout1(context) + seq)
        output = self.linear1(context).relu()
        output = self.linear2(output)
        output = self.layer_norm2(self.dropout2(output) + context)
        return output

    
