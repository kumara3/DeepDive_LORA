import os
from pathlib import Path
import urllib
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.gamma * x_norm + self.beta

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.net(x)

class MultiheadedAttn(nn.Module):
    def __init__(self,d_in,d_out,context_length,num_head,dropout,qkv_bias):
        super().__init__()
        assert d_out % num_head == 0, "d_out must be divisible by num_head"
        self.num_head = num_head
        self.head_dim = d_out // num_head
        self.d_out = d_out

        self.W_q = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_k = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.W_v = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # masked matrix to attend only previous and current tokens


    def forward(self,x):
        batch,num_tokens,d_in = x.shape

        queries = self.W_q(x)
        keys = self.W_k(x)
        values = self.W_v(x)

        # unroll last dimension into (num_head, head_dim)
        queries = queries.view(batch,num_tokens,self.num_head,self.head_dim)
        keys = keys.view(batch,num_tokens,self.num_head,self.head_dim)
        values = values.view(batch,num_tokens,self.num_head,self.head_dim)

        # transpose to get dimensions: (batch, num_head, num_tokens, head_dim)
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        # compute self-attention scores
        attn_scores = queries @ keys.transpose(2,3)

        # mask attention scores to prevent attending to future tokens
        mask_bolean = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bolean, float('-inf'))
        attn_weights = torch.softmax(attn_scores / keys.shape(-1)**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vectors = (attn_weights @ values).transpose(1,2)

        # combine heads
        context_vectors = context_vectors.reshape(batch,num_tokens,self.d_out)
        context_vectors = self.out(context_vectors)
        return context_vectors

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiheadedAttn(
            d_in=config["emb_dim"],
            d_out=config["emd_dim"],
            context_length=config["context_length"],
            num_head=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_out = nn.Dropout(config["drop_rate"])

    def forward(self,x):
        # Multi-Head Attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_out(x)
        x = x + shortcut
        # Feed Forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_out(x)
        x = x + shortcut
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.position_embedding = nn.Embedding(config["context_length"],config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["num_layers"])])
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self,idx):
        batch_size, seq_len = idx.shape
        token_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(torch.arange(seq_len,device=idx.device))
        x = token_embed + pos_embed
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out(x)
        return logits
