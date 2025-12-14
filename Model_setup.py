import os
from pathlib import Path
import urllib
import zipfile
import math
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
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return self.scale * x_norm + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)

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
        keys = keys.view(batch,num_tokens,self.num_head,self.head_dim)
        queries = queries.view(batch,num_tokens,self.num_head,self.head_dim)
        values = values.view(batch,num_tokens,self.num_head,self.head_dim)

        # transpose to get dimensions: (batch, num_head, num_tokens, head_dim)
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        # compute self-attention scores
        attn_scores = queries @ keys.transpose(2,3)

        # mask attention scores to prevent attending to future tokens
        mask_bolean = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bolean, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vectors = (attn_weights @ values).transpose(1,2)

        # combine heads
        context_vectors = context_vectors.reshape(batch, num_tokens, self.d_out)
        context_vectors = self.out(context_vectors)
        return context_vectors

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiheadedAttn(
            d_in=config["emb_dim"],
            d_out=config["emb_dim"],
            context_length=config["context_length"],
            num_head=config["n_heads"],
            dropout=config["drop_rate"],
            qkv_bias=config["qkv_bias"])
        self.ff = FeedForward(config)
        self.norm1 = LayerNorm(config["emb_dim"])
        self.norm2 = LayerNorm(config["emb_dim"])
        self.drop_resid = nn.Dropout(config["drop_rate"])

    def forward(self,x):
        # Multi-Head Attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.drop_resid(x)
        x = x + shortcut
        # Feed Forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"],config["emb_dim"])
        self.position_embedding = nn.Embedding(config["context_length"],config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self,idx):
        batch_size, seq_len = idx.shape
        token_embed = self.token_embedding(idx)
        pos_embed = self.position_embedding(torch.arange(seq_len,device=idx.device))
        x = token_embed + pos_embed
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def assign_check(target, source):
    if target.shape != source.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, source {source.shape}")
    return torch.nn.Parameter(torch.tensor(source))

def load_weights(gpt, gpt_hf, BASE_CONFIG):
    d = gpt_hf.state_dict()
    gpt.position_embedding.weight = assign_check(gpt.position_embedding.weight, d["wpe.weight"])
    gpt.token_embedding.weight = assign_check(gpt.token_embedding.weight, d["wte.weight"]) 
    for b in range(BASE_CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_q.weight = assign_check(gpt.transformer_blocks[b].attention.W_q.weight, q_w.T)
        gpt.transformer_blocks[b].attention.W_k.weight = assign_check(gpt.transformer_blocks[b].attention.W_k.weight, k_w.T)
        gpt.transformer_blocks[b].attention.W_v.weight = assign_check(gpt.transformer_blocks[b].attention.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_q.bias = assign_check(gpt.transformer_blocks[b].attention.W_q.bias, q_b)
        gpt.transformer_blocks[b].attention.W_k.bias = assign_check(gpt.transformer_blocks[b].attention.W_k.bias, k_b)
        gpt.transformer_blocks[b].attention.W_v.bias = assign_check(gpt.transformer_blocks[b].attention.W_v.bias, v_b)


        gpt.transformer_blocks[b].attention.out.weight = assign_check(gpt.transformer_blocks[b].attention.out.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.transformer_blocks[b].attention.out.bias = assign_check(gpt.transformer_blocks[b].attention.out.bias, d[f"h.{b}.attn.c_proj.bias"])
        gpt.transformer_blocks[b].ff.layers[0].weight = assign_check(gpt.transformer_blocks[b].ff.layers[0].weight, d[f"h.{b}.mlp.c_fc.weight"].T)
        gpt.transformer_blocks[b].ff.layers[0].bias = assign_check(gpt.transformer_blocks[b].ff.layers[0].bias, d[f"h.{b}.mlp.c_fc.bias"])
        gpt.transformer_blocks[b].ff.layers[2].weight = assign_check(gpt.transformer_blocks[b].ff.layers[2].weight, d[f"h.{b}.mlp.c_proj.weight"].T)
        gpt.transformer_blocks[b].ff.layers[2].bias = assign_check(gpt.transformer_blocks[b].ff.layers[2].bias, d[f"h.{b}.mlp.c_proj.bias"])

        gpt.transformer_blocks[b].norm1.scale = assign_check(gpt.transformer_blocks[b].norm1.scale, d[f"h.{b}.ln_1.weight"])
        gpt.transformer_blocks[b].norm1.shift = assign_check(gpt.transformer_blocks[b].norm1.shift, d[f"h.{b}.ln_1.bias"])
        gpt.transformer_blocks[b].norm2.scale = assign_check(gpt.transformer_blocks[b].norm2.scale, d[f"h.{b}.ln_2.weight"])
        gpt.transformer_blocks[b].norm2.shift = assign_check(gpt.transformer_blocks[b].norm2.shift, d[f"h.{b}.ln_2.bias"])

        gpt.final_norm.scale = assign_check(gpt.final_norm.scale, d["ln_f.weight"])
        gpt.final_norm.shift = assign_check(gpt.final_norm.shift, d["ln_f.bias"])
        gpt.out_head.weight = assign_check(gpt.out_head.weight, d["wte.weight"])

def calc_accuracy(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            logits = model(input_batch)[:, -1, :] # get logits for the last token only
            predicted_labels = torch.argmax(logits, dim=-1) # get the predicted labels
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss(
                input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device,num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss

def train_classifier(model, train_loader, val_loader, optimizer, device,num_epochs, eval_iter, eval_iter_val):
    train_losses, val_losses, train_acc, val_acc = [], [], [], []
    examples_seen = 0
    step = -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss(input_batch, target_batch, model,device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            step += 1

            if step % eval_iter == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device,eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1} (Step {step:06d}): "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
        train_accuracy = calc_accuracy(train_loader, model, device, num_batches=eval_iter_val)
        val_accuracy = calc_accuracy(val_loader, model, device, num_batches=eval_iter_val)

        train_acc.append(train_accuracy)
        val_acc.append(val_accuracy)
        print(f"Train Accuracy: {train_accuracy*100:.4f}% | ",end="")
        print(f"Val Accuracy: {val_accuracy*100:.4f}%")

    return train_losses, val_losses, train_acc, val_acc, examples_seen
    
def plot_metrics(epoch, examples, train_val, val_vals, label="loss"):
    fig,ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(epoch, train_val, label=f"Training{label}")
    ax1.plot(epoch, val_vals, label=f"Validation{label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples, train_val, alpha=0)
    ax2.set_xlabel("Examples Seen")

    fig.tight_layout()
    plt.savefig(f"{label}_plot.png")
    plt.show()
