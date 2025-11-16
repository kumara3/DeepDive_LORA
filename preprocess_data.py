import os
from pathlib import Path
from pyexpat import model
import urllib
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model


def download_data(url,zip_filename,extracted_filepath,data_file_path):
    if data_file_path.exists():
        print(f"Data file already exists at {data_file_path}")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_filename, 'wb') as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extracted_filepath)
    
    original_file_path = extracted_filepath / 'SMSSpamCollection'
    os.rename(original_file_path, data_file_path)
    print(f"Data downloaded and extracted to {data_file_path}")

def create_balanced_dataset(df):
    # count the number of instances in each class
    num_spam = df[df['Label'] == "spam"].shape[0]

    # downsample the majority class (ham) to match the minority class (spam)
    subset_ham = df[df['Label'] == "ham"].sample(n=num_spam, random_state=42)

    # combine the balanced datasets
    balanced_df = pd.concat([subset_ham,df[df["Label"] == "spam"]])
    
    return balanced_df

def randomly_split_dataset(df, train_frac, val_frac):
    # shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # calculate the number of instances for each split
    total_len = len(df)
    train_end = int(train_frac * total_len)
    val_end = train_end + int(val_frac * total_len)
    
    # split the dataframe
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df

class makespamDataset(Dataset):
    
    def __init__(self, csv_file,tokenizer,max_length=None,pad_token=50256):
        self.dataframe = pd.read_csv(csv_file)
        
        self.encodings = [tokenizer.encode(el) for el in self.dataframe["Text"]]
        
        if max_length is None:
            self.max_length = self._longest_sequence_length()
        else:
            self.max_length = max_length

            self.encodings = [enc[:self.max_length] for enc in self.encodings]
        self.encodings = [enc+[pad_token] *(self.max_length-len(enc)) for enc in self.encodings]

    def __getitem__(self, idx):

        encoded_text = self.encodings[idx]
        label = self.dataframe.iloc[idx]["Label"]

        return(torch.tensor(encoded_text, dtype=torch.long),torch.tensor(label,dtype=torch.long))
    
    def __len__(self):
        return len(self.dataframe)
    
    def _longest_sequence_length(self):
        max_len = 0
        max_len = max(len(enc) for enc in self.encodings if len(enc) > max_len)
        return max_len

def assign_check(target, source):
    if target.shape != source.T.shape:
        raise ValueError(f"Shape mismatch: target {target.shape}, source {source.shape}")
    return torch.nn.Parameter(source.clone().detach())

def load_weights(gpt, gpt_hf, BASE_CONFIG):

    d = gpt_hf.state_dict()

    gpt.position_embedding.weight = assign_check(gpt.position_embedding.weight, d["wpe.weight"])
    gpt.token_embedding.weight = assign_check(gpt.token_embedding.weight, d["wte.weight"])
    
    for b in range(BASE_CONFIG["n_layers"]):
        q_w, k_w, v_w = np.split(d[f"h.{b}.attn.c_attn.weight"], 3, axis=-1)
        gpt.transformer_blocks[b].att.W_q.weight = assign_check(gpt.transformer_blocks[b].att.W_q.weight, q_w.T)
        gpt.transformer_blocks[b].att.W_k.weight = assign_check(gpt.transformer_blocks[b].att.W_k.weight, k_w.T)
        gpt.transformer_blocks[b].att.W_v.weight = assign_check(gpt.transformer_blocks[b].att.W_v.weight, v_w.T)

        q_b, k_b, v_b = np.split(d[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
        gpt.transformer_blocks[b].att.W_q.bias = assign_check(gpt.transformer_blocks[b].att.W_q.bias, q_b)
        gpt.transformer_blocks[b].att.W_k.bias = assign_check(gpt.transformer_blocks[b].att.W_k.bias, k_b)
        gpt.transformer_blocks[b].att.W_v.bias = assign_check(gpt.transformer_blocks[b].att.W_v.bias, v_b)


        gpt.transformer_blocks[b].att.out.weight = assign_check(gpt.transformer_blocks[b].att.out.weight, d[f"h.{b}.attn.c_proj.weight"].T)
        gpt.transformer_blocks[b].att.out.bias = assign_check(gpt.transformer_blocks[b].att.out.bias, d[f"h.{b}.attn.c_proj.bias"])

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



