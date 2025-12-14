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
    
    #extracted_filepath = Path(extracted_filepath)
    #data_file_path = Path(data_file_path)
    with urllib.request.urlopen(url) as response:
        with open(zip_filename, 'wb') as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extracted_filepath)
    
    original_file_path = Path(extracted_filepath) / 'SMSSpamCollection'
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

        for enc in self.encodings:
            enc_length = len(enc)
            if enc_length > max_len:
                max_len = enc_length
        return max_len