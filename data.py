'''
Create dataloaders
'''
import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import pandas as pd
from numpy import genfromtxt

PATH = {'twitter': 'datasets/twitter.csv',  # 'hate_speech' col: # of users who marked it as hateful
        'gab': 'datasets/gab.csv',
        'reddit': 'datasets/reddit.csv',
        'parler': 'datasets/parler_pretrain.ndjson'
        }

class TwitterDataset(Dataset):
    def __init__(self, data_csv, sample=False):
        return
        
    def __getitem__(self, index):
        fp, _, idx = self.data[index]
        idx = int(idx)
        return (img, idx)

    def __len__(self):
        return len(self.data)

def get_dataset(csv_path):
    return TwitterDataset(csv_path)

def create_dataloaders(train_set, val_set, test_set):
    train_dataloader = DataLoader(train_set, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=2, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size=2, shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

def get_dataloaders(train_csv):
    data = get_dataset(train_csv)
    train_set, val_set, test_set = train_val_test_split(data)
    dataloaders = create_dataloaders(train_set, val_set, test_set)
    return dataloaders

def train_val_test_split(dataset): 
    train_size = int(len(dataset) * 0.7) # 70 15 15
    val_size = int(len(dataset) * 0.15)
    test_size = int(len(dataset) * 0.15)
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], 
                                            generator=torch.Generator().manual_seed(42))
    return train_subset, val_subset, test_subset
