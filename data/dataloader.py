import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# CNN Pretraining Dataset 
class CNNPretrainDataset(Dataset):
    """Loads individual samples, noisy and clean"""
    def __init__(self, npy_file, noise_std: float = 0.1):
        self.noise_std = noise_std
        self.data = np.load(npy_file)  
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        clean_frame = self.data[idx]
        frame_tensor = torch.tensor(clean_frame, dtype=torch.float32).unsqueeze(0)
        noise = torch.randn_like(clean_frame) * self.noise_std
        noisy_frame = clean_frame + noise
        return  frame_tensor, noisy_frame

def get_cnn_pretrain_loader(npy_file, batch_size=64, shuffle=True):
    ds = CNNPretrainDataset(npy_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

class LSTMPretrainDataset(Dataset):
    """Loads a contiguous sequence of frames"""
    def __init__(self, npy_file, seq_len=20):
        self.data = np.load(npy_file)  
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, self.data.shape[0] - self.seq_len + 1)
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]  
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(1)  
        return seq_tensor

def get_lstm_pretrain_loader(npy_file, seq_len=20, batch_size=16, shuffle=True):
    ds = LSTMPretrainDataset(npy_file, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# Classifier Dataset (sequence + label)
class ClassifierDataset(Dataset):
    """Gets full sequences with labels"""
    def __init__(self, npy_file, label_file, seq_len=20):
        self.data = np.load(npy_file)  
        self.seq_len = seq_len
        self.labels = np.load(label_file)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq_start = idx  
        seq = self.data[seq_start:seq_start+self.seq_len]
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(1) 
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return seq_tensor, label

def get_classifier_loader(npy_file, label_file, seq_len=20, batch_size=16, shuffle=True):
    ds = ClassifierDataset(npy_file, label_file, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

