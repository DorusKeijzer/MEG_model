import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from utils import noise_mask, available_device


class DenoisingCNNPretrainDataset(Dataset):
    """Loads individual samples, with gaussian nois and clean"""
    def __init__(self, npy_file, noise_std: float = 0.0001):
        self.noise_std = noise_std
        self.data = np.load(npy_file, mmap_mode='r')  # lazily loads the data
        self.length = self.data.shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        clean_frame = torch.tensor(self.data[idx], dtype=torch.float32)

        frame_tensor = torch.tensor(clean_frame, dtype=torch.float32).unsqueeze(0)
        noise = torch.randn_like(clean_frame) * self.noise_std
        noise = noise * noise_mask
        noisy_frame = (clean_frame + noise).unsqueeze(0)
        return  frame_tensor, noisy_frame

def get_denoising_cnn_pretrain_loader(batch_size=64, shuffle=True):
    file_path = "./data/processed_data/"
    datasets = []
    for task in ["Cross", "Intra"]:
        for subdir in os.listdir(os.path.join(file_path, task)):
            dirname = subdir
            subdir = os.path.join(file_path, task, dirname)
            if os.path.isdir(subdir) and "test" not in dirname:
                for file in os.listdir(subdir):
                    if os.path.splitext(file)[-1] == ".npy":
                        full_path = os.path.join(subdir, file)
                        datasets.append(DenoisingCNNPretrainDataset(full_path))
                        ds =DenoisingCNNPretrainDataset(full_path)
                        print(f"Length of combined dataset: {len(ds)}")
                        print(f"batches {len(ds)/64}")

                        return DataLoader(ds)
        return


    ds = torch.utils.data.ConcatDataset(datasets)
    print(f"Length of combined dataset: {len(ds)}")

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


class MaskingCNNPretrainedDataset(Dataset):
    """Loads individual samples, with random masking and clean"""
    def __init__(self, npy_file, noise_std: float = 0.1):
        self.noise_std = noise_std
        self.data = np.load(npy_file, allow_pickle=True)  
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        clean_frame = self.data[idx]
        frame_tensor = torch.tensor(clean_frame, dtype=torch.float32).unsqueeze(0)
        mask = torch.rand((20, 21), device=available_device) < 0.7
        noisy_frame = frame_tensor * mask * noise_mask
        return  frame_tensor, noisy_frame

def get_masked_cnn_pretrain_loader(npy_file, batch_size=64, shuffle=True):
    ds = MaskingCNNPretrainedDataset(npy_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)



class LSTMPretrainDataset(Dataset):
    """Loads a contiguous sequence of frames"""
    def __init__(self, npy_file, seq_len=20):
        self.data = np.load(npy_file, allow_pickle=True)  
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
        self.data = np.load(npy_file, allow_pickle=True)  
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    # compare noisy + clean frames

    file = "data/processed_data/Cross/test3/task_working_memory_735148_4.npy"
    noided = get_denoising_cnn_pretrain_loader()
    masked = get_masked_cnn_pretrain_loader(file)

    # Pick one loader for visualization
    for name, task in zip(["gaussian noise", "random masking"], [noided, masked]):
        clean, noisy = next(iter(task))  # use the same 'task' for both clean & noisy
        print(f"clean shape: {clean.shape}")
        print(f"noisy shape: {noisy.shape}")
        clean = clean[:8]  # take first 8 samples
        noisy = noisy[:8]


        fig, axs = plt.subplots(nrows=8, ncols=3, figsize=(6, 16))
        fig.suptitle(name, fontsize=16)

        for row in range(8):
            axs[row, 0].imshow(clean[row].squeeze(), cmap='viridis')
            axs[row, 0].set_title("Clean")
            axs[row, 0].axis('off')

            axs[row, 1].imshow(noisy[row].squeeze(), cmap='viridis')
            axs[row, 1].set_title("Noisy")
            axs[row, 1].axis('off')

            axs[row, 2].imshow((noisy[row]-clean[row]).squeeze(), cmap='viridis')
            axs[row, 2].set_title("noise (noisy-clean)")
            axs[row, 2].axis('off')


        plt.tight_layout()
        plt.show()


