import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from utils import noise_mask, available_device


class DenoisingCNNPretrainDataset(Dataset):
    """Loads individual samples, with gaussian noise and clean"""
    def __init__(self, npy_file, noise_std: float = 0.2, noise_mask=None):
        self.noise_std = noise_std
        self.data = np.load(npy_file, mmap_mode='r')  # lazy loading
        self.length = self.data.shape[0]
        
        if noise_mask is not None:
            self.noise_mask = torch.tensor(noise_mask, dtype=torch.float32)
        else:
            self.noise_mask = None
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        clean_frame = torch.from_numpy(self.data[idx].copy()).float()  # keep on CPU

        # Add batch and channel dims (1,1,H,W)
        frame_tensor = clean_frame.unsqueeze(0).unsqueeze(0)

        noise = torch.randn_like(clean_frame) * self.noise_std
        
        if self.noise_mask is not None:
            noise = noise * self.noise_mask

        noisy_frame = (clean_frame + noise).unsqueeze(0).unsqueeze(0)
        
        return frame_tensor, noisy_frame

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
                        datasets.append(DenoisingCNNPretrainDataset(full_path, noise_mask=noise_mask))
                        ds =DenoisingCNNPretrainDataset(full_path)
                        # print(f"Length of combined dataset: {len(ds)}")
                        # print(f"batches {len(ds)/64}")

                        # return DataLoader(ds)
        # return


    ds = torch.utils.data.ConcatDataset(datasets)
    print(f"Length of combined dataset: {len(ds)}")

    print(f"batches {len(ds)/64}")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)


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
        mask = torch.rand((20, 21), device=available_device) < 0.8
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

        clean, noisy = next(iter(task))  # get batch
        print(clean.shape)
        print(noisy.shape)
        batch_size = clean.shape[0]
        num_samples = min(8, batch_size)  # use smaller of 8 or actual batch size

        clean = clean[:num_samples]
        noisy = noisy[:num_samples]

        fig, axs = plt.subplots(nrows=num_samples, ncols=3, figsize=(6, 2 * num_samples))
        axs = axs.reshape(num_samples, 3)  # force axs to 2D array

        fig.suptitle(name, fontsize=16)

        for row in range(num_samples):
            clean_img = clean[row].squeeze().cpu().numpy()
            noisy_img = noisy[row].squeeze().cpu().numpy()
            noise_img = (noisy[row] - clean[row]).squeeze().cpu().numpy()

            axs[row, 0].imshow(clean_img, cmap='viridis')
            axs[row, 0].set_title("Clean")
            axs[row, 0].axis('off')

            axs[row, 1].imshow(noisy_img, cmap='viridis')
            axs[row, 1].set_title("Noisy")
            axs[row, 1].axis('off')

            axs[row, 2].imshow(noise_img, cmap='viridis')
            axs[row, 2].set_title("noise (noisy-clean)")
            axs[row, 2].axis('off')

        plt.tight_layout()
        plt.show()
