import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from random import random, randint
from utils import available_device, noise_mask


class DenoisingCNNPretrainDataset(Dataset):
    """Loads individual samples, with gaussian noise and clean"""
    def __init__(self, npy_file, max_noise_std: float = 0.3, noise_mask=None):
        self.max_noise_std = max_noise_std
        self.data = np.load(npy_file, mmap_mode='r')
        self.length = self.data.shape[0]

        if noise_mask is not None:
            self.noise_mask = torch.tensor(noise_mask, dtype=torch.float32)
        else:
            self.noise_mask = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean_frame = torch.from_numpy(self.data[idx].copy()).float()
        frame_tensor = clean_frame.unsqueeze(0).unsqueeze(0)

        noise = torch.randn_like(clean_frame) * random() * self.max_noise_std
        if self.noise_mask is not None:
            noise = noise * self.noise_mask

        noisy_frame = (clean_frame + noise).unsqueeze(0).unsqueeze(0)
        return frame_tensor, noisy_frame


def get_denoising_cnn_pretrain_dataset(batch_size=64, shuffle=True, noise_mask=None):
    file_path = "./data/processed_data/"
    datasets = []
    for task in ["Cross", "Intra"]:
        for subdir in os.listdir(os.path.join(file_path, task)):
            full_subdir = os.path.join(file_path, task, subdir)
            if os.path.isdir(full_subdir) and "test" not in subdir:
                for file in os.listdir(full_subdir):
                    if file.endswith(".npy"):
                        full_path = os.path.join(full_subdir, file)
                        datasets.append(DenoisingCNNPretrainDataset(full_path, noise_mask=noise_mask))

    ds = ConcatDataset(datasets)
    print(f"Length of combined dataset: {len(ds)}")
    print(f"Batches (batch size {batch_size}): {len(ds)/batch_size}")
    return ds


class MaskingCNNPretrainDataset(Dataset):
    """Loads individual samples, with random masking and clean"""
    def __init__(self, npy_file, noise_mask=None):
        self.data = np.load(npy_file, mmap_mode='r')
        self.noise_mask = torch.tensor(noise_mask, dtype=torch.float32) if noise_mask is not None else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        clean_frame = torch.tensor(self.data[idx], dtype=torch.float32)
        mask = (torch.rand((20, 21)) < 0.8).float()
        if self.noise_mask is not None:
            mask = mask * self.noise_mask
        masked_frame = clean_frame * mask
        return clean_frame.unsqueeze(0), masked_frame.unsqueeze(0)


def get_masked_cnn_pretrain_dataset(batch_size=64, shuffle=True, noise_mask=None):
    file_path = "./data/processed_data/"
    datasets = []
    for task in ["Cross", "Intra"]:
        for subdir in os.listdir(os.path.join(file_path, task)):
            full_subdir = os.path.join(file_path, task, subdir)
            if os.path.isdir(full_subdir) and "test" not in subdir:
                for file in os.listdir(full_subdir):
                    if file.endswith(".npy"):
                        full_path = os.path.join(full_subdir, file)
                        datasets.append(MaskingCNNPretrainDataset(full_path, noise_mask=noise_mask))

    ds = ConcatDataset(datasets)
    print(f"Length of combined masked dataset: {len(ds)}")
    return ds




class MaskedNoisyCNNPretrainDataset(Dataset):
    """Combines both masking and noise on the same input frame"""
    def __init__(self, npy_file, max_noise_std: float = 0.3, noise_mask=None):
        self.data = np.load(npy_file, mmap_mode='r')
        self.max_noise_std = max_noise_std
        self.noise_mask = torch.tensor(noise_mask, dtype=torch.float32) if noise_mask is not None else None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        clean_frame = torch.tensor(self.data[idx], dtype=torch.float32)

        noise = torch.randn_like(clean_frame) * random() * self.max_noise_std
        if self.noise_mask is not None:
            noise = noise * self.noise_mask

        masked = (torch.rand_like(clean_frame) < 0.8).float()
        if self.noise_mask is not None:
            masked = masked * self.noise_mask

        noisy_masked = (clean_frame + noise) * masked

        return clean_frame.unsqueeze(0), noisy_masked.unsqueeze(0)


def get_masked_noisy_cnn_pretrain_dataset(batch_size=64, shuffle=True, noise_mask=None):
    file_path = "./data/processed_data/"
    datasets = []
    for task in ["Cross", "Intra"]:
        for subdir in os.listdir(os.path.join(file_path, task)):
            full_subdir = os.path.join(file_path, task, subdir)
            if os.path.isdir(full_subdir) and "test" not in subdir:
                for file in os.listdir(full_subdir):
                    if file.endswith(".npy"):
                        full_path = os.path.join(full_subdir, file)
                        datasets.append(MaskedNoisyCNNPretrainDataset(full_path, noise_mask=noise_mask))

    ds = ConcatDataset(datasets)
    print(f"Length of combined masked+noisy dataset: {len(ds)}")
    return ds


class MEGVolumeDataset(Dataset):
    TASK_LABELS = {
        'rest': 0,
        'task_motor': 1,
        'task_story_math': 2,
        'task_working_memory': 3
    }
    
    def __init__(self, root_dir, mode='train'):
        self.samples = []
        
        for task_group in ['Intra', 'Cross']:
            task_group_path = os.path.join(root_dir, task_group)
            
            subfolders = ['train'] if mode == 'train' else ['test'] if task_group == 'Intra' else ['test1', 'test2', 'test3']
            
            for subfolder in subfolders:
                print(subfolder)


                folder_path = os.path.join(task_group_path, subfolder)
                if not os.path.exists(folder_path):
                    continue
                
                for file in os.listdir(folder_path):

                    if file.endswith('.npy'):
                        # Parsing label logic
                        filename = file.lower()
                        
                        if filename.startswith('rest'):
                            label = self.TASK_LABELS['rest']
                        else:
                            # Non-resting: filenames start with "task_"
                            # Extract full task name prefix before subjectID/number parts
                            parts = filename.split('_')
                            # gather first two parts: task and the actual task type words, eg ['task', 'motor', ...]
                            # task label could be multiple words joined by underscore: e.g. task_story_math
                            # so try to reconstruct task label by joining parts until numeric hit
                            
                            first_num_idx = None
                            for i, p in enumerate(parts):
                                if p.isdigit():
                                    first_num_idx = i
                                    break
                            if first_num_idx is None:
                                # no digit found, skip this file just in case
                                continue
                            
                            task_name = '_'.join(parts[:first_num_idx])
                            label = self.TASK_LABELS.get(task_name, None)
                            if label is None:
                                # skip unknown labels
                                continue
                        
                        full_path = os.path.join(folder_path, file)
                        self.samples.append((full_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        volume = np.load(path)  # shape: (20, 21, T)
        
        volume = torch.tensor(volume, dtype=torch.float32)  # (20, 21, T)
        volume = volume.permute(2, 0, 1).unsqueeze(1)       # (T, 1, 20, 21)
        
        return volume, label


class MaskedMEGSequenceDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        min_seq_len: int = 100,
        max_seq_len: int = 3000,
        mask_ratio: float = 0.3,
        max_mask_blocks: int = 5,
    ):
        self.files = files
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.mask_ratio = mask_ratio
        self.max_mask_blocks = max_mask_blocks
        self.chunk_index = []
        self._index_volumes()

    def _index_volumes(self):
        for file_path in self.files:
            data = np.load(file_path, mmap_mode='r')
            volume_length = data.shape[0]

            # Only index volumes large enough
            if volume_length >= self.min_seq_len:
                self.chunk_index.append((file_path, volume_length))

    def __len__(self):
        return len(self.chunk_index)

    def __getitem__(self, idx):
        file_path, volume_length = self.chunk_index[idx]
        data = np.load(file_path, mmap_mode='r')

        # Randomly sample a sequence length
        seq_len = torch.randint(self.min_seq_len, min(volume_length, self.max_seq_len) + 1, (1,)).item()

        # Random start position
        start_idx = torch.randint(0, volume_length - seq_len + 1, (1,)).item()
        data = data[start_idx:start_idx + seq_len]  # shape: (T, 20, 21)

        data = torch.tensor(data, dtype=torch.float32).unsqueeze(1)  # (T, 1, 20, 21)
        mask = torch.zeros(seq_len, dtype=torch.bool)

        total_to_mask = int(seq_len * self.mask_ratio)

        g = torch.Generator()
        g.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())

        remaining = total_to_mask
        blocks = 0
        while remaining > 0 and blocks < self.max_mask_blocks:
            max_block_size = remaining
            block_size = torch.randint(1, max_block_size + 1, (1,), generator=g).item()
            start = torch.randint(0, seq_len - block_size + 1, (1,), generator=g).item()

            # Ensure block doesn't overlap
            if not mask[start:start + block_size].any():
                mask[start:start + block_size] = True
                remaining -= block_size
                blocks += 1

        masked_data = data.clone()
        masked_data[mask] = 0.0  # could use noise or learned tokens too

        return {
            'input': masked_data,   # shape: (T, 1, 20, 21)
            'mask': mask,           # shape: (T,)
            'target': data,         # original full data
            'seq_len': seq_len,     # useful for batching if variable-length
        }

if __name__ == "__main__":
    # sanity checks and visualizing
    from matplotlib import pyplot as plt

    
    dataset = MEGVolumeDataset("./data/processed_data/", mode='train')
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    for volumes, labels in loader:
        print(volumes.shape)  # (B, T, 1, 20, 21)
        print(labels)         # (B,)
        break

    masked_temporal_dataset = MaskedMEGSequenceDataset(files = ["./data/processed_data/Intra/train/task_working_memory_105923_5.npy"], min_seq_len=100, max_seq_len=300, mask_ratio=0.3, max_mask_blocks=2)
    loader = torch.utils.data.DataLoader(masked_temporal_dataset, batch_size=40, shuffle=True)

    for volumes in loader:
        print(volumes["input"].shape)  # (B, T, 1, 20, 21)
        print(volumes["mask"].shape)  # (B, T,)
        print(volumes["target"].shape)  # (B, T, 1, 20, 21)
        break


        # Get a single sequence
    for volumes in loader:
        inputs = volumes["input"][0]     # (T, 1, 20, 21)
        targets = volumes["target"][0]   # (T, 1, 20, 21)
        mask = volumes["mask"][0]
        print(mask)
        break

    T = min(10, inputs.shape[0])

    fig, axes = plt.subplots(2, T, figsize=(T * 2, 4))  # 2 rows: input + target

    for t in range(T):
        # Top row: input
        axes[0, t].imshow(inputs[t, 0], cmap='viridis')
        axes[0, t].set_title(f"Input T{t}")
        axes[0, t].axis('off')
        
        # Bottom row: target
        axes[1, t].imshow(targets[t, 0], cmap='viridis')
        axes[1, t].set_title(f"Target T{t}")
        axes[1, t].axis('off')

    plt.tight_layout()
    plt.show()



    # compare noisy + clean frames

    file = "data/processed_data/Cross/test3/task_working_memory_735148_4.npy"
    noidedset = get_denoising_cnn_pretrain_dataset(noise_mask=noise_mask)
    maskedset = get_masked_cnn_pretrain_dataset(noise_mask=noise_mask)
    bothset = get_masked_noisy_cnn_pretrain_dataset(noise_mask=noise_mask)
    noided = DataLoader(noidedset)
    masked = DataLoader(maskedset)
    bothed = DataLoader(bothset)
    

    # Pick one loader for visualization
    for name, task in zip(["gaussian noise", "random masking", "both masking and noise"], [noided, masked, bothed]):

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
