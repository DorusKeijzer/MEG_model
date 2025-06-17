
from os.path import isdir
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import h5py
from config import DOWNSAMPLE_FACTOR


layout_matrix = np.zeros((20, 21), dtype=int)
positions = {
    (0, 10): 121, (1, 8): 122, (1, 9): 90, (1, 10): 89, (1, 11): 120, (1, 12): 152,
    (2, 7): 123, (2, 8): 91, (2, 9): 62, (2, 10): 61, (2, 11): 88, (2, 12): 119, (2, 13): 151,
    (3, 6): 124, (3, 7): 92, (3, 8): 63, (3, 9): 38, (3, 10): 37, (3, 11): 60, (3, 12): 87, (3, 13): 118, (3, 14): 150,
    (4, 4): 177, (4, 5): 153, (4, 6): 93, (4, 7): 64, (4, 8): 39, (4, 9): 20, (4, 10): 19, (4, 11): 36, (4, 12): 59,
    (4, 13): 86, (4, 14): 117, (4, 15): 149, (4, 16): 176, (4, 17): 195,
    (5, 0): 229, (5, 1): 212, (5, 2): 178, (5, 3): 154, (5, 4): 126, (5, 5): 94, (5, 6): 65, (5, 7): 40,
    (5, 8): 21, (5, 9): 6, (5, 10): 5, (5, 11): 18, (5, 12): 35, (5, 13): 58, (5, 14): 85, (5, 15): 116,
    (5, 16): 148, (5, 17): 175, (5, 18): 194, (5, 19): 228, (5, 20): 248,
    (6, 0): 230, (6, 1): 213, (6, 2): 179, (6, 3): 155, (6, 4): 127, (6, 5): 95, (6, 6): 66, (6, 7): 41,
    (6, 8): 22, (6, 9): 7, (6, 10): 4, (6, 11): 17, (6, 12): 34, (6, 13): 57, (6, 14): 84, (6, 15): 115,
    (6, 16): 147, (6, 17): 174, (6, 18): 193, (6, 19): 227, (6, 20): 247,
    (7, 1): 231, (7, 2): 196, (7, 3): 156, (7, 4): 128, (7, 5): 96, (7, 6): 67, (7, 7): 42, (7, 8): 23,
    (7, 9): 8, (7, 10): 3, (7, 11): 16, (7, 12): 33, (7, 13): 56, (7, 14): 83, (7, 15): 114, (7, 16): 146,
    (7, 17): 173, (7, 18): 211, (7, 19): 246,
    (8, 1): 232, (8, 2): 197, (8, 3): 157, (8, 4): 129, (8, 5): 97, (8, 6): 68, (8, 7): 43, (8, 8): 24,
    (8, 9): 9, (8, 10): 2, (8, 11): 15, (8, 12): 32, (8, 13): 55, (8, 14): 82, (8, 15): 113, (8, 16): 145,
    (8, 17): 172, (8, 18): 210, (8, 19): 245,
    (9, 1): 233, (9, 2): 198, (9, 3): 158, (9, 4): 130, (9, 5): 98, (9, 6): 69, (9, 7): 44, (9, 8): 25,
    (9, 9): 10, (9, 10): 1, (9, 11): 14, (9, 12): 31, (9, 13): 54, (9, 14): 81, (9, 15): 112, (9, 16): 144,
    (9, 17): 171, (9, 18): 209, (9, 19): 244,
    (10, 2): 214, (10, 3): 180, (10, 4): 131, (10, 5): 99, (10, 6): 70, (10, 7): 45, (10, 8): 26, (10, 9): 11,
    (10, 10): 12, (10, 11): 13, (10, 12): 30, (10, 13): 53, (10, 14): 80, (10, 15): 111, (10, 16): 143,
    (10, 17): 192, (10, 18): 226,
    (11, 4): 159, (11, 5): 132, (11, 6): 100, (11, 7): 71, (11, 8): 46, (11, 9): 27, (11, 10): 28, (11, 11): 29,
    (11, 12): 52, (11, 13): 79, (11, 14): 110, (11, 15): 142, (11, 16): 170,
    (12, 3): 181, (12, 4): 160, (12, 5): 133, (12, 6): 101, (12, 7): 72, (12, 8): 47, (12, 9): 48,
    (12, 10): 49, (12, 11): 50, (12, 12): 51, (12, 13): 78, (12, 14): 109, (12, 15): 141, (12, 16): 169,
    (12, 17): 191,
    (13, 2): 215, (13, 3): 199, (13, 4): 182, (13, 5): 161, (13, 6): 134, (13, 7): 102, (13, 8): 73,
    (13, 9): 74, (13, 10): 75, (13, 11): 76, (13, 12): 77, (13, 13): 108, (13, 14): 140, (13, 15): 168,
    (13, 16): 190, (13, 17): 208, (13, 18): 225,
    (14, 2): 234, (14, 3): 216, (14, 4): 200, (14, 5): 183, (14, 6): 162, (14, 7): 135, (14, 8): 103,
    (14, 9): 104, (14, 10): 105, (14, 11): 106, (14, 12): 107, (14, 13): 139, (14, 14): 167, (14, 15): 189,
    (14, 16): 207, (14, 17): 224, (14, 18): 243,
    (15, 4): 235, (15, 5): 217, (15, 6): 201, (15, 7): 184, (15, 8): 163, (15, 9): 136, (15, 10): 137,
    (15, 11): 138, (15, 12): 166, (15, 13): 188, (15, 14): 206, (15, 15): 223, (15, 16): 242,
    (16, 6): 236, (16, 7): 218, (16, 8): 202, (16, 9): 185, (16, 10): 164, (16, 11): 165, (16, 12): 187,
    (16, 13): 205, (16, 14): 222, (16, 15): 241,
    (17, 8): 219, (17, 9): 203, (17, 10): 186, (17, 11): 204, (17, 12): 221,
    (18, 9): 237, (18, 10): 220, (18, 11): 240,
    (19, 10): 238, (19, 11): 239,
}
for (r, c), val in positions.items():
    layout_matrix[r, c] = val
# Convert to zero-based indices for sensor mapping (subtract 1)
layout_matrix = layout_matrix - 1


def compute_mean_std(file_path, downsample_factor):
    """computes the mean and std over the full dataset without crashing your computer"""
    count = 0
    mean = None
    M2 = None

    for task in ["Cross", "Intra"]:
        for subdir in os.listdir(os.path.join(file_path, task)):
            dirname = subdir
            subdir_path = os.path.join(file_path, task, dirname)
            if os.path.isdir(subdir_path) and "test" not in dirname:
                for file in os.listdir(subdir_path):
                    if file.endswith(".h5"):
                        h5_filepath = os.path.join(subdir_path, file)
                        with h5py.File(h5_filepath, 'r') as f:
                            dataset_name = list(f.keys())[0]
                            data = f[dataset_name][()]  # (248, time)
                            data = data.T[::downsample_factor]  # (time_downsampled, 248)

                        for x in data:
                            count += 1
                            x = x.astype(np.float64)
                            if mean is None:
                                mean = np.zeros_like(x)
                                M2 = np.zeros_like(x)
                            delta = x - mean
                            mean += delta / count
                            delta2 = x - mean
                            M2 += delta * delta2

    variance = M2 / (count - 1)
    std = np.sqrt(variance)
    return mean, std



def get_dataset_name(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]


def process_meg_data(h5_filepath, downsample_factor, dirname, mean, std, save_dir="./data/processed_data"):
    """
    Opens .h5 file, downsamples, applies time-wise Z-score normalization, spatial transform,
    then saves a npy object to specified save_dir.
    """
    with h5py.File(h5_filepath, 'r') as f:
        dataset_name = list(f.keys())[0]
        print(f"Loading dataset: {dataset_name}")
        data = f[dataset_name][()]  # shape (248, time_steps)

    # Transpose to shape (time_steps, 248)
    data = data.T

    # Downsample temporally
    data_ds = data[::downsample_factor]

    # Make sure mean and std shape matches sensors axis (248)
    if mean.ndim == 2 and mean.shape[1] == 1:
        mean = mean.squeeze()
    if std.ndim == 2 and std.shape[1] == 1:
        std = std.squeeze()

    # Time-wise Z-score normalization
    data_ds_norm = (data_ds - mean) / std  # shape (time_steps_downsampled, 248)

    # Initialize spatial reshaped data array
    num_time_steps = data_ds_norm.shape[0]
    spatial_data = np.zeros((num_time_steps, 20, 21), dtype=data_ds_norm.dtype)

    # Map channels to spatial grid using layout_matrix
    for r in range(20):
        for c in range(21):
            ch_idx = layout_matrix[r, c]
            if ch_idx >= 0:
                spatial_data[:, r, c] = data_ds_norm[:, ch_idx]

    # Save processed data
    new_file_dir = os.path.join(save_dir, dirname)
    os.makedirs(new_file_dir, exist_ok=True)
    base_name = get_dataset_name(h5_filepath)

    save_path = os.path.join(new_file_dir, f"{base_name}.npy")
    np.save(save_path, spatial_data)

    print(f"Processed data saved to {save_path}")
    return save_path

if __name__ == "__main__":
    file_path = "./data/files/"


    mean, std = compute_mean_std(file_path,DOWNSAMPLE_FACTOR )

    for task in ["Cross", "Intra"]:
        for subdir in os.listdir(os.path.join(file_path, task)):
            dirname = subdir
            subdir = os.path.join(file_path, task, dirname)
            if isdir(subdir):
                for file in os.listdir(subdir):
                    if os.path.splitext(file)[-1] == ".h5":
                        full_path = os.path.join(subdir, file)
                        process_meg_data(full_path, DOWNSAMPLE_FACTOR, os.path.join(task, dirname), mean, std)
