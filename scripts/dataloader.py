import h5py
import numpy as np
import os

#README als het goed is kun je de data in dezelfde map zetten als dit project,
#dan kune je de data processen door deze file te runnen. 
#zet de data dus in dezelfde map als de folder MEG_MODEL.
#De data komt in de map data te staan, deze map wordt niet op github gezet, want te groot enz jwz

def get_dataset_name(file_name_with_dir):
    filename_without_dir = file_name_with_dir.split('/')[-1]
    filename_without_dir = filename_without_dir.split('\\')[-1]  # Remove file extension
    temp = filename_without_dir.split('_')[:-1]
    dataset_name = '_'.join(temp)
    return dataset_name

#filename_path = "../Final Project data/Final Project data/Intra/train"

filename_path = "data/processed_rest_105923.h5"
with h5py.File(filename_path, 'r') as f:
    print(f"Keys in the file: {list(f.keys())}")
    dataset_name = get_dataset_name(filename_path)
    matrix = f.get(dataset_name)[()]
    print(type(matrix))
    print(matrix.shape)


def preprocess_and_save(path, downsample):
    for file in os.scandir(path):
        path = file.path
        print(f"Processing file: {path}")
        with h5py.File(path, 'r') as f:
            dataset_name = get_dataset_name(path)
            matrix = f.get(dataset_name)[()]
            matrix = matrix[:, ::downsample]

            new_filename = os.path.join(os.getcwd(), 'data/processed_' + dataset_name + '.h5')
            with h5py.File(new_filename, 'a') as new_f:
                suffix = 1
                base_name = dataset_name
                while f"{base_name}_{suffix}" in new_f:
                    suffix += 1
                new_f.create_dataset(f"{base_name}_{suffix}", data=matrix)


#preprocess_and_save(filename_path, 10)