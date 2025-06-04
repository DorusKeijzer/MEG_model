import h5py
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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

train_path = "../Final Project data/Intra"
test_path = "../Final Project data/Intra/test"




def preprocess_and_save_intra(path, downsample):
    train = []
    test = []
    trainlabel = []
    testlabel = []
    for dir in os.scandir(path):
        if not dir.is_dir():
            continue
        for file in os.scandir(dir):
            path = file.path
            print(f"Processing file: {path}")
            with h5py.File(path, 'r') as f:
                dataset_name = get_dataset_name(path)
                matrix = f.get(dataset_name)[()]
                matrix = matrix[:, ::downsample]
                if path.split('\\')[-2] == "train":
                    train.append(matrix)
                    trainlabel.append(dataset_name)
                elif path.split('\\')[-2] == "test":
                    test.append(matrix)
                    testlabel.append(dataset_name)


    train = np.array(train)
    test = np.array(test)
    n_samples_train, h_train, w_train = train.shape
    n_samples_test, h_test, w_test = test.shape

    train_flat = train.reshape(-1, w_train)
    test_flat = test.reshape(-1, w_test)
    scaler = MinMaxScaler()
    scaler.fit(train_flat)

    train = scaler.transform(train_flat).reshape(train.shape)
    test = scaler.transform(test_flat).reshape(test.shape)


    return train, trainlabel, test, testlabel
    
            # new_filename = os.path.join(os.getcwd(), 'data/processed_' + dataset_name + '.h5')
            # with h5py.File(new_filename, 'a') as new_f:
            #     suffix = 1
            #     base_name = dataset_name
            #     while f"{base_name}_{suffix}" in new_f:
            #         suffix += 1
            #     new_f.create_dataset(f"{base_name}_{suffix}", data=matrix)

# with h5py.File(filename_path, 'r') as f:
#     print(f"Keys in the file: {list(f.keys())}")
#     dataset_name = get_dataset_name(filename_path)
#     matrix = f.get(dataset_name)[()]
#     print(type(matrix))
#     print(matrix.shape)

train_X, train_Y, test_X, test_Y= preprocess_and_save_intra(train_path, 10)
#test_X, test_Y = preprocess_and_save(test_path, 10, 50)

print(f"Train X shape: {train_X.shape}")
print(test_Y)
print(train_Y)