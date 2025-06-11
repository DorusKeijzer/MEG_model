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

pathIntra = "../Final Project data/Intra"
pathCross = "../Final Project data/Cross"



def preprocess_and_save_intra(path, downsample):
    # This function processes the intra dataset and returns train and test sets
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
                    trainlabel.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test":
                    test.append(matrix)
                    testlabel.append(dataset_name.split("_")[:-1])


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


def preprocess_and_save_cross(path, downsample):
    # This function processes the cross dataset and returns train and test sets
    train = []
    test1 = []
    test2 = []
    test3 = []
    trainlabel = []
    test1label = []
    test2label = []
    test3label = []
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
                    trainlabel.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test1":
                    test1.append(matrix)
                    test1label.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test2":
                    test2.append(matrix)
                    test2label.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test3":
                    test3.append(matrix)
                    test3label.append(dataset_name.split("_")[:-1])


    train = np.array(train)
    test1 = np.array(test1)
    test2 = np.array(test2)
    test3 = np.array(test3)
    n_samples_train, h_train, w_train = train.shape
    n_samples_test1, h_test1, w_test1 = test1.shape
    n_samples_test2, h_test2, w_test2 = test2.shape
    n_samples_test3, h_test3, w_test3 = test3.shape

    train_flat = train.reshape(-1, w_train)
    test_flat1 = test1.reshape(-1, w_test1)
    test_flat2 = test2.reshape(-1, w_test2)
    test_flat3 = test3.reshape(-1, w_test3)
    scaler = MinMaxScaler()
    scaler.fit(train_flat)

    train = scaler.transform(train_flat).reshape(train.shape)
    test1 = scaler.transform(test_flat1).reshape(test1.shape)
    test2 = scaler.transform(test_flat2).reshape(test2.shape)
    test3 = scaler.transform(test_flat3).reshape(test3.shape)


    return train, trainlabel, test1, test1label, test2, test2label, test3, test3label
    


def preprocess_and_save_intra_PCA(path, downsample, PCA_components):
    # This function processes the intra dataset and returns train and test sets
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
                    trainlabel.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test":
                    test.append(matrix)
                    testlabel.append(dataset_name.split("_")[:-1])


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

    train_flat = train.reshape(h_train, -1)
    test_flat = test.reshape(h_test, -1)

    pca = PCA(PCA_components)
    pca.fit(train_flat.T)

    train = pca.transform(train_flat.T)
    test = pca.transform(test_flat.T)

    train = train.T.reshape(n_samples_train, PCA_components, w_train)
    test = test.T.reshape(n_samples_test, PCA_components, w_test)





    return train, trainlabel, test, testlabel


def preprocess_and_save_cross_PCA(path, downsample, PCA_components):
    # This function processes the cross dataset and returns train and test sets
    train = []
    test1 = []
    test2 = []
    test3 = []
    trainlabel = []
    test1label = []
    test2label = []
    test3label = []
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
                    trainlabel.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test1":
                    test1.append(matrix)
                    test1label.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test2":
                    test2.append(matrix)
                    test2label.append(dataset_name.split("_")[:-1])
                elif path.split('\\')[-2] == "test3":
                    test3.append(matrix)
                    test3label.append(dataset_name.split("_")[:-1])


    train = np.array(train)
    test1 = np.array(test1)
    test2 = np.array(test2)
    test3 = np.array(test3)
    n_samples_train, h_train, w_train = train.shape
    n_samples_test1, h_test1, w_test1 = test1.shape
    n_samples_test2, h_test2, w_test2 = test2.shape
    n_samples_test3, h_test3, w_test3 = test3.shape

    train_flat = train.reshape(-1, w_train)
    test_flat1 = test1.reshape(-1, w_test1)
    test_flat2 = test2.reshape(-1, w_test2)
    test_flat3 = test3.reshape(-1, w_test3)
    scaler = MinMaxScaler()
    scaler.fit(train_flat)

    train = scaler.transform(train_flat).reshape(train.shape)
    test1 = scaler.transform(test_flat1).reshape(test1.shape)
    test2 = scaler.transform(test_flat2).reshape(test2.shape)
    test3 = scaler.transform(test_flat3).reshape(test3.shape)

    train_flat = train.reshape(h_train, -1)
    test1_flat = test1.reshape(h_test1, -1)
    test2_flat = test2.reshape(h_test2, -1)
    test3_flat = test3.reshape(h_test3, -1)

    pca = PCA(PCA_components)
    pca.fit(train_flat.T)

    train = pca.transform(train_flat.T)
    test1 = pca.transform(test1_flat.T)
    test2 = pca.transform(test2_flat.T)
    test3 = pca.transform(test3_flat.T)

    train = train.T.reshape(n_samples_train, PCA_components, w_train)
    test1 = test1.T.reshape(n_samples_test1, PCA_components, w_test1)
    test2 = test2.T.reshape(n_samples_test2, PCA_components, w_test2)
    test3 = test3.T.reshape(n_samples_test3, PCA_components, w_test3)


    return train, trainlabel, test1, test1label, test2, test2label, test3, test3label
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

#train_X, train_Y, test_X, test_Y= preprocess_and_save_intra(pathIntra, 10)  
#test_X, test_Y = preprocess_and_save(test_path, 10, 50)
train_X, train_Y, test1_X, test1_Y, test2_X, test2_Y, test3_X, test3_Y = preprocess_and_save_cross_PCA(pathCross, 10, 50)
#train_X, train_Y, test_X, test_Y= preprocess_and_save_intra_PCA(pathIntra, 10, 50)  

print(f"Train shape: {train_X.shape}, Train labels: {len(train_Y)}")
print(f"Test1 shape: {test1_X.shape}, Test1 labels: {len(test1_Y)}")    
print(f"Test2 shape: {test2_X.shape}, Test2 labels: {len(test2_Y)}")
print(f"Test3 shape: {test3_X.shape}, Test3 labels: {len(test3_Y)}")