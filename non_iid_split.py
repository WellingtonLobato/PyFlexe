import tensorflow as tf
import numpy as np
import pandas as pd
import random
import pickle
import os
from core.dataset.dataset_utils_tf import ManageDatasets

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#NCLIENTS = [60, 150, 250] #Total Number of Vehicles
NCLIENTS = [10, 20, 50, 60, 100, 150, 200, 250] #Total Number of Vehicles
TELEPORT_FACTOR = 0.1 #Percent of Teleport Vehicles
#DATASET = "Fashion-MNIST"
#DATASET = "SIGN"
#DATASET = "MNIST"
#DATASET = "CIFAR10"
#DATASET = "UCI-HAR"
#DATASET = "Motion-Sense"
#DATASET = "Argoverse2"
DATASET = "CORL2017"

if DATASET == "Fashion-MNIST":
    x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="FMNIST", n_clients=1, non_iid=False) #All Data
    NUM_CLASSES = 10
elif DATASET == "CIFAR10":
    x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="CIFAR10", n_clients=1, non_iid=False) #All Data
    NUM_CLASSES = 10
elif DATASET == "SIGN":
    x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="SIGN", n_clients=1, non_iid=False) #All Data
    y_train = np.where(y_train==1)[1]
    y_test = np.where(y_test==1)[1]
    NUM_CLASSES = 43
elif DATASET == "UCI-HAR":
    x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="UCI-HAR", n_clients=1, non_iid=False) #All Data
    NUM_CLASSES = 6 #[WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING]
elif DATASET == "Motion-Sense":
    x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="Motion-Sense", n_clients=1, non_iid=False) #All Data
    NUM_CLASSES = 6 #[dws: downstairs, ups: upstairs, sit: sitting, std: standing, wlk: walking, jog: jogging]
elif DATASET == "Argoverse2":
    x_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="Motion-Sense", n_clients=1, non_iid=False) #All Data
    NUM_CLASSES = 6 #[dws: downstairs, ups: upstairs, sit: sitting, std: standing, wlk: walking, jog: jogging]
elif DATASET == "CORL2017":
    dataset_train, dataset_test = ManageDatasets(0).select_dataset(dataset_name="CORL2017", n_clients=1, non_iid=False) #All Data
else:
    x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="MNIST", n_clients=1, non_iid=False) #All Data
    NUM_CLASSES = 10
    


for nclients in NCLIENTS:
    nclients_range = int((nclients*TELEPORT_FACTOR)+nclients)
    for _ in range(nclients_range):
        if DATASET == "CORL2017":
            random.shuffle(dataset_train)
            random.shuffle(dataset_test)
            num_files_train = random.randint(2, len(dataset_train)/2.0)
            num_files_test = random.randint(2, len(dataset_test)/2.0)
            files_train = random.sample(dataset_train, num_files_train)
            files_test = random.sample(dataset_test, num_files_test)
            
            filename_train = f"./data/{DATASET}/{nclients}/idx_train_{_}.pickle"
            filename_test  = f"./data/{DATASET}/{nclients}/idx_test_{_}.pickle"
            os.makedirs(os.path.dirname(filename_train), exist_ok=True)
            os.makedirs(os.path.dirname(filename_test), exist_ok=True)
            		
            with open(filename_train, 'wb') as handle:
                pickle.dump(files_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(filename_test, 'wb') as handle:
                pickle.dump(files_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        else:
            index_train   = []
            index_test    = []
            classes_2b_selected = random.randint(2, 6)
            for selected_class in range(classes_2b_selected):
                label = random.randint(0, NUM_CLASSES-1)
                index_labels_train = np.where(y_train == label)
                index_labels_test = np.where(y_test == label)
                sample_size_train = random.randint(1, 40) / 100.0
                sample_size_test = random.randint(1, 30) / 100.0
                
                index_sampled_train = random.choices(index_labels_train[0], k=int(sample_size_train * len(index_labels_train[0])))
                index_sampled_test = random.choices(index_labels_test[0], k=int(sample_size_test * len(index_labels_test[0])))
                
                index_train += [item for item in index_sampled_train]
                index_test += [item for item in index_sampled_test]

                filename_train = f"./data/{DATASET}/{nclients}/idx_train_{_}.pickle"
                filename_test  = f"./data/{DATASET}/{nclients}/idx_test_{_}.pickle"
                
                os.makedirs(os.path.dirname(filename_train), exist_ok=True)
                os.makedirs(os.path.dirname(filename_test), exist_ok=True)
                
                with open(filename_train, 'wb') as handle:
                    pickle.dump(index_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(filename_test, 'wb') as handle:
                    pickle.dump(index_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
""""""
