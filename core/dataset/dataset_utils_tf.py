import tensorflow as tf
import numpy as np
import random
import pickle
import pandas as pd
import os
from PIL import Image
from glob import glob
import cv2
import gc
import h5py
from sklearn.model_selection import train_test_split
import csv
from core.dataset.Motion_Sense_Splitter import DataFrameSplitter


#from sklearn.preprocessing import Normalizer

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class ManageDatasets():
	def __init__(self, cid):
		self.cid = cid
		#random.seed(self.cid)
		
	def load_Argoverse2(self, n_clients, non_iid=False):
		print("LOAD Argoverse2")
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/MNIST/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/MNIST/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)

			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
			x_train, x_test = x_train/255.0, x_test/255.0

			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]
			
			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
		else:
		    print("IID n_clients:", n_clients, " cid: ", self.cid)
		    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		    x_train, x_test                      = x_train/255.0, x_test/255.0
		    x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test
	
	def batch_generator(self, file_names, batch_size=200, masks=None):
		file_idx = -1
		while True:
			if file_idx > len(file_names):
				file_idx = 0
			file_idx = file_idx + 1
			batch_x = []
			batch_y = []
			batch_s = []
			for i in range(0, batch_size):
				data = h5py.File(file_names[file_idx], 'r')
				for mask in masks:
					if data['targets'][i][24] == mask:
						batch_x.append(data['rgb'][i])
						batch_y.append(data['targets'][i][:3])
						batch_s.append(data['targets'][i][10])
					if mask == 1:
						batch_x.append(data['rgb'][i])
						batch_s.append(data['targets'][i][10])
				data.close()
				gc.collect()
			if masks[0] == 1:
				yield ([np.array(batch_x)], [np.array(batch_s)])
			else:
				yield ([np.array(batch_x), np.array(batch_s)], [np.array(batch_s) if mask == 1 else np.array(batch_y) for mask in masks])
		
	def load_CORL2017(self, n_clients, non_iid=False):
		print("LOAD CORL2017")
		dataset_train = glob("data/CORL2017/SeqTrain/*")
		dataset_test = glob("data/CORL2017/SeqVal/*")
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/CORL2017/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				dataset_train = pickle.load(handle)

			with open(f'data/CORL2017/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				dataset_test = pickle.load(handle)
		else:
			print("IID n_clients:", n_clients, " cid: ", self.cid)
		
		return dataset_train, dataset_test

	def load_UCIHAR(self, n_clients, non_iid=False):
		print("LOAD UCI-HAR")		
		x_train = pd.read_csv('data/UCI-HAR/train/X_train.txt', delim_whitespace=True, header=None)
		x_train = x_train.values.tolist()
		x_train = np.array(x_train)
		
		y_train = pd.read_csv('data/UCI-HAR/train/y_train.txt', delim_whitespace=True, header=None)
		y_train = y_train.values.tolist()
		y_train = np.array(y_train)
		
		x_test = pd.read_csv('data/UCI-HAR/test/X_test.txt', delim_whitespace=True, header=None)
		x_test = x_test.values.tolist()
		x_test = np.array(x_test)
		
		y_test = pd.read_csv('data/UCI-HAR/test/y_test.txt', delim_whitespace=True, header=None)
		y_test = y_test.values.tolist()
		y_test = np.array(y_test)
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/UCI-HAR/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/UCI-HAR/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)
				
			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]

			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
		else:
			print("IID n_clients:", n_clients, " cid: ", self.cid)


		return x_train, y_train, x_test, y_test
		
	def create_MotionSenseDataset(self):
		print("CREATE Motion-Sense")
		
		folder_name = "data/Motion-Sense/A_DeviceMotion_data"
		mode="raw" #raw or mag
		labeled=True
		sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
		ds_list = pd.read_csv("data/Motion-Sense/data_subjects_info.csv")
		ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
		TRIAL_CODES = {
			ACT_LABELS[0]:[1,2,11],
			ACT_LABELS[1]:[3,4,12],
			ACT_LABELS[2]:[7,8,15],
			ACT_LABELS[3]:[9,16],
			ACT_LABELS[4]:[6,14],
			ACT_LABELS[5]:[5,13]
		}
		act_labels = ACT_LABELS [0:6]
		trial_codes = [TRIAL_CODES[act] for act in act_labels]
		
		dt_list = []
		for t in sdt:
			if t != "attitude":
				dt_list.append([t+".x",t+".y",t+".z"])
			else:
				dt_list.append([t+".roll", t+".pitch", t+".yaw"])				
		
		num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)
		if labeled:
			dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial]
		else:
			dataset = np.zeros((0,num_data_cols))
		
		for sub_id in ds_list["code"]:
			for act_id, act in enumerate(act_labels):
				for trial in trial_codes[act_id]:
					fname = folder_name+'/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
					raw_data = pd.read_csv(fname)
					raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
					vals = np.zeros((len(raw_data), num_data_cols))
					for x_id, axes in enumerate(dt_list):
						if mode == "mag":
							vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5
						else:
							vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
						vals = vals[:,:num_data_cols]
					if labeled:
						lbls = np.array([[act_id,
								sub_id-1,
								ds_list["weight"][sub_id-1],
								ds_list["height"][sub_id-1],
								ds_list["age"][sub_id-1],
								ds_list["gender"][sub_id-1],
								trial
								]]*len(raw_data), dtype=int)
						vals = np.concatenate((vals, lbls), axis=1)
					dataset = np.append(dataset,vals, axis=0)
		
		cols = []
		for axes in dt_list:
			if mode == "raw":
				cols += axes
			else:
				cols += [str(axes[0][:-2])]
		
		if labeled:
			cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
		dataset = pd.DataFrame(data=dataset, columns=cols)
		dfs = DataFrameSplitter(method="trials")
		train_data, test_data = dfs.train_test_split(dataset=dataset, labels = ("id","trial"), trial_col='trial', train_trials=[1.,2.,3.,4.,5.,6.,7.,8.,9.], verbose=2)
		
		Features = dataset.columns[:-7]
		labels_or_info = dataset.columns[-7:]
		
		x_train = train_data[Features]
		y_train = train_data["act"]

		x_test = test_data[Features]
		y_test = test_data["act"]	
		
		x_train.to_csv("data/Motion-Sense/x_train.csv", index=False)
		x_test.to_csv("data/Motion-Sense/x_test.csv", index=False)
		y_train.to_csv("data/Motion-Sense/y_train.csv", index=False)
		y_test.to_csv("data/Motion-Sense/y_test.csv", index=False)	        


	def load_MotionSense(self, n_clients, non_iid=False):
		print("LOAD Motion-Sense")
		
		x_train = pd.read_csv('data/Motion-Sense/x_train.csv', header=1)
		x_train = x_train.values.tolist()
		x_train = np.array(x_train)
		
		y_train = pd.read_csv('data/Motion-Sense/y_train.csv', header=1)
		y_train = y_train.values.tolist()
		y_train = np.array(y_train)
		
		x_test = pd.read_csv('data/Motion-Sense/x_test.csv', header=1)
		x_test = x_test.values.tolist()
		x_test = np.array(x_test)
		
		y_test = pd.read_csv('data/Motion-Sense/y_test.csv', header=1)
		y_test = y_test.values.tolist()
		y_test = np.array(y_test)
		
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/Motion-Sense/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/Motion-Sense/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)
				
			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]

			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
		else:
			print("IID n_clients:", n_clients, " cid: ", self.cid)
  
	    
		return x_train, y_train, x_test, y_test


	def load_MNIST(self, n_clients, non_iid=False):
		print("LOAD MNIST")
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/MNIST/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/MNIST/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)

			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
			x_train, x_test = x_train/255.0, x_test/255.0

			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]
			
			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
		else:
		    print("IID n_clients:", n_clients, " cid: ", self.cid)
		    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		    x_train, x_test                      = x_train/255.0, x_test/255.0
		    x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test
		
	def load_FMNIST(self, n_clients, non_iid=False):
		print("LOAD Fashion-MNIST")
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/Fashion-MNIST/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/Fashion-MNIST/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)

			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
			x_train, x_test = x_train/255.0, x_test/255.0

			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]
			
			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
		else:
			print("IID n_clients:", n_clients, " cid: ", self.cid)
			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
			x_train, x_test                      = x_train/255.0, x_test/255.0
			x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test
		

	def load_CIFAR10(self, n_clients, non_iid=False):
		print("LOAD CIFAR-10")
		if non_iid:
			print("Non-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/CIFAR10/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/CIFAR10/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)


			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
			x_train, x_test = x_train/255.0, x_test/255.0

			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]

			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
			
		else:
			print("IID n_clients:", n_clients, " cid: ", self.cid)
			(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
			x_train, x_test                      = x_train/255.0, x_test/255.0
			x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

		return x_train, y_train, x_test, y_test


	def load_CIFAR100(self, n_clients, non_iid=False):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
		x_train, x_test                      = x_train/255.0, x_test/255.0
		return x_train, y_train, x_test, y_test
	
	
	def load_SIGN(self, n_clients, non_iid=False):
		print("LOAD SIGN ", non_iid)
		# Assigning Path for Dataset
		data_dir = 'data/gtsrb-german-traffic-sign'
		train_path = 'data/gtsrb-german-traffic-sign/Train'
		test_path = 'data/gtsrb-german-traffic-sign/'
		IMG_HEIGHT = 30
		IMG_WIDTH = 30
		channels = 3
		
		# Finding Total Classes
		NUM_CATEGORIES = len(os.listdir(train_path))

		
		# Collecting the Training Data
		image_data = []
		image_labels = []
		for i in range(NUM_CATEGORIES):
			path = data_dir + '/Train/' + str(i)
			images = os.listdir(path)
			for img in images:
				try:
					image = cv2.imread(path + '/' + img)
					image_fromarray = Image.fromarray(image, 'RGB')
					resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
					image_data.append(np.array(resize_image))
					image_labels.append(i)
				except:
					print("Error in " + img)
		
		# Changing the list to numpy array
		image_data = np.array(image_data)
		image_labels = np.array(image_labels)
		
		# Shuffling the training data
		shuffle_indexes = np.arange(image_data.shape[0])
		np.random.shuffle(shuffle_indexes)
		image_data = image_data[shuffle_indexes]
		image_labels = image_labels[shuffle_indexes]
		
		# Splitting the data into train and validation set
		x_train, x_test, y_train, y_test = train_test_split(image_data, image_labels, test_size=0.3, random_state=42, shuffle=True)
		x_train = x_train/255 
		x_test = x_test/255
		
		#One hot encoding the labels
		y_train = tf.keras.utils.to_categorical(y_train, NUM_CATEGORIES)
		y_test = tf.keras.utils.to_categorical(y_test, NUM_CATEGORIES)
		
		if non_iid:
			print("NON-IID n_clients:", n_clients, " cid: ", self.cid)
			with open(f'data/SIGN/{n_clients}/idx_train_{self.cid}.pickle', 'rb') as handle:
				idx_train = pickle.load(handle)

			with open(f'data/SIGN/{n_clients}/idx_test_{self.cid}.pickle', 'rb') as handle:
				idx_test = pickle.load(handle)
				
			x_train = x_train[idx_train]
			x_test  = x_test[idx_test]

			y_train = y_train[idx_train]
			y_test  = y_test[idx_test]
		else:
			print("IID n_clients:", n_clients, " cid: ", self.cid)
		
		return x_train, y_train, x_test, y_test


	def slipt_dataset(self, x_train, y_train, x_test, y_test, n_clients):
		p_train = int(len(x_train)/n_clients)
		p_test  = int(len(x_test)/n_clients)

		selected_train = random.sample(range(len(x_train)), p_train)
		selected_test  = random.sample(range(len(x_test)), p_test)
		
		x_train  = x_train[selected_train]
		y_train  = y_train[selected_train]

		x_test   = x_test[selected_test]
		y_test   = y_test[selected_test]


		return x_train, y_train, x_test, y_test


	def select_dataset(self, dataset_name, n_clients, non_iid):
		if dataset_name == 'MNIST':
			return self.load_MNIST(n_clients, non_iid)

		elif dataset_name == 'CIFAR100':
			return self.load_CIFAR100(n_clients, non_iid)

		elif dataset_name == 'CIFAR10':
			return self.load_CIFAR10(n_clients, non_iid)

		elif dataset_name == 'Motion-Sense':
			return self.load_MotionSense(n_clients, non_iid)

		elif dataset_name == 'UCI-HAR':
			return self.load_UCIHAR(n_clients, non_iid)
			
		elif dataset_name == 'FMNIST':
			return self.load_FMNIST(n_clients, non_iid)
		
		elif dataset_name == 'SIGN':
			return self.load_SIGN(n_clients, non_iid)
			
		elif dataset_name == 'Argoverse2':
			return self.load_Argoverse2(n_clients, non_iid)

		elif dataset_name == 'CORL2017':
			return self.load_CORL2017(n_clients, non_iid)


	def normalize_data(self, x_train, x_test):
		x_train = Normalizer().fit_transform(np.array(x_train))
		x_test  = Normalizer().fit_transform(np.array(x_test))
		return x_train, x_test
	
