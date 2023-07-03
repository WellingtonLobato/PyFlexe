import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D, Dense, InputLayer, BatchNormalization, Dropout, MaxPooling2D, concatenate
from tensorflow.keras import initializers
from keras.optimizers import Adam
from core.dataset.dataset_utils_tf import ManageDatasets

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(42)

#x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="FMNIST", n_clients=1, non_iid=False) #All Data
x_train, y_train, x_test, y_test = ManageDatasets(0).select_dataset(dataset_name="SIGN", n_clients=1, non_iid=False) #All Data
#num_classes = 10
num_classes = 43
"""
model = tf.keras.models.Sequential([
	tf.keras.layers.Flatten(input_shape=(x_train.shape[1:])),
	tf.keras.layers.Dense(128, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros()),
	tf.keras.layers.Dropout(0.2),
	tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros()),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
"""
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=x_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=64)
#model.save("./model_generic.h5")
model.save("./model_CNN.h5")

