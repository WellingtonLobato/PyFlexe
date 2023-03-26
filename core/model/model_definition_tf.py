import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D, Dense, InputLayer, BatchNormalization, Dropout

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(0)

class ModelCreation():
	"""
    create_DNN 

    :param input_shape: Quantidade de amostras para treino
    :param num_classes: Quantidade de amostras para teste
    """
	def create_DNN(self, input_shape, num_classes):
		input = Input(shape=(input_shape[1:]))
		x = Flatten()(input)
		x = Dense(512, activation='relu')(x)
		x = Dense(256, activation='relu')(x)
		x = Dense(32,  activation='relu')(x)
		out = Dense(num_classes, activation='softmax')(x)
		model = Model(inputs=input, outputs=[out])
		model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return model

	"""
    create_generic_model 

    :param input_shape: Quantidade de amostras para treino
    :param num_classes: Quantidade de amostras para teste
    """
	def create_generic_model(self, input_shape, num_classes):
		# CREATE GENERIC MODEL
		model = tf.keras.models.Sequential([
			tf.keras.layers.Flatten(input_shape=(input_shape[1:])),
			tf.keras.layers.Dense(128, activation="relu"),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(num_classes, activation="softmax"),
		])
		model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
		# CREATE GENERIC MODEL

		return model

	"""
    create_CNN 

    :param input_shape: Quantidade de amostras para treino
    :param num_classes: Quantidade de amostras para teste
    """
	def create_CNN(self, input_shape, num_classes):
		if len(input_shape) == 3:
			input = Input(shape=(input_shape[1], input_shape[2], 1))
		else:
			input = Input(shape=(input_shape[1:]))
		x = Conv2D(128, (5, 5), activation='relu', strides=(1, 1), padding='same')(input)
		x = MaxPool2D(pool_size=(2, 2))(x)

		x = Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding='same')(x)
		x = MaxPool2D(pool_size=(2, 2))(x)
		x = BatchNormalization()(x)

		x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same')(x)
		x = MaxPool2D(pool_size=(2, 2))(x)
		x = BatchNormalization()(x)

		x = Flatten()(x)

		x = Dense(100, activation='relu')(x)
		x = Dense(100, activation='relu')(x)
		x = Dropout(0.25)(x)

		out = Dense(num_classes, activation='softmax')(x)

		model = Model(inputs=input, outputs=[out])
		model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

		return model

	"""
    create_LogisticRegression 

    :param input_shape: Quantidade de amostras para treino
    :param num_classes: Quantidade de amostras para teste
    """
	def create_LogisticRegression(self, input_shape, num_classes):
		if len(input_shape) == 3:
			input = Input(shape=(input_shape[1], input_shape[2], 1))
		else:
			input = Input(shape=(input_shape[1:]))

		x = Flatten()(input)
		out = Dense(num_classes, activation='sigmoid')(x)

		model = Model(inputs=input, outputs=[out])
		model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		return model
