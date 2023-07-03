import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, Flatten, MaxPool2D, Dense, InputLayer, BatchNormalization, Dropout, MaxPooling2D, concatenate
from tensorflow.keras import initializers
from keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf.random.set_seed(42)

class ModelCreation():
	"""
    create_CNN 

    :param input_shape: Quantidade de amostras para treino
    :param num_classes: Quantidade de amostras para teste
    """
	def create_CNN_SIGN(self, input_shape, num_classes):
		model = Sequential()
		model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape[1:]))
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

		# Compilation of the model
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		return model

	"""
    create_DNN 

    :param input_shape: Quantidade de amostras para treino
    :param num_classes: Quantidade de amostras para teste
    """
	def create_DNN(self, input_shape, num_classes):
		input = Input(shape=(input_shape[1:]))
		x = Flatten()(input)
		x = Dense(512, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
		x = Dense(256, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
		x = Dense(32,  activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
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
			tf.keras.layers.Dense(128, activation="relu", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros()),
			tf.keras.layers.Dropout(0.2),
			tf.keras.layers.Dense(num_classes, activation="softmax", kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros()),
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
		x = Conv2D(128, (5, 5), activation='relu', strides=(1, 1), padding='same', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(input)
		x = MaxPool2D(pool_size=(2, 2))(x)

		x = Conv2D(64, (5, 5), activation='relu', strides=(2, 2), padding='same', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
		x = MaxPool2D(pool_size=(2, 2))(x)
		x = BatchNormalization()(x)

		x = Conv2D(32, (3, 3), activation='relu', strides=(2, 2), padding='same', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
		x = MaxPool2D(pool_size=(2, 2))(x)
		x = BatchNormalization()(x)

		x = Flatten()(x)

		x = Dense(100, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
		x = Dense(100, activation='relu', kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42), bias_initializer=initializers.Zeros())(x)
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
	
	
	def create_ImitationLearning(self, mask):
		image_size = (88, 200, 3)
		input_image = (image_size[0], image_size[1], image_size[2])
		input_speed = (1,)

		branch_config = [
		["Speed"], #Speed
		["Steer", "Gas", "Brake"], #Follow
		["Steer", "Gas", "Brake"], #Left
		["Steer", "Gas", "Brake"], #Right
		["Steer", "Gas", "Brake"] #Straight
		]

		branch_names = ['Speed', 'Follow', 'Left', 'Right', 'Straight']

		branches = []

		def conv_block(inputs, filters, kernel_size, strides):
			x = Conv2D(filters, (kernel_size, kernel_size), strides=strides, activation='relu')(inputs)
			x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
			x = BatchNormalization()(x)
			x = Dropout(0.2)(x)
			return x

		def fc_block(inputs, units):
			fc = Dense(units, activation='relu')(inputs)
			fc = Dropout(0.5)(fc)
			return fc

		xs = Input(shape=input_image, name='rgb')
		'''inputs, filters, kernel_size, strides'''
		""" Conv 1 """
		x = conv_block(xs, 32, 5, 2)
		x = conv_block(x, 32, 3, 1)
		""" Conv 2 """
		x = conv_block(x, 64, 3, 2)
		x = conv_block(x, 64, 3, 1)
		""" Conv 3 """
		x = conv_block(x, 128, 3, 2)
		x = conv_block(x, 128, 3, 1)
		""" Conv 4 """
		x = conv_block(x, 256, 3, 1)
		x = conv_block(x, 256, 3, 1)
		""" Reshape """
		x = Flatten()(x)
		""" FC1 """
		x = fc_block(x, 512)
		""" FC2 """
		x = fc_block(x, 512)
		"""Process Control"""
		""" Speed (measurements) """

		sm = Input(shape=input_speed, name='speed')
		speed = fc_block(sm, 128)
		speed = fc_block(speed, 128)
		""" Joint sensory """
		j = concatenate([x, speed])
		j = fc_block(j, 512)

		for i in range(len(branch_config)):
			if branch_config[i][0] == "Speed":
				branch_output = fc_block(x, 256)
				branch_output = fc_block(branch_output, 256)
			else:
				branch_output = fc_block(j, 256)
				branch_output = fc_block(branch_output, 256)
			fully_connected = Dense(len(branch_config[i]), name=branch_names[i])(branch_output)
			branches.append(fully_connected)

		if mask == 1: #Speed
			for branche in branches:
				if "Speed" in branche.name:
					model = Model(inputs=[xs], outputs=[branche])
					break
		elif mask == 2: #Follow
			for branche in branches:
				if "Follow" in branche.name:
					model = Model(inputs=[xs, sm], outputs=[branche])
					break
		elif mask == 3: #Left
			for branche in branches:
				if "Left" in branche.name:
					model = Model(inputs=[xs, sm], outputs=[branche])
					break
		elif mask == 4: #Right
			for branche in branches:
				if "Right" in branche.name:
					model = Model(inputs=[xs, sm], outputs=[branche])
					break
		else: #Straight
			for branche in branches:
				if "Straight" in branche.name:
					model = Model(inputs=[xs, sm], outputs=[branche])
					break
		return model
