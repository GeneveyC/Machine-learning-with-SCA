from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics, initializers

from models.metrics_costume import mean_rank


class NeuralNetwork():
	def __init__(self, nb_channels, arch, save_folder):
		self.nb_channels = nb_channels
		self.arch = arch
		self.save_folder = save_folder

		if self.arch == "big_data":
			self.model = self.cnn_perso_700_big_data()
		elif self.arch == "ascad":
			self.model = self.cnn_perso_700_1()
		elif self.arch == "leakly":
			self.mdoel = self.cnn_perso_700_2()
		elif self.arch == "leakly+dropout":
			self.model = self.cnn_perso_700_3()
		elif self.arch == "zaid":
			self.model = self.zaid_desync_0()
		elif self.arch == "noConv1":
			self.model = self.noConv1_desync_0()


	def zaid_desync_0(self, learning_rate=0.00001, classes=256):
		# Designing input layer
		input_shape = (700, self.nb_channels)
		img_input = Input(shape=input_shape)

		# 1st convolutional block
		x = Conv1D(4, 1, kernel_initializer='he_uniform', activation='selu', padding='same', name='block1_conv1')(img_input)
		x = BatchNormalization(name='block1_norm1')(x)
		x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
            
		x = Flatten(name='flatten')(x)

		# Classification layer
		x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc1')(x)
		x = Dense(10, kernel_initializer='he_uniform', activation='selu', name='fc2')(x)
            
		# Logits layer              
		x = Dense(classes, activation='softmax', name='predictions')(x)

		# Create model
		inputs = img_input
		model = Model(inputs, x, name='zaid')
		optimizer = Adam(lr=learning_rate)
		model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy', mean_rank])
		return model


	def noConv1_desync_0(self, learning_rate=0.00001,classes=256):
		input_shape = (700, self.nb_channels)
		trace_input = Input(shape=input_shape)
		x = AveragePooling1D(2, strides=2, name='initial_pool')(trace_input)
		x = Flatten(name='flatten')(x)

		x = Dense(10, activation='selu', name='fc1')(x)
		x = Dense(10, activation='selu', name='fc2')(x)          
		x = Dense(classes, activation='softmax', name='predictions')(x)

		model = Model(trace_input, x, name='noConv1_ascad_desync_0')
		optimizer = Adam(lr=learning_rate)
		model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy', mean_rank])
		return model


	### CNN perso model (default model with relu function)
	def cnn_perso_700_big_data(self, classes=256):
		# From VGG16 design
		input_shape = (700, 1)
		img_input = Input(shape=input_shape)
		# Block 1
		x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
		x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
		# Block 2
		x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
		# Block 3
		x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
		# Block 4
		x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
		# Block 5
		x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
		# Classification block
		x = Flatten(name='flatten')(x)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)

		inputs = img_input
		# Create model.
		model = Model(inputs, x, name='cnn_best')
		optimizer = RMSprop(lr=0.00001)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', mean_rank])
		return model


	### CNN perso model (default model with relu function)
	def cnn_perso_700_1(self, classes=256):
		# From VGG16 design
		input_shape = (700, self.nb_channels)
		img_input = Input(shape=input_shape)
		# Block 1
		x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(img_input)
		x = AveragePooling1D(2, strides=2, name='block1_pool')(x)
		# Block 2
		x = Conv1D(128, 11, activation='relu', padding='same', name='block2_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block2_pool')(x)
		# Block 3
		x = Conv1D(256, 11, activation='relu', padding='same', name='block3_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block3_pool')(x)
		# Block 4
		x = Conv1D(512, 11, activation='relu', padding='same', name='block4_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block4_pool')(x)
		# Block 5
		x = Conv1D(512, 11, activation='relu', padding='same', name='block5_conv1')(x)
		x = AveragePooling1D(2, strides=2, name='block5_pool')(x)
		# Classification block
		x = Flatten(name='flatten')(x)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)

		inputs = img_input
		# Create model.
		model = Model(inputs, x, name='cnn_best')
		optimizer = RMSprop(lr=0.00001)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', mean_rank])
		return model


	### CNN perso model (LeaklyReLU activation function)
	def cnn_perso_700_2(self, classes=256):
		# From VGG16 design
		model = Sequential()
		# Block 1
		model.add(Conv1D(64, 11, input_shape=(700, self.nb_channels), padding='same', name='block1_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(AveragePooling1D(2, strides=2, name='block1_pool'))
		
		# Block 2
		model.add(Conv1D(128, 11, padding='same', name='block2_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(AveragePooling1D(2, strides=2, name='block2_pool'))
		
		# Block 3
		model.add(Conv1D(256, 11, padding='same', name='block3_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(AveragePooling1D(2, strides=2, name='block3_pool'))

		# Block 4
		model.add(Conv1D(512, 11, padding='same', name='block4_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(AveragePooling1D(2, strides=2, name='block4_pool'))

		# Block 5
		model.add(Conv1D(512, 11, padding='same', name='block5_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(AveragePooling1D(2, strides=2, name='block5_pool'))

		# Classification block
		model.add(Flatten(name='flatten'))
		
		model.add(Dense(4096, name='fc1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dense(4096, name='fc2'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dense(classes, name='predictions'))
		model.add(Activation('softmax'))

		optimizer = RMSprop(lr=0.00001)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', mean_rank])
		return model


	### CNN perso model (LeaklyReLU activation function and dropout)
	def cnn_perso_700_3(self, classes=256):
		# From VGG16 design
		model = Sequential()
		
		# Block 1
		model.add(Conv1D(64, 11, input_shape=(700, self.nb_channels), padding='same', name='block1_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(AveragePooling1D(2, strides=2, name='block1_pool'))

		# Block 2
		model.add(Conv1D(128, 11, padding='same', name='block2_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(AveragePooling1D(2, strides=2, name='block2_pool'))

		# Block 3
		model.add(Conv1D(256, 11, padding='same', name='block3_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(AveragePooling1D(2, strides=2, name='block3_pool'))

		# Block 4
		model.add(Conv1D(512, 11, padding='same', name='block4_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(AveragePooling1D(2, strides=2, name='block4_pool'))

		# Block 5
		model.add(Conv1D(512, 11, padding='same', name='block5_conv1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(AveragePooling1D(2, strides=2, name='block5_pool'))

		# Classification block
		model.add(Flatten(name='flatten'))
		
		model.add(Dense(4096, name='fc1'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(Dense(4096, name='fc2'))
		model.add(LeakyReLU(alpha=0.3))
		model.add(Dropout(0.2))
		model.add(Dense(classes, name='predictions'))
		model.add(Activation('softmax'))

		optimizer = RMSprop(lr=0.00001)
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', mean_rank])
		return model