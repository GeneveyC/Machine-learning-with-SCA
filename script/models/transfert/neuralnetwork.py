from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D, AveragePooling1D, AveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics, initializers

from models.metrics_costume import mean_rank


class NeuralNetwork_transfert_learning():
	def __init__(self, nb_channels, arch, save_folder, load_filename_model):
		self.nb_channels = nb_channels
		self.arch = arch
		self.save_folder = save_folder
		self.load_filename_model = load_filename_model

		if self.arch == "transfert0":
			self.model = self.cnn_perso_700_transfer_0()
		elif self.arch == "transfert1":
			self.model = self.cnn_perso_700_transfer_1()
		else:
			raise NotImpelementedError


	def cnn_perso_700_transfer_0(self, classes=256):
		model_final = load_model(self.load_filename_model, custom_objects={'mean_rank': mean_rank})
		return model_final


	def cnn_perso_700_transfer_1(self, classes=256):
		base_model = load_model(self.load_filename_model)
		inputs = base_model.input
		
		for layer in base_model.layers:
			layer.trainable = False

		x = base_model.layers[11].output
		x = Dense(4096, activation='relu', name='tf_fc1')(x)
		x = Dense(4096, activation='relu', name='tf_fc2')(x)
		x = Dense(classes, activation='softmax', name='predictions')(x)

		model_final = Model(inputs, x, name='tl_model')
		optimizer = RMSprop(lr=0.00001)
		model_final.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', mean_rank])
		print(model_final.summary())
		return model_final