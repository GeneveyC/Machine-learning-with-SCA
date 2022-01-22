from __future__ import print_function, division

import os
import sys
import h5py
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Activation, Dense, Conv1D, AveragePooling1D, Input, Reshape, Dropout, Flatten, BatchNormalization, LeakyReLU, concatenate, Conv1DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import activations

from src.multichannels.normalizations import instance_local_stand

import matplotlib.pyplot as plt
import numpy as np


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)


def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
		print("Error: provided file path '%s' does not exist!" % file_path)
		sys.exit(-1)
	return


def load_ascad(ascad_database_file, load_metadata=False):
	check_file_exists(ascad_database_file)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(ascad_database_file, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % ascad_database_file)
		sys.exit(-1)
	# Load profiling traces
	X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.int8)
	# Load profiling labels
	Y_profiling = np.array(in_file['Profiling_traces/labels'])
	# Load attacking traces
	X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.int8)
	# Load attacking labels
	Y_attack = np.array(in_file['Attack_traces/labels'])
	if load_metadata == False:
		return (X_profiling, Y_profiling), (X_attack, Y_attack)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['Profiling_traces/metadata'], in_file['Attack_traces/metadata'])



def load_perso_with_validation_for_kfold(database, channels, ntp=1.0, load_metadata=False):
		check_file_exists(database)
		# Open the ASCAD database HDF5 for reading
		try:
			in_file  = h5py.File(database, "r")
		except:
			print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database)
			sys.exit(-1)

		# Load profiling traces
		if len(channels) == 1:
			X_profiling = np.array(in_file["LEARN/traces_"+channels[0]])
			X_profiling = X_profiling[0:int(ntp*X_profiling.shape[0]), :]
			
			#X_attack = np.array(in_file["ATTACK/traces_"+channels[0]])
		# elif len(channels) == 2:
		# 	X2 = np.array(in_file["LEARN/traces_"+channels[0]])
		# 	X3 = np.array(in_file["LEARN/traces_"+channels[1]])
		# 	X = np.dstack((X2, X3))
		# 	X_profiling = X[:, :, :]

		# 	X5 = np.array(in_file['ATTACK/traces_'+channels[0]])
		# 	X6 = np.array(in_file['ATTACK/traces_'+channels[1]])
		# 	X_2 = np.dstack((X5, X6))
		# 	X_attack = X_2[:, :, :]
		# elif len(channels) == 3:
		# 	X2 = np.array(in_file['LEARN/traces_'+channels[0]])
		# 	X3 = np.array(in_file['LEARN/traces_'+channels[1]])
		# 	X4 = np.array(in_file['LEARN/traces_'+channels[2]])
		# 	X = np.dstack((X2, X3, X4))
		# 	X_profiling = X[:, :, :]
            
		# 	X5 = np.array(in_file['ATTACK/traces_'+channels[0]])
		# 	X6 = np.array(in_file['ATTACK/traces_'+channels[1]])
		# 	X7 = np.array(in_file['ATTACK/traces_'+channels[2]])
		# 	X_2 = np.dstack((X5, X6, X7))
		# 	X_attack = X_2[:, :, :]
		else:
			raise NotImpelementedError

		# Load profiling labels
		Y_profiling = np.array(in_file['LEARN/labels'])
		Y_profiling = Y_profiling[0:int(ntp*Y_profiling.shape[0])]

		#Y_attack = np.array(in_file["ATTACK/labels"])

		if load_metadata == False:
			return (X_profiling, Y_profiling), (X_attack, Y_attack)
		else:
			return (X_profiling, Y_profiling), (None, None), (in_file['LEARN/data'], None)


class ACGAN:
	def __init__(self, n_channels, n_classes, save_folder, model_d="acgan-dnn", transfert=False, load_folder=None):
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.save_folder = save_folder
		self.latent_dim = 100
		self.model_d = model_d

		if transfert:
		 	if self.model_d == "acgan-dnn":
		 		self.discriminator = load_model(load_folder+"discriminator_best.hdf5")
		 		self.generator = load_model(load_folder+"generator_best.hdf5")
		 		self.cgan_model = load_model(load_folder+"cgan_best.hdf5")
		 	else:
		 		raise NotImplementedError
		else:
			optimizer = Adam(0.0002, 0.5)
			
			# Build and compile the discriminator
			if self.model_d == "acgan-dnn":
				self.discriminator = self.build_discriminator_dnn()
			elif self.model_d == "acgan-mlp":
				self.discriminator = self.build_discriminator_mlp()

			self.discriminator.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
										optimizer=optimizer,
										metrics=['accuracy'])

			self.generator = self.build_generator_mlp()

			noise = Input(shape=(self.latent_dim, ))
			label = Input(shape=(self.n_classes,))
			img = self.generator([noise, label])

			# during generator updating,  the discriminator is fixed (will not be updated).
			self.discriminator.trainable = False

			# The discriminator takes generated image and label as input and determines its validity
			validity = self.discriminator(img)

			self.cgan_model = Model([noise, label], validity)
			self.cgan_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'],
									optimizer=optimizer,
									metrics=['accuracy'])


	def build_discriminator_mlp(self, node=200, layer_nb=6):
		inputs = Input(shape=(700,), name='descriminator_input')
		x = Dense(node, activation='relu')(inputs)

		for _ in range(layer_nb-2):
			x = Dense(node, activation='relu')(x)

		out1 = Dense(1, activation='sigmoid', name='dense_loss_bin')(x)
		out2 = Dense(256, activation='softmax', name='dense_loss_cat')(x)

		discriminator = Model(inputs, [out1, out2], name='discriminator')
		return discriminator


	def build_discriminator_dnn(self):
		# From VGG16 design
		inputs = Input(shape=(700,1))

		# Block 1
		x = Conv1D(64, 11, activation='relu', padding='same', name='block1_conv1')(inputs)
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

		out1 = Dense(1, activation='sigmoid', name='dense_loss_bin')(x)
		out2 = Dense(256, activation='softmax', name='dense_loss_cat')(x)

		# Create model.
		discriminator = Model(inputs, [out1, out2], name='discriminator')
		return discriminator


	def build_discriminator_dnn_optimized_1(self):
		# From VGG16 design
		inputs = Input(shape=(700,1))

		# Block 1
		x = Conv1D(64, 11, strides=2, padding='same', name='block1_conv1')(inputs)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)
		#x = AveragePooling1D(2, strides=2, name='block1_pool')(x)

		# Block 2
		x = Conv1D(128, 11, strides=2, padding='same', name='block2_conv1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)
		#x = AveragePooling1D(2, strides=2, name='block2_pool')(x)

		# Block 3
		x = Conv1D(256, 11, strides=2, padding='same', name='block3_conv1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)
		#x = AveragePooling1D(2, strides=2, name='block3_pool')(x)

		# Block 4
		x = Conv1D(512, 11, strides=2, padding='same', name='block4_conv1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)
		#x = AveragePooling1D(2, strides=2, name='block4_pool')(x)

		# Block 5
		x = Conv1D(512, 11, strides=2, padding='same', name='block5_conv1')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)
		#x = AveragePooling1D(2, strides=2, name='block5_pool')(x)

		# Classification block
		x = Flatten(name='flatten')(x)

		out1 = Dense(1, activation='sigmoid', name='dense_loss_bin')(x)
		out2 = Dense(256, activation='softmax', name='dense_loss_cat')(x)

		# Create model.
		discriminator = Model(inputs, [out1, out2], name='discriminator')
		return discriminator


	def build_generator_mlp(self, latent_dim=100, n_classes=256):
		inputs = Input(shape=(latent_dim,), name='z_input')
		labels = Input(shape=(n_classes,), name='class_labels')
		x = concatenate([inputs, labels], axis=1)

		x = Dense(256)(x)
		x = LeakyReLU(0.2)(x)
		x = BatchNormalization()(x)

		x = Dense(512)(x)
		x = LeakyReLU(0.2)(x)
		x = BatchNormalization()(x)

		x = Dense(1024)(x)
		x = LeakyReLU(0.2)(x)
		x = BatchNormalization()(x)
 
		x = Dropout(0.5)(x)
		x = Dense(700, activation='tanh')(x)

		# input is conditioned by labels
		generator = Model([inputs, labels], x, name='generator')
		return generator


	def build_generator_dnn_optimized_1(self, latent_dim=100, n_classes=256):
		inputs = Input(shape=(latent_dim,), name='z_input')
		labels = Input(shape=(n_classes,), name='class_labels')
		x = concatenate([inputs, labels], axis=1)

		x = Reshape((x.shape[1], 1))(x)

		# Block 1
		x = Conv1D(64, 11, strides=2, padding='same', name='block1_conv1_g')(x)
		x = BatchNormalization()(x)
		x = Activation(activations.relu)(x)

		# Block 2
		x = Conv1D(128, 11, strides=2, padding='same', name='block2_conv1_g')(x)
		x = BatchNormalization()(x)
		x = Activation(activations.relu)(x)

		# Block 3
		x = Conv1D(256, 11, strides=2, padding='same', name='block3_conv1_g')(x)
		x = BatchNormalization()(x)
		x = Activation(activations.relu)(x)
		
		# Block 4
		x = Conv1D(512, 11, strides=2, padding='same', name='block4_conv1_g')(x)
		x = BatchNormalization()(x)
		x = Activation(activations.relu)(x)

		# Block 5
		x = Conv1D(512, 11, strides=2, padding='same', name='block5_conv1_g')(x)
		x = BatchNormalization()(x)
		x = Activation(activations.relu)(x)

		# Classification block
		x = Flatten(name='flatten_g')(x)
		x = Dense(700, activation='tanh')(x)

		# input is conditioned by labels
		generator = Model([inputs, labels], x, name='generator')
		return generator


	def generate_noise(self, type_of_noise, batch_size):
		if type_of_noise == "normal_noise":
			return np.random.normal(0, 1, size=[batch_size, self.latent_dim])
		elif type_of_noise == "uniform_noise":
			return np.random.uniform(-1.0, 1.0, size=[batch_size, self.latent_dim])


	def train(self, x_train, y_train, x_validation, y_validation, folder_samples, epochs=10, batch_size=128, save_at=5):	
		batch_per_epoch = int(x_train.shape[0] / batch_size)
		n_steps = batch_per_epoch * epochs
		half_batch = int(batch_size/2)

		# reshape for DNN
		if self.model_d == "acgan-dnn":
			x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
			x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))	


		y_train = to_categorical(y_train, num_classes=256)
		y_validation = to_categorical(y_validation, num_classes=256)

		# Adversarial ground truths
		real_train = np.ones((half_batch, 1))
		fake_train = np.zeros((half_batch, 1))

		real_test = np.ones((y_validation.shape[0], 1))
		fake_test = np.zeros((y_validation.shape[0], 1))

		list_d_loss_both_test = []
		list_d_loss_bin_test = []
		list_d_loss_cat_test = []
		list_d_acc_bin_test = []
		list_d_acc_cat_test = []

		old_loss_on_validation = 1000.0
		for i in range(n_steps):
			#  --------------------- Train Discriminator ---------------------
			# Select a random half batch of images
			idx = np.random.randint(0, x_train.shape[0], size=half_batch)
			imgs, labels_real = x_train[idx], y_train[idx]

			# Generate sample noise for generator input
			noise = self.generate_noise("normal_noise", half_batch)

			# Generate a half batch of new images
			# we can use labels instead of fake_labels; because it is fake for noise
			gen_imgs = self.generator.predict([noise, labels_real])

			if self.model_d == "acgan-dnn":
				gen_imgs = gen_imgs.reshape((gen_imgs.shape[0], gen_imgs.shape[1], 1))


			# --------------------- Train the Discriminator ---------------------
			d_loss_real = self.discriminator.train_on_batch(imgs, [real_train, labels_real])
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake_train, labels_real])
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			#  --------------------- Train the Generator ---------------------
			# Condition on labels (random one-hot labels)
			fake_labels = np.eye(self.n_classes)[np.random.choice(self.n_classes, half_batch)]

			# Train the generator
			cgan_loss = self.cgan_model.train_on_batch([noise, fake_labels], [real_train, fake_labels])
			print(d_loss)

			#print('>%d, d_loss=[%.3f,%.3f], g_loss=[%.3f,%.3f]' % (i+1, d_loss[1], d_loss[2], cgan_loss[1], cgan_loss[2]))

			if i % batch_per_epoch == 0:
				noise_test = self.generate_noise("normal_noise", y_validation.shape[0])
				gen_imgs_test = self.generator.predict([noise_test, y_validation])

				if self.model_d == "acgan-dnn":
					gen_imgs_test = gen_imgs_test.reshape((gen_imgs_test.shape[0], gen_imgs_test.shape[1], 1))

				loss_eval = self.discriminator.evaluate(x_validation, [real_test, y_validation])
				list_d_loss_both_test.append(loss_eval[0])
				list_d_loss_bin_test.append(loss_eval[1])
				list_d_loss_cat_test.append(loss_eval[2])
				list_d_acc_bin_test.append(loss_eval[3])
				list_d_acc_cat_test.append(loss_eval[4])

				epoch = i / batch_per_epoch
				self.generate_sample(x_validation, y_validation, epoch, folder_samples)
				
				if epoch % save_at == 0:
					self.discriminator.save(self.save_folder+"discriminator_%04d.hdf5" % (epoch))
					self.generator.save(self.save_folder+"generator_%04d.hdf5" % (epoch))
					self.cgan_model.save(self.save_folder+"cgan_%04d.hdf5" % (epoch))


				# save the best discriminator
				if loss_eval[2] < old_loss_on_validation:
					self.discriminator.save(self.save_folder+"discriminator_best.hdf5")
					self.generator.save(self.save_folder+"generator_best.hdf5")
					self.cgan_model.save(self.save_folder+"cgan_best.hdf5")
					old_loss_on_validation = loss_eval[2]



		#self.discriminator = load_model(self.save_folder+"discriminator.hdf5")
		self.generate_sample(x_validation, y_validation, epochs, folder_samples)
		self.discriminator.save(self.save_folder+"discriminator_%04d.hdf5" % (epochs))
		self.generator.save(self.save_folder+"generator_%04d.hdf5" % (epochs))
		self.cgan_model.save(self.save_folder+"cgan_%04d.hdf5" % (epoch))

		noise_test = self.generate_noise("normal_noise", y_validation.shape[0])
		gen_imgs_test = self.generator.predict([noise_test, y_validation])
		
		if self.model_d == "acgan-dnn":
			gen_imgs_test = gen_imgs_test.reshape((gen_imgs_test.shape[0], gen_imgs_test.shape[1], 1))

		loss_eval = self.discriminator.evaluate(x_validation, [real_test, y_validation])
		print(loss_eval)

		return list_d_loss_both_test, list_d_loss_bin_test, list_d_loss_cat_test, list_d_acc_bin_test, list_d_acc_cat_test



	def generate_sample(self, x_validation, y_validation, epoch, folder_samples):
		noise_test = self.generate_noise("normal_noise", y_validation.shape[0])
		gen_imgs_test = self.generator.predict([noise_test, y_validation])

		if self.model_d == "acgan-dnn":
			gen_imgs_test = gen_imgs_test.reshape((gen_imgs_test.shape[0], gen_imgs_test.shape[1], 1))

		# compute SNR
		if epoch == 0:
			snr_real = self.compute_SNR(x_validation, np.argmax(y_validation, axis=1))

			plt.figure()
			plt.plot(snr_real, label="real")
			plt.xticks(fontsize=12)
			plt.yticks(fontsize=12)
			plt.xlabel('Time samples', fontsize=12)
			plt.ylabel('SNR value', fontsize=12)
			plt.legend(prop={"size": 12})
			plt.savefig(folder_samples+"/snr_real_gan.png")

		snr_fake = self.compute_SNR(gen_imgs_test, np.argmax(y_validation, axis=1))

		plt.figure()
		plt.plot(snr_fake, label="fake")
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.xlabel('Time samples', fontsize=12)
		plt.ylabel('SNR value', fontsize=12)
		plt.legend(prop={"size": 12})
		plt.savefig(folder_samples+"/snr_fake_%04d_gan.png" % (epoch))


	def compute_SNR(self, curves, partitions):
		self.nb_samples = curves.shape[1]
		self.counts = {}
		self.means = {}
		self.vars = {}

		for i in range(0, curves.shape[0]):
			curve = curves[i]
			partition = partitions[i]
			if partition in self.counts:
				self.counts[partition] += 1
				self.means[partition] = np.add(self.means[partition],curve)
				self.vars[partition] = np.add(self.vars[partition],np.square(curve))
			else:
				self.counts[partition] = 1
				self.means[partition] = curve
				self.vars[partition] = np.square(curve)

		for partition in self.counts:
			self.means[partition] /= self.counts[partition]
			self.vars[partition] /= self.counts[partition]
			self.vars[partition] -= np.square(self.means[partition])
		snr = np.zeros((self.nb_samples,),dtype=float)
		for i in range(self.nb_samples):
			means = []
			variances = []
			for partition in self.counts:
				if self.counts[partition]>3:
					means.append(self.means[partition][i])
					variances.append(self.vars[partition][i])
			snr[i] = np.var(means)/np.mean(variances)
		return snr


class Simple_Classifier():
	def __init__(self, n_channels, n_classes, save_folder, model_d="dnn", number_PoI=700):
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.save_folder = save_folder
		self.model_d = model_d
		self.number_PoI = number_PoI

		#optimizer = Adam(0.0002, 0.5)
		#optimizer = RMSprop(lr=0.00001)
		optimizer = RMSprop(lr=1e-5)

		# Build and compile the discriminator
		if self.model_d == "dnn":
			self.model = self.build_best_cnn()
		elif self.model_d == "mlp":
			self.model = self.build_best_mlp()

		self.model.compile(loss=['categorical_crossentropy'],
								optimizer=optimizer,
								metrics=['accuracy'])


	def build_best_mlp(self, node=200, layer_nb=6):
		inputs = Input(shape=(self.number_PoI,), name='descriminator_input')
		x = Dense(node, activation='relu')(inputs)

		for _ in range(layer_nb-2):
			x = Dense(node, activation='relu')(x)

		out2 = Dense(256, activation='softmax')(x)

		model = Model(inputs, out2, name='best_model')
		return model


	def build_best_cnn(self):
		# From VGG16 design
		input_shape = (self.number_PoI, 1)
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
		x = Dense(256, activation='softmax', name='predictions')(x)

		inputs = img_input
		# Create model.
		model = Model(inputs, x, name='cnn_best')
		return model


	def train(self, x_train, y_train, x_validation, y_validation, epochs=10, batch_size=128):

		print(x_train.shape)
		print(x_validation.shape)

		# reshape for DNN
		if self.model_d == "dnn":
			x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
			x_validation = x_validation.reshape((x_validation.shape[0], x_validation.shape[1], 1))

		y_train = to_categorical(y_train, num_classes=256)
		y_validation = to_categorical(y_validation, num_classes=256)

		save_model = ModelCheckpoint(self.save_folder+self.model_d+".hdf5", monitor='val_loss', mode='min', save_best_only=True)
		callbacks=[save_model]

		hist = self.model.fit(x=x_train, 
						y=y_train, 
						batch_size=batch_size, 
						verbose = 1, 
						epochs=epochs, 
						validation_data=(x_validation, y_validation),
						callbacks=callbacks)

		self.model = load_model(self.save_folder+self.model_d+".hdf5")

		loss_eval = self.model.evaluate(x_validation, y_validation)
		print(loss_eval)

		return hist



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command')

	# parser create
	parser_load = subparsers.add_parser('create', help='create dataset')
	parser_load.add_argument('-database', type=str, dest='database', help='database used for the profiling')
	parser_load.add_argument('-save_scaler', type=str, dest='save_scaler', help='save scaler')
	parser_load.add_argument('-save_data', type=str, dest='save_data', help='folder to save the data')
	parser_load.add_argument('-channels', type=str, dest="channels", help="channels used for profiling/attack")
	parser_load.add_argument('-ntp', type=float, dest="ntp", help="number of data to select")
	parser_load.add_argument('-ascad', type=int, dest="ascad", help="bool to see if ascad")

	# parser train
	parser_train = subparsers.add_parser('train', help='train models (acgan-dnn, acgan-mlp or dnn)')
	parser_train.add_argument('-load_data', type=str, dest='load_data', help='folder to load the data')
	parser_train.add_argument('-save_model', type=str, dest='save_model', help='folder to save model')
	parser_train.add_argument('-save_loss', type=str, dest='save_loss', help='folder to save loss')
	parser_train.add_argument('-save_samples', type=str, dest='save_samples', help='folder to save samples')
	parser_train.add_argument('-network', type=str, dest="network", help="network to use (acgan-dnn, acgan-mlp, dnn or mlp)")
	parser_train.add_argument('-PoI', type=int, dest="number_PoI", help="frame used (700 or 1024)")
	parser_train.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")

	# parser transfert
	parser_transfert = subparsers.add_parser('transfert', help='transfert models (acgan-dnn or dnn)')
	parser_transfert.add_argument('-load_data', type=str, dest='load_data', help='folder to load the data')
	parser_transfert.add_argument('-load_model', type=str, dest='load_model', help='folder to load model')
	parser_transfert.add_argument('-save_model', type=str, dest='save_model', help='folder to save model')
	parser_transfert.add_argument('-save_loss', type=str, dest='save_loss', help='folder to save loss')
	parser_transfert.add_argument('-save_samples', type=str, dest='save_samples', help='folder to save samples')
	parser_transfert.add_argument('-network', type=str, dest="network", help="network to use (acgan-dnn or dnn)")
	parser_transfert.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")

	# parser generate samples with pre-trained models
	parser_generate = subparsers.add_parser('generate', help='generate samples with pre-trained model (acgan-dnn or acgan-mlp)')
	parser_generate.add_argument('-load_model', type=str, dest='folder_model', help='folder to load the model')
	parser_generate.add_argument('-samples', type=int, dest='samples', help='number of samples to generate')
	parser_generate.add_argument('-save', type=str, dest="save_traces", help="folder to save the new trace")
	parser_generate.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")


	args = parser.parse_args()
	if len(sys.argv) < 2: 
		parser.print_help()
		sys.exit(1)

	if args.command == 'create':
		database = args.database
		save_folder_scaler = args.save_scaler
		save_data = args.save_data
		arg_channels = args.channels
		channels = arg_channels.split('-')
		ntp = args.ntp
		ascad = args.ascad

		if ascad:
			(X, Y), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_ascad(database, True)
			X_profiling = X[0:40000, 188:700]
			Y_profiling = Y[0:40000]
			X_validation = X[40000:, 188:700]
			Y_validation = Y[40000:]
		else:
			(X, Y), (X_attack, Y_attack), (Metadata_profiling, Metadata_attack) = load_perso_with_validation_for_kfold(database, channels, ntp, True)
			X_profiling = X[0:40000, :]
			Y_profiling = Y[0:40000]
			X_validation = X[40000:, :]
			Y_validation = Y[40000:]

		print(X.shape)
		print(Y.shape)

		print(X_profiling.shape)
		print(Y_profiling.shape)
		print(X_validation.shape)
		print(Y_validation.shape)
	
		# Normalize for tanh
		scaler = MinMaxScaler(feature_range=(-1, 1))
		scaler.fit(X_profiling)

		# save scaler
		joblib.dump(scaler, save_folder_scaler+'scaler.pkl')

		X_profiling = scaler.transform(X_profiling)
		X_validation = scaler.transform(X_validation)

		# Save for transfert learning
		np.save(save_data+'test_Xprofiling.npy', X_profiling)
		np.save(save_data+'test_Yprofiling.npy', Y_profiling)
		np.save(save_data+'test_Xvalidation.npy', X_validation)
		np.save(save_data+'test_Yvalidation.npy', Y_validation)

	elif args.command == 'train':
		save_folder_model = args.save_model
		save_folder_loss = args.save_loss
		save_folder_samples = args.save_samples
		load_data = args.load_data
		network = args.network
		number_PoI = args.number_PoI
		gpu = args.gpu

		X_profiling = np.load(load_data+'test_Xprofiling.npy')
		X_validation = np.load(load_data+'test_Xvalidation.npy')
		Y_profiling = np.load(load_data+'test_Yprofiling.npy')
		Y_validation = np.load(load_data+'test_Yvalidation.npy')

		nb_channels = 1
		if network == "dnn" or network == "mlp":
			with tf.device("/gpu:"+gpu):
				model = Simple_Classifier(nb_channels, 256, save_folder_model, network, number_PoI)
				hist = model.train(X_profiling, Y_profiling, X_validation, Y_validation, epochs=100, batch_size=128)

			plt.figure()
			plt.plot(hist.history['val_loss'], label="loss")
			plt.legend()
			plt.savefig(save_folder_loss+"loss_cat_"+network+".png")

			plt.figure()
			plt.plot(hist.history['val_accuracy'], label='acc')
			plt.legend()
			plt.savefig(save_folder_loss+"acc_cat_"+network+".png")

		elif network == "acgan-dnn" or network == "acgan-mlp":
			with tf.device("/gpu:"+gpu):
				acgan = ACGAN(nb_channels, 256, save_folder_model, network)
				d_loss_both_test, d_loss_bin_test, d_loss_cat_test, d_acc_bin_test, d_acc_cat_test = acgan.train(X_profiling, Y_profiling, X_validation, Y_validation, save_folder_samples, epochs=100, batch_size=200, save_at=10)

			plt.figure()
			plt.plot(d_loss_both_test, label="d_loss")
			plt.legend()
			plt.savefig(save_folder_loss+"loss_both_"+network+".png")

			plt.figure()
			plt.plot(d_loss_bin_test, label="d_loss")
			plt.legend()
			plt.savefig(save_folder_loss+"loss_bin_"+network+".png")

			plt.figure()
			plt.plot(d_loss_cat_test, label='d_loss')
			plt.legend()
			plt.savefig(save_folder_loss+"loss_cat_"+network+".png")

			plt.figure()
			plt.plot(d_acc_bin_test, label='d_acc')
			plt.legend()
			plt.savefig(save_folder_loss+"acc_bin_"+network+".png")

			plt.figure()
			plt.plot(d_acc_cat_test, label='d_acc')
			plt.legend()
			plt.savefig(save_folder_loss+"acc_cat_"+network+".png")
		else:
			raise NotImpelementedError
	elif args.command == "transfert":
		save_folder_model = args.save_model
		load_folder_model = args.load_model
		save_folder_loss = args.save_loss
		save_folder_samples = args.save_samples
		load_data = args.load_data
		network = args.network
		gpu = args.gpu

		X_profiling = np.load(load_data+'test_Xprofiling.npy')
		X_validation = np.load(load_data+'test_Xvalidation.npy')
		Y_profiling = np.load(load_data+'test_Yprofiling.npy')
		Y_validation = np.load(load_data+'test_Yvalidation.npy')

		nb_channels = 1
		if network == "acgan-dnn":
			with tf.device("/gpu:"+gpu):
				acgan = ACGAN(nb_channels, 256, save_folder_model, network, True, load_folder_model)
				d_loss_both_test, d_loss_bin_test, d_loss_cat_test, d_acc_bin_test, d_acc_cat_test = acgan.train(X_profiling, Y_profiling, X_validation, Y_validation, save_folder_samples, epochs=100, batch_size=200, save_at=10)

			plt.figure()
			plt.plot(d_loss_both_test, label="d_loss")
			plt.legend()
			plt.savefig(save_folder_loss+"loss_both_"+network+".png")

			plt.figure()
			plt.plot(d_loss_bin_test, label="d_loss")
			plt.legend()
			plt.savefig(save_folder_loss+"loss_bin_"+network+".png")

			plt.figure()
			plt.plot(d_loss_cat_test, label='d_loss')
			plt.legend()
			plt.savefig(save_folder_loss+"loss_cat_"+network+".png")

			plt.figure()
			plt.plot(d_acc_bin_test, label='d_acc')
			plt.legend()
			plt.savefig(save_folder_loss+"acc_bin_"+network+".png")

			plt.figure()
			plt.plot(d_acc_cat_test, label='d_acc')
			plt.legend()
			plt.savefig(save_folder_loss+"acc_cat_"+network+".png")
	elif args.command == "generate":
		folder_model = args.folder_model
		number_samples = args.samples
		save_traces = args.save_traces
		gpu = args.gpu

		with tf.device("/gpu:"+gpu):
			generator = load_model(folder_model+"generator.hdf5")
			noise = np.random.normal(0, 1, size=[number_samples, 100])
			fake_labels = np.eye(256)[np.random.choice(256, number_samples)]
			trace_generated = generator.predict([noise, fake_labels])
			print(trace_generated.shape)

			first_trace = trace_generated[0, :]

			plt.figure()
			plt.plot(first_trace)
			plt.savefig(save_traces+"/0.png")
	else:
		raise NotImplementedError
