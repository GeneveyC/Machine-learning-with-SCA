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
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy import load
from numpy import zeros
from numpy import ones

# weight initialization
init = RandomNormal(mean=0., stddev=0.02)


class Pix2Pix_1D():
	def __init__(self, arch, save_folder_model, save_folder_samples, save_folder_snr, optimizer_, activation_, lambda_1, load_folder_model=""):
		self.arch = arch
		self.save_folder_model = save_folder_model
		self.save_folder_samples = save_folder_samples
		self.save_folder_snr = save_folder_snr

		if self.arch == "default":
			self.clean_input_shape = (700, 1)
			self.noisy_input_shape = ( 700, 1)

			self.D = self.build_pix2pix_discriminator(self.clean_input_shape, self.noisy_input_shape, activation_)
			self.G = self.build_pix2pix_generator(self.noisy_input_shape, activation_)

			if optimizer_ == "rmsprop":
				self.D.compile(optimizer=RMSprop(), loss='binary_crossentropy')
			elif optimizer_ == "adam":
				self.D.compile(optimizer=Adam(), loss='binary_crossentropy')
			elif optimizer_ == "sgd":
				self.D.compile(optimizer=SGD(), loss='binary_crossentropy')
			else:
				raise NotImplementedError

			self.D.trainable = False
			noisy_input = Input(shape=(700, 1))
			clean_input = Input(shape=(700, 1))
			denoised_output = self.G([noisy_input])
			D_fake = self.D([noisy_input, denoised_output])
			self.D_of_G = Model(inputs=[noisy_input, clean_input], outputs=[D_fake, denoised_output])

			if optimizer_ == "rmsprop":
				self.D_of_G.compile(optimizer=RMSprop(), loss=['binary_crossentropy', 'mae'], loss_weights=[lambda_1, 1])
			elif optimizer_ == "adam":
				self.D_of_G.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mae'], loss_weights=[lambda_1, 1])
			elif optimizer_ == "sgd":
				self.D_of_G.compile(optimizer=SGD(), loss=['binary_crossentropy', 'mae'], loss_weights=[lambda_1, 1])
			else:
				raise NotImplementedError

		elif self.arch == "load":
			self.clean_input_shape = (700, 1)
			self.noisy_input_shape = ( 700, 1)

			self.D = load_model(load_folder_model+"model_best_gan_D.h5")
			self.G = load_model(load_folder_model+"model_best_gan_G.h5")

			if optimizer_ == "rmsprop":
				self.D.compile(optimizer=RMSprop(), loss='binary_crossentropy')
			elif optimizer_ == "adam":
				self.D.compile(optimizer=Adam(), loss='binary_crossentropy')
			elif optimizer_ == "sgd":
				self.D.compile(optimizer=SGD(), loss='binary_crossentropy')
			else:
				raise NotImplementedError

			self.D.trainable = False
			noisy_input = Input(shape=(700, 1))
			clean_input = Input(shape=(700, 1))
			denoised_output = self.G([noisy_input])
			D_fake = self.D([noisy_input, denoised_output])
			self.D_of_G = Model(inputs=[noisy_input, clean_input], outputs=[D_fake, denoised_output])

			if optimizer_ == "rmsprop":
				self.D_of_G.compile(optimizer=RMSprop(), loss=['binary_crossentropy', 'mae'], loss_weights=[lambda_1, 1])
			elif optimizer_ == "adam":
				self.D_of_G.compile(optimizer=Adam(), loss=['binary_crossentropy', 'mae'], loss_weights=[lambda_1, 1])
			elif optimizer_ == "sgd":
				self.D_of_G.compile(optimizer=SGD(), loss=['binary_crossentropy', 'mae'], loss_weights=[lambda_1, 1])
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError



	# define the discriminator model, by default = [64,128,256,512,512,1], strides=11
	def build_pix2pix_discriminator(self, noisy_input_shape, clean_input_shape,
								activation_):

		# source image input
		in_src_image = Input(shape=noisy_input_shape)
		# target image input
		in_target_image = Input(shape=clean_input_shape)

		# concatenate images channel-wise
		merged = Concatenate(-1)([in_src_image, in_target_image])

		# C64
		d = Conv1D(32, 11, padding='same', kernel_initializer=init)(merged)
		if activation_ == "prelu":
			d = PReLU()(d)
		elif activation_ == "lrelu":
			d = LeakyReLU()(d)
		elif activation_ == "tanh":
			d = Activation("tanh")(d)
		else:
			raise NotImplementedError

		# C128
		d = Conv1D(64, 11, padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		if activation_ == "prelu":
			d = PReLU()(d)
		elif activation_ == "lrelu":
			d = LeakyReLU()(d)
		elif activation_ == "tanh":
			d = Activation("tanh")(d)
		else:
			raise NotImplementedError

		# C256
		d = Conv1D(128, 11, padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		if activation_ == "prelu":
			d = PReLU()(d)
		elif activation_ == "lrelu":
			d = LeakyReLU()(d)
		elif activation_ == "tanh":
			d = Activation("tanh")(d)
		else:
			raise NotImplementedError

		# C512
		d = Conv1D(256, 11, padding='same', kernel_initializer=init)(d)
		d = BatchNormalization()(d)
		if activation_ == "prelu":
			d = PReLU()(d)
		elif activation_ == "lrelu":
			d = LeakyReLU()(d)
		elif activation_ == "tanh":
			d = Activation("tanh")(d)
		else:
			raise NotImplementedError

		# second last output layer
		# d = Conv1D(512, 11, padding='same', kernel_initializer=init)(d)
		# d = BatchNormalization()(d)
		# if activation_ == "prelu":
		# 	d = PReLU()(d)
		# elif activation_ == "lrelu":
		# 	d = LeakyReLU()(d)
		# elif activation_ == "tanh":
		# 	d = Activation("tanh")(d)
		# else:
		# 	raise NotImplementedError

		# patch output
		d = Conv1D(1, 11, padding='same', kernel_initializer=init)(d)
		if activation_ == "prelu":
			d = PReLU()(d)
		elif activation_ == "lrelu":
			d = LeakyReLU()(d)
		elif activation_ == "tanh":
			d = Activation("tanh")(d)
		else:
			raise NotImplementedError
		
		# Adding this
		#d = Reshape((700,))(d)
		#d = Dense(256, activation=None, use_bias=True)(d)
		#d = PReLU()(d)
		#d = Dense(128, activation=None, use_bias=True)(d)
		#d = PReLU()(d)
		#patch_out = Dense(1, activation=None, use_bias=True)(d)

		#patch_out = Activation('sigmoid')(d)
		d = Flatten(name='flatten')(d)
		patch_out = Dense(1, activation='sigmoid', name='dense_loss_bin')(d)

		# define model
		model = Model([in_src_image, in_target_image], patch_out)
		return model


	# define an encoder block
	def define_encoder_block(self, layer_in, n_filters, activation_, batchnorm=True):
		# weight initialization
		init = RandomNormal(mean=0., stddev=0.02)
		# add downsampling layer
		g = Conv1D(n_filters, 11, padding='same', kernel_initializer=init)(layer_in)
		# conditionally add batch normalization
		if batchnorm:
			g = BatchNormalization()(g, training=True)
		
		if activation_ == "prelu":
			g = PReLU()(g)
		elif activation_ == "lrelu":
			g = LeakyReLU()(g)
		elif activation_ == "tanh":
			g = Activation("tanh")(g)
		else:
			raise NotImplementedError

		return g


	# define a decoder block
	def decoder_block(self, layer_in, skip_in, n_filters, activation_, dropout=True):
		# weight initialization
		init = RandomNormal(mean=0., stddev=0.02)
		# add upsampling layer
		g = Conv1DTranspose(n_filters, 11, padding='same', kernel_initializer=init)(layer_in)
		# add batch normalization
		g = BatchNormalization()(g, training=True)
		# conditionally add dropout
		if dropout:
			g = Dropout(0.5)(g, training=True)
		# merge with skip connection
		g = Concatenate()([g, skip_in])
		
		if activation_ == "prelu":
			g = PReLU()(g)
		elif activation_ == "lrelu":
			g = LeakyReLU()(g)
		elif activation_ == "tanh":
			g = Activation("tanh")(g)
		else:
			raise NotImplementedError

		return g


	# define the standalone generator model
	def build_pix2pix_generator(self, noisy_input_shape, 
								activation_):	
		# image input
		in_image = Input(shape=noisy_input_shape)
		# encoder model
		e1 = self.define_encoder_block(in_image, 32, activation_, batchnorm=False)
		e2 = self.define_encoder_block(e1, 64, activation_)
		e3 = self.define_encoder_block(e2, 128, activation_)
		e4 = self.define_encoder_block(e3, 256, activation_)
		e5 = self.define_encoder_block(e4, 256, activation_)
		e6 = self.define_encoder_block(e5, 256, activation_)
		e7 = self.define_encoder_block(e6, 256, activation_)
		# bottleneck, no batch norm and relu
		b = Conv1D(512, 11, padding='same', kernel_initializer=init)(e7)
		
		if activation_ == "prelu":
			b = PReLU()(b)
		elif activation_ == "lrelu":
			b = LeakyReLU()(b)
		elif activation_ == "tanh":
			b = Activation("tanh")(b)
		else:
			raise NotImplementedError

		# decoder model
		d1 = self.decoder_block(b, e7, 256, activation_, dropout=False)
		d2 = self.decoder_block(d1, e6, 256, activation_, dropout=False)
		d3 = self.decoder_block(d2, e5, 256, activation_, dropout=False)
		d4 = self.decoder_block(d3, e4, 256, activation_, dropout=False)
		d5 = self.decoder_block(d4, e3, 128, activation_, dropout=False)
		d6 = self.decoder_block(d5, e2, 64, activation_, dropout=False)
		d7 = self.decoder_block(d6, e1, 32, activation_, dropout=False)

		# output
		g = Conv1DTranspose(1, 11, padding='same', kernel_initializer=init)(d7)
		out_image = Activation('tanh')(g)
		# define model
		model = Model(in_image, out_image)
		return model 