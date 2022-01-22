# example of training a cyclegan on the horse2zebra dataset
import argparse
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import random
from numpy import load
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from matplotlib import pyplot


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)


def define_discriminator_test(clean_input_shape,
							n_filters=[64, 128, 256, 512, 512],
							kernel_size=(1,4),
							strides=(1,4)):

	weight_init = RandomNormal(mean=0., stddev=0.02)
	clean_input = Input(shape=clean_input_shape)

	x = clean_input

	# convolution layers
	for i in range(len(n_filters)):
		x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
					strides=(1,4), padding='same', use_bias=True,
					kernel_initializer=weight_init)(x)
		x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
		x = LeakyReLU()(x)

	x = Reshape((512, ))(x)

	# dense layers
	x = Dense(256, activation=None, use_bias=True)(x)
	x = LeakyReLU()(x)
	x = Dense(128, activation=None, use_bias=True)(x)
	x = LeakyReLU()(x)
	x = Dense(1, activation=None, use_bias=True)(x)

	# create model graph
	model = Model(inputs=clean_input, outputs=x)
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])

	print("\nDiscriminator")
	model.summary()
	return model


def define_generator_test(noisy_input_shape,
						n_filters=[64, 128, 256, 512],
						kernel_size=(4,4),
						strides=[(1,2),(1,2),(1,5),(1,5)],
						use_upsampling=False):

	weight_init = RandomNormal(mean=0., stddev=0.02)
	noisy_input = Input(shape=noisy_input_shape)
	#z_input = Input(shape=z_input_shape)

	# skip connections
	skip_connections = []

	# encode
	x = noisy_input
	for i in range(len(n_filters)):
		x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
					strides=strides[i], padding='same', use_bias=True,
					kernel_initializer=weight_init)(x)
		x = LeakyReLU()(x)
		skip_connections.append(x)

	# prepend single channel filter and remove the last filter size
	n_filters = [1] + n_filters[:-1]

	# update current x input
	#x = z_input

	# decode
	for i in range(len(n_filters)-1, -1, -1):
		x = Concatenate(3)([x, skip_connections[i]])
		if use_upsampling:
			x = UpSampling2D(size=(1, 4))(x)
			x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
						strides=(1, 1), padding='same',
						kernel_initializer=weight_init, use_bias=True)(x)
		else:
			x = Conv2DTranspose(filters=n_filters[i], kernel_size=kernel_size,
								strides=strides[i], padding='same',
								kernel_initializer=weight_init)(x)

		x = LeakyReLU()(x) if i > 0 else Activation("tanh")(x)

	# create model graph
	model = Model(inputs=noisy_input, outputs=x)

	print("\nGenerator")
	model.summary()
	return model



# define a composite model for updating generators by adversarial and cycle loss
def define_composite_model(g_model_1, d_model, g_model_2, image_shape):
	# ensure the model we're updating is trainable
	g_model_1.trainable = True
	# mark discriminator as not trainable
	d_model.trainable = False
	# mark other generator model as not trainable
	g_model_2.trainable = False
	# discriminator element
	input_gen = Input(shape=image_shape)
	gen1_out = g_model_1(input_gen)
	output_d = d_model(gen1_out)
	# identity element
	input_id = Input(shape=image_shape)
	output_id = g_model_1(input_id)
	# forward cycle
	output_f = g_model_2(gen1_out)
	# backward cycle
	gen2_out = g_model_2(input_id)
	output_b = g_model_1(gen2_out)
	# define model graph
	model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
	# define optimization algorithm configuration
	opt = Adam(lr=0.0002, beta_1=0.5)
	# compile model with weighting of least squares loss and L1 loss
	model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
	return model


# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y


# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, dataset):
	# generate fake instance
	X = g_model.predict(dataset)
	# create 'fake' class labels (0)
	y = zeros((len(X), 1))
	return X, y


# save the generator models to file
def save_models(step, g_model_AtoB, g_model_BtoA, folder):
	# save the first generator model
	filename1 = folder+'g_model_AtoB_%06d.h5' % (step+1)
	g_model_AtoB.save(filename1)
	# save the second generator model
	filename2 = folder+'g_model_BtoA_%06d.h5' % (step+1)
	g_model_BtoA.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))


# train cyclegan models
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, folder):
	# define properties of the training run
	n_epochs, n_batch, = 100, 128
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		X_realA, y_realA = generate_real_samples(trainA, n_batch)
		X_realB, y_realB = generate_real_samples(trainB, n_batch)
		# generate a batch of fake samples
		X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB)
		X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA)
		# update generator B->A via adversarial and cycle loss
		g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])
		# update discriminator for A -> [real/fake]
		dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
		dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)
		# update generator A->B via adversarial and cycle loss
		g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])
		# update discriminator for B -> [real/fake]
		dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
		dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		# summarize performance
		print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2))
		# evaluate the model performance every so often
		if (i% bat_per_epo * 10) == 0:
			# save the models
			save_models(i, g_model_AtoB, g_model_BtoA, folder)


def summarize_performance(step, g_model, trainXA, trainXB, trainYB, folder, name):

	snr = SNR()
	
	trainXB_reshape = trainXB.reshape((trainXB.shape[0], trainXB.shape[2], 1))
	print(trainXB_reshape.shape)

	snr_real_b = snr.compute_SNR(trainXB_reshape, np.argmax(trainYB, axis=1))
	
	plt.figure()
	plt.plot(snr_real_b, label="real")
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlabel('Time samples', fontsize=12)
	plt.ylabel('SNR value', fontsize=12)
	plt.legend(prop={"size": 12})

	if name == "AtoB":
		plt.savefig(folder+"snr_real_%06d_gan_AtoB.png" % (step+1))
	elif name == "BtoA":
		plt.savefig(folder+"snr_real_%06d_gan_BtoA.png" % (step+1))
	else:
		raise NotImplementedError

	fake_batch = g_model.predict(trainXA)
	
	fake_batch = fake_batch.reshape((fake_batch.shape[0], fake_batch.shape[2], 1))
	snr_fake_b = snr.compute_SNR(fake_batch, np.argmax(trainYB, axis=1))

	plt.figure()
	plt.plot(snr_fake_b, label="fake")
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlabel('Time samples', fontsize=12)
	plt.ylabel('SNR value', fontsize=12)
	plt.legend(prop={"size": 12})

	if name == "AtoB":
		plt.savefig(folder+"snr_fake_%06d_gan_AtoB.png" % (step+1))
	elif name == "BtoA":
		plt.savefig(folder+"snr_fake_%06d_gan_BtoA.png" % (step+1))
	else:
		raise NotImplementedError


class SNR:
	def __init__(self):
		print("SNR")


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




if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest='command')

	# parser train cycle
	parser_train = subparsers.add_parser('train', help='train Cycle')
	parser_train.add_argument('-load_data_noisy', type=str, dest='load_data_noisy', help='data noisy used for the profiling')
	parser_train.add_argument('-load_data_clean', type=str, dest='load_data_clean', help='data clean used for the profiling')
	parser_train.add_argument('-save_model', type=str, dest='save_model', help='folder to save models')
	parser_train.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")

	# parser snr 
	parser_snr = subparsers.add_parser('snr', help='snr')
	parser_snr.add_argument('-load_data_noisy', type=str, dest='load_data_noisy', help='data noisy used for the profiling')
	parser_snr.add_argument('-load_data_clean', type=str, dest='load_data_clean', help='data clean used for the profiling')
	parser_snr.add_argument('-step', type=int, dest='step', help='step used to load model')
	parser_snr.add_argument('-load_model', type=str, dest='load_model', help='folder to load models')
	parser_snr.add_argument('-save_samples', type=str, dest='save_samples', help='folder to save samples')


	args = parser.parse_args()
	if len(sys.argv) < 2: 
		parser.print_help()
		sys.exit(1)

	if args.command == 'train':
		folder_data_noisy = args.load_data_noisy
		folder_data_clean = args.load_data_clean
		save_model = args.save_model
		gpu = args.gpu

		image_shape = ((1, 700, 1))
		XA_profiling = np.load(folder_data_noisy+"test_Xprofiling.npy")
		XB_profiling = np.load(folder_data_clean+"test_Xprofiling.npy")

		XA_profiling = XA_profiling.reshape((XA_profiling.shape[0], 1, XA_profiling.shape[1], 1))
		XB_profiling = XB_profiling.reshape((XB_profiling.shape[0], 1, XB_profiling.shape[1], 1))

		dataset = (XA_profiling, XB_profiling)

		# generator: A -> B
		g_model_AtoB = define_generator_test(image_shape)
		# generator: B -> A
		g_model_BtoA = define_generator_test(image_shape)
		# discriminator: A -> [real/fake]
		d_model_A = define_discriminator_test(image_shape)
		# discriminator: B -> [real/fake]
		d_model_B = define_discriminator_test(image_shape)
		# composite: A -> B -> [real/fake, A]
		c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, image_shape)
		# composite: B -> A -> [real/fake, B]
		c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, image_shape)
		# train models
		
		with tf.device("/gpu:"+gpu):
			train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, dataset, save_model)


	elif args.command == "snr":
		folder_data_noisy = args.load_data_noisy
		folder_data_clean = args.load_data_clean
		folder_model = args.load_model
		save_samples = args.save_samples
		step = args.step

		XA_validation = np.load(folder_data_noisy+"test_Xvalidation.npy")
		YA_validation = np.load(folder_data_noisy+"test_Yvalidation.npy")

		XB_validation = np.load(folder_data_clean+"test_Xvalidation.npy")
		YB_validation = np.load(folder_data_clean+"test_Yvalidation.npy")

		XA_validation = XA_validation.reshape((XA_validation.shape[0], 1, XA_validation.shape[1], 1))
		XB_validation = XB_validation.reshape((XB_validation.shape[0], 1, XB_validation.shape[1], 1))

		YA_validation = to_categorical(YA_validation, num_classes=256)
		YB_validation = to_categorical(YB_validation, num_classes=256)

		validation = (XA_validation, YA_validation, XB_validation, YB_validation)

		file_AtoB = folder_model+"g_model_AtoB_%06d.h5" % (step+1)
		file_BtoA = folder_model+"g_model_BtoA_%06d.h5" % (step+1)

		g_model_AtoB = load_model(file_AtoB, custom_objects={'InstanceNormalization':InstanceNormalization})
		g_model_BtoA = load_model(file_BtoA, custom_objects={'InstanceNormalization':InstanceNormalization})

		summarize_performance(step, g_model_AtoB, XA_validation, XB_validation, YA_validation, save_samples, "AtoB")
		summarize_performance(step, g_model_BtoA, XB_validation, XA_validation, YB_validation, save_samples, "BtoA")