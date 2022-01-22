import os
import h5py
import joblib
import sys
import json
import pickle
import numpy as np
from numpy.random import randint

import datetime
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import load_model

from models.metrics_costume import mean_rank
from models.multichannels.neuralnetwork import NeuralNetwork
from models.transfert.neuralnetwork import NeuralNetwork_transfert_learning
from models.generative.segan_1D import SEGAN_1D
from models.generative.pix2pix_1D import Pix2Pix_1D

from timeit import default_timer as timer


tf.config.experimental_run_functions_eagerly(True)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
	except RuntimeError as e:
		print(e)


class TimingCallback(Callback):
	def __init__(self, logs={}):
		self.logs=[]
	def on_epoch_begin(self, epoch, logs={}):
		self.starttime = timer()
	def on_epoch_end(self, epoch, logs={}):
		self.logs.append(timer()-self.starttime)


def train_classical_model(network, X_profiling, Y_profiling, X_validation, Y_validation, kfold=1, initial_epoch_=0, epochs=100, batch_size=100):

	folder_save_best_loss = network.save_folder+"models/best-loss/"
	folder_save_best_accu = network.save_folder+"models/best-accu/"
	folder_save_best_rank = network.save_folder+"models/best-rank/"

	if not os.path.exists(folder_save_best_loss):
		os.makedirs(folder_save_best_loss)
	if not os.path.exists(folder_save_best_accu):
		os.makedirs(folder_save_best_accu)
	if not os.path.exists(folder_save_best_rank):
		os.makedirs(folder_save_best_rank)

	save_model_based_on_loss = ModelCheckpoint(folder_save_best_loss+"dnn_{epoch:04d}_best-loss.hdf5", monitor="val_loss", mode='min', save_best_only=True)
	save_model_based_on_accu = ModelCheckpoint(folder_save_best_accu+"dnn_{epoch:04d}_best-accu.hdf5", monitor="val_accuracy", mode="max", save_best_only=True)	
	save_model_based_on_rank = ModelCheckpoint(folder_save_best_rank+"dnn_{epoch:04d}_best-rank.hdf5", monitor="val_mean_rank", mode='min', save_best_only=True)
	cb = TimingCallback()

	callbacks=[save_model_based_on_loss, save_model_based_on_accu, save_model_based_on_rank, cb]

	history = network.model.fit(x=X_profiling, 
			y=to_categorical(Y_profiling, num_classes=256), 
			batch_size=batch_size,
			verbose = 1,
			initial_epoch=initial_epoch_,
			epochs=epochs, 
			validation_data=(X_validation, to_categorical(Y_validation, num_classes=256)),
			callbacks=callbacks)

	history.history["time_duration"] = cb.logs
	
	#network.model = load_model(network.save_folder+"dnn_"+str(kfold)+".hdf5", custom_objects={"mean_rank": mean_rank})
	#scores_validation = network.model.evaluate(X_validation, to_categorical(Y_validation, num_classes=256))
	#return history, scores_validation
	return history


def generate_real_samples(noisy_traces, clean_traces, n_samples):
	# choose random instances
	ix = randint(0, noisy_traces.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = noisy_traces[ix], clean_traces[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, 1), dtype=np.float32)
	return [X1, X2], y


def train_gan_model(gan, X_profiling_noisy, X_profiling_clean, X_validation_noisy, X_validation_clean, Y_validation_clean, epochs=100, batch_size=100, latent_z=True):

	ones = np.ones((batch_size, 1), dtype=np.float32)
	zeros = np.zeros((batch_size, 1), dtype=np.float32)
	dummy = np.zeros((batch_size, 1), dtype=np.float32)

	epoch = 0
	batch_per_epochs = int(X_profiling_noisy.shape[0] / batch_size)
	n_iterations = batch_per_epochs * epochs

	for i in range(0, n_iterations):

		gan.D.trainable = True
		gan.G.trainable = False

		if latent_z:
			z = np.random.normal(0, 1, size=(batch_size, ) + gan.z_input_shape)
		[noisy_batch, clean_batch], _ = generate_real_samples(X_profiling_noisy, X_profiling_clean, batch_size)

		if latent_z:
			fake_batch = gan.G.predict([noisy_batch, z])
		else:
			fake_batch = gan.G.predict([noisy_batch])

		loss_real = gan.D.train_on_batch([noisy_batch, clean_batch], ones)
		loss_fake = gan.D.train_on_batch([noisy_batch, fake_batch], zeros)
		loss_d = [loss_real + loss_fake, loss_real, loss_fake]

		gan.D.trainable = False
		gan.G.trainable = True

		[noisy_batch, clean_batch], _ = generate_real_samples(X_profiling_noisy, X_profiling_clean, batch_size)

		if latent_z:
			z = np.random.normal(0, 1, size=(batch_size, ) + gan.z_input_shape)
			loss_g = gan.D_of_G.train_on_batch([noisy_batch, z, clean_batch], [ones, clean_batch])
		else:
			loss_g = gan.D_of_G.train_on_batch([noisy_batch, clean_batch], [ones, clean_batch])

		print('>%d, g1[%3f], g2[%3f]' % (i+1, loss_g[1], loss_g[2]))

		if (i % batch_per_epochs) == 0:
			summarize_performance_gan(epoch, gan, X_validation_noisy, X_validation_clean, Y_validation_clean, latent_z)
			epoch = epoch + 1


# generate samples and save as a plot and save the model
def summarize_performance_gan(epoch, gan, validation_noisy_dataset, validation_clean_dataset, validation_clean_label, latent_z=True):
	g_model = gan.G
	d_model  = gan.D

	folder_samples = gan.save_folder_samples
	folder_models = gan.save_folder_model
	folder_snr = gan.save_folder_snr

	ones = np.ones((validation_noisy_dataset.shape[0], 1), dtype=np.float32)

	if latent_z:
		z_input_shape = gan.z_input_shape

	snr = SNR()

	if epoch == 0:
		snr_real_b = snr.compute_SNR(validation_clean_dataset, np.argmax(validation_clean_label, axis=1))
		plt.figure()
		plt.plot(snr_real_b, label="real")
		plt.xticks(fontsize=12)
		plt.yticks(fontsize=12)
		plt.xlabel('Time samples', fontsize=12)
		plt.ylabel('SNR value', fontsize=12)
		plt.legend(prop={"size": 12})
		plt.savefig(folder_samples+"snr_real_%03d_gan.png" % epoch)

	if latent_z:
		z = np.random.normal(0, 1, size=(validation_noisy_dataset.shape[0], ) + z_input_shape)
		fake_batch = g_model.predict([validation_noisy_dataset, z])
	else:
		fake_batch = g_model.predict([validation_noisy_dataset])

	snr_fake_b = snr.compute_SNR(fake_batch, np.argmax(validation_clean_label, axis=1))

	plt.figure()
	plt.plot(snr_fake_b, label="fake")
	plt.xticks(fontsize=12)
	plt.yticks(fontsize=12)
	plt.xlabel('Time samples', fontsize=12)
	plt.ylabel('SNR value', fontsize=12)
	plt.legend(prop={"size": 12})
	plt.savefig(folder_samples+"snr_fake_%03d_gan.png" % epoch)


	print("Epoch %03d/%03d" % (epoch, 200))

	if epoch == 0:
		np.save(folder_snr+"best_snr_values.npy", snr_fake_b) # save best snr_value
		filename1 = folder_models+"model_best_snr_over_25.h5"
		g_model.save(filename1)

		filename2 = folder_models+"model_best_snr_over_50.h5"
		g_model.save(filename2)

		filename3 = folder_models+"model_best_snr_over_100.h5"
		g_model.save(filename3)
		
		filename4 = folder_models+"model_best_snr_over_200.h5"
		g_model.save(filename4)
	
		print('>Saved: %s' % (filename1))
		print('>Saved: %s' % (filename2))
		print('>Saved: %s' % (filename3))
		print('>Saved: %s' % (filename4))
	else:
		best_snr_fake_b = np.load(folder_snr+"best_snr_values.npy")
		best_snr_values = sum(best_snr_fake_b)
		new_snr_values = sum(snr_fake_b)

		# save the best model based on snr values
		if new_snr_values > best_snr_values:
			if epoch <= 25:
				filename = folder_models+"model_best_snr_over_25.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			if epoch <= 50:
				filename = folder_models+"model_best_snr_over_50.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			if epoch <= 100:
				filename = folder_models+"model_best_snr_over_100.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			if epoch <= 200:
				filename = folder_models+"model_best_snr_over_200.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			np.save(folder_snr+"best_snr_values.npy", snr_fake_b) # save best snr_value


	if epoch == 0:
		g_model.save(folder_models+"model_best_gan_G.h5")
		d_model.save(folder_models+"model_best_gan_D.h5")

		filename1 = folder_models+"model_best_mae_over_25.h5"
		g_model.save(filename1)

		filename2 = folder_models+"model_best_mae_over_50.h5"
		g_model.save(filename2)

		filename3 = folder_models+"model_best_mae_over_100.h5"
		g_model.save(filename3)
		
		filename4 = folder_models+"model_best_mae_over_200.h5"
		g_model.save(filename4)

		print('>Saved: %s' % (filename1))
		print('>Saved: %s' % (filename2))
		print('>Saved: %s' % (filename3))
		print('>Saved: %s' % (filename4))
	else:
		validation_clean_dataset2 = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], validation_clean_dataset.shape[1]))
		fake_batch2 = fake_batch.reshape((fake_batch.shape[0], fake_batch.shape[1]))
		current_mae = mean_absolute_error(validation_clean_dataset2, fake_batch2)

		best_g_model = load_model(folder_models+"model_best_gan_G.h5")
		
		if latent_z:
			best_fake_batch = best_g_model.predict([validation_noisy_dataset, z])
		else:
			best_fake_batch = best_g_model.predict([validation_noisy_dataset])

		best_fake_batch2 = best_fake_batch.reshape((best_fake_batch.shape[0], best_fake_batch.shape[1]))
		best_mae = mean_absolute_error(validation_clean_dataset2, best_fake_batch2)

		# save the best model based on mae loss
		if current_mae < best_mae:
			if epoch <= 25:
				filename = folder_models+"model_best_mae_over_25.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			if epoch <= 50:
				filename = folder_models+"model_best_mae_over_50.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			if epoch <= 100:
				filename = folder_models+"model_best_mae_over_100.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			if epoch <= 200:
				filename = folder_models+"model_best_mae_over_200.h5"
				g_model.save(filename)
				print('>Saved: %s' % (filename))
			g_model.save(folder_models+"model_best_gan_G.h5")
			d_model.save(folder_models+"model_best_gan_D.h5")



class SNR:
	def __init__(self):
		pass


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



#IHM
if __name__ == "__main__":
	
	if len(sys.argv) < 2:
		print('> python src/models/main.py [CONFIG]')
		sys.exit(1)

	config_json = sys.argv[1]
	with open(config_json) as json_file:
		json_data = json.load(json_file)

	if "dataset" in json_data.keys():
		data_dataset = json_data['dataset']
	else:
		print("[Error] The config file need to have a 'dataset' dict!")
		sys.exit(1)

	dataset_name = data_dataset["name"]
	size_frame = data_dataset["size_frame"]
	predictor = data_dataset["predictor"]
	byte = data_dataset["byte"]
	root_folder_data = data_dataset["root_folder_data"]
	normalization = data_dataset["normalization"]

	if "training" in json_data.keys(): # test if training exist
		for method in json_data['training'].keys():
			data_training = json_data['training'][method]

			if method == "multichannel":
				data = data_training["data"]
				root_folder_models = data_training["root_folder_models"]
				nb_kfold = data_training["kfold"]
				ntp = data_training["ntp"]
				gpu = data_training["gpu"]
				arch = data_training["arch"]
				
				epoch_ = data_training["hyperparameters"]["epoch"]
				batch_size_ = data_training["hyperparameters"]["batch_size"]

				for channels in data:
					print(channels)

					# if "*" in channels:
					# 	list_channel = []
					# 	list_label = []
						
					# 	for channel in channels.split("*"):
					# 		load_data = root_folder_data + "data/"+channel+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
					# 		X1 = np.load(load_data+"X_profiling.npy")
					# 		Y1 = np.load(load_data+"Y_profiling.npy")
					# 		list_channel.append(X1)
					# 		list_label.append(Y1)

					# 	X = np.concatenate(list_channel)
					# 	Y = np.concatenate(list_label)

					# 	X = X.reshape((X.shape[0], X.shape[1], 1))
						
					#else: # multichannel
					list_channel = []

					for channel in channels.split('*'):
						load_data = root_folder_data + "data/"+channel+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
						X1 = np.load(load_data+"X_profiling.npy")
						Y = np.load(load_data+"Y_profiling.npy")
						list_channel.append(X1)

					X = np.dstack(list_channel) # make 2D if two channel "*"
				
					print(X.shape)
					print(Y.shape)

					X_profiling = X[0:int(ntp*X.shape[0])]
					Y_profiling = Y[0:int(ntp*X.shape[0])]
					X_validation = X[int(ntp*X.shape[0]):]
					Y_validation = Y[int(ntp*X.shape[0]):]

					nb_channels = X_profiling.shape[2]

					print(nb_channels)

					subfolder = str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels) + "_" + channels
					save_folder_model = root_folder_models + subfolder

		
					if not os.path.exists(save_folder_model):
						os.makedirs(save_folder_model)
						print(save_folder_model+" was created! ")

					for i in range(0, nb_kfold):
						
						nn = NeuralNetwork(nb_channels, arch, save_folder_model+"/")

						# if we want to continue a training 
						if "continue_from_model" in data_training["hyperparameters"].keys():
							continue_from_model = data_training["hyperparameters"]["continue_from_model"]

							load_filename_model = root_folder_models+str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels) + "_" + channels + "/models/" + continue_from_model + "/"
							list_models_load = os.listdir(load_filename_model)
							list_models_load.sort()
							
							print(list_models_load)

							initial_epoch = int(list_models_load[-1].split("_")[1])
							print(list_models_load[-1], initial_epoch)
							nn.model = load_model(load_filename_model+list_models_load[-1], custom_objects={'mean_rank': mean_rank})

							filename_hist = "hist-%04d_%1d.npy" % (initial_epoch, i+1)
						else:
							initial_epoch = 0
							filename_hist = "hist-%04d_%1d.npy" % (0, i+1)

						print(filename_hist)

						with tf.device("/gpu:"+str(gpu)):
							hist = train_classical_model(nn,
																X_profiling,
																Y_profiling,
																X_validation,
																Y_validation,
																kfold=i+1,
																initial_epoch_ = initial_epoch,
																epochs=epoch_,
																batch_size=batch_size_)
							

							np.save(save_folder_model+"/"+filename_hist, hist.history) # save history


			elif method == "transfer":
				
				data = data_training["data"]
			
				root_folder_pretrained = data_training["root_folder_pretrained"]
				root_folder_models = data_training["root_folder_models"]
				
				nb_kfold = data_training["kfold"]
				ntp = data_training["ntp"]
				gpu = data_training["gpu"]
				arch = data_training["arch"]
				metric_load = data_training["metric_load"]
				pretrain_with_epoch = data_training["pretrain_with_epoch"]
				epoch_ = data_training["hyperparameters"]["epoch"]
				batch_size_ = data_training["hyperparameters"]["batch_size"]

				if len(data) != len(pretrain_with_epoch):
					print("Error: the size of 'data' and 'pretrain_with_epoch' is not the same!")
					sys.exit(-1)
				
				for device, pwe in zip(data, pretrain_with_epoch):

					pretrain_on, train_on = device.split("+")		

					load_data = root_folder_data + "data/"+train_on+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
					X = np.load(load_data+"X_profiling.npy")
					Y = np.load(load_data+"Y_profiling.npy")

					print(X.shape)
					print(Y.shape)

					X_profiling = X[0:int(ntp*X.shape[0])]
					Y_profiling = Y[0:int(ntp*X.shape[0])]
					X_validation = X[int(ntp*X.shape[0]):]
					Y_validation = Y[int(ntp*X.shape[0]):]


					if len(X_profiling.shape) == 2:
						X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
						X_validation = X_validation.reshape((X_validation.shape[0], X_validation.shape[1], 1))

					nb_channels = X_profiling.shape[2]

					load_filename_model = root_folder_pretrained+str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels) + "_" + pretrain_on + "/models/" + metric_load + "/" + "dnn_%04d_%s.hdf5" % (pwe, metric_load)

					subfolder = str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels) + "_" + device
					save_folder_model = root_folder_models + subfolder

					if not os.path.exists(save_folder_model):
						os.makedirs(save_folder_model)
						print(save_folder_model+" was created! ")

					for i in range(0, nb_kfold):
						
						print("Pretrained model load: ", load_filename_model)

						nn = NeuralNetwork_transfert_learning(nb_channels, arch, save_folder_model+"/", load_filename_model)

						if "continue_from_model" in data_training["hyperparameters"].keys():
							continue_from_model = data_training["hyperparameters"]["continue_from_model"]

							load_filename_model = root_folder_models+str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels) + "_" + device + "/models/" + continue_from_model + "/"
							list_models_load = os.listdir(load_filename_model)
							list_models_load.sort()

							print(list_models_load)

							initial_epoch = int(list_models_load[-1].split("_")[1])
							print("Transfer model load: ", list_models_load[-1], initial_epoch)
							nn.model = load_model(load_filename_model+list_models_load[-1], custom_objects={'mean_rank': mean_rank})

							filename_hist = "hist-%04d_%1d.npy" % (initial_epoch, i+1)
						else:
							initial_epoch = 0
							filename_hist = "hist-%04d_%1d.npy" % (0, i+1)

						
						with tf.device("/gpu:"+str(gpu)):
							hist = train_classical_model(nn,
														X_profiling,
														Y_profiling,
														X_validation,
														Y_validation,
														kfold=i+1,
														initial_epoch_ = initial_epoch,
														epochs=epoch_,
														batch_size=batch_size_)

							np.save(save_folder_model+"/"+filename_hist, hist.history) # save history

			elif method == "generative":
				data = data_training["data"]
				root_folder_model = data_training["root_folder_models"]
				root_folder_sample = data_training["root_folder_samples"]
				root_folder_snr = data_training["root_folder_snr"]
				nb_kfold = data_training["kfold"]
				ntp = data_training["ntp"]
				gpu = data_training["gpu"]
				arch = data_training["arch"]
				gan_model = data_training["gan_model"]
				hyperparameters = data_training["hyperparameters"]

				# hyperparameters:
				optimizer_ = hyperparameters["optimizer"]
				activation_ = hyperparameters["activation"]
				epoch = hyperparameters["epoch"]
				batch = hyperparameters["batch_size"]
				lambda_1 = hyperparameters["lambda_1"]

				for device in data:
					data_noisy, data_clean = device.split("+")
		
					load_data_noisy = root_folder_data + "data/"+data_noisy+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"
					load_data_clean = root_folder_data + "data/"+data_clean+"/"+predictor+"/"+str(size_frame)+"/"+normalization+"/"

					X1 = np.load(load_data_noisy+"X_profiling.npy")
					Y1 = np.load(load_data_noisy+"Y_profiling.npy")
					X2 = np.load(load_data_clean+"X_profiling.npy")
					Y2 = np.load(load_data_clean+"Y_profiling.npy")

					train_noisy_dataset = X1[0:int(ntp*X1.shape[0])]
					validation_noisy_dataset = X1[int(ntp*X1.shape[0]):]
					train_clean_dataset = X2[0:int(ntp*X2.shape[0])]
					validation_clean_dataset = X2[int(ntp*X2.shape[0]):]

					validation_clean_label = Y2[int(ntp*Y2.shape[0]):]
					validation_clean_label = to_categorical(validation_clean_label, num_classes=256)

					print(train_noisy_dataset.shape)
					print(train_clean_dataset.shape)

					if len(train_noisy_dataset.shape) == 2:
						train_noisy_dataset = train_noisy_dataset.reshape((train_noisy_dataset.shape[0], train_noisy_dataset.shape[1], 1))
						validation_noisy_dataset = validation_noisy_dataset.reshape((validation_noisy_dataset.shape[0], validation_noisy_dataset.shape[1], 1))

					if len(train_clean_dataset.shape) == 2:
						train_clean_dataset = train_clean_dataset.reshape((train_clean_dataset.shape[0], train_clean_dataset.shape[1], 1))
						validation_clean_dataset = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], validation_clean_dataset.shape[1], 1))

					nb_channels_noisy = train_noisy_dataset.shape[-1]
					nb_channels_clean = train_clean_dataset.shape[-1]

					if nb_channels_noisy == nb_channels_clean == 1:
						print('[INFO] Good channels')
					else:
						print('Error: Noisy data or clean data contains more than one channel')
						sys.exit(1)

					subfolder = str(nb_kfold) + "kfold" + "_" + dataset_name.replace('.h5','') + "_" + predictor + "_" + str(nb_channels_clean) + "_" + device
					folder_model = root_folder_model + subfolder + "/"
					folder_sample = root_folder_sample + subfolder + "/"
					folder_snr = root_folder_snr + subfolder + "/"

					if not os.path.exists(folder_model):
						os.makedirs(folder_model)
						print(folder_model+" was created! ")

					if not os.path.exists(folder_sample):
						os.makedirs(folder_sample)
						print(folder_sample+" was created! ")

					if not os.path.exists(folder_snr):
						os.makedirs(folder_snr)
						print(folder_snr+" was created! ")

					if gan_model == "segan":
						if arch == "load":
							filename_gan = data_training["filename_gan"]
							gan = SEGAN_1D(arch, folder_model, folder_sample, folder_snr, optimizer_, activation_, lambda_1, filename_gan)
						else:
							gan = SEGAN_1D(arch, folder_model, folder_sample, folder_snr, optimizer_, activation_, lambda_1)
						lt_z = True
					elif gan_model == "pix2pix":
						if arch == "load":
							filename_gan = data_training["filename_gan"]
							gan = Pix2Pix_1D(arch, folder_model, folder_sample, folder_snr, optimizer_, activation_, lambda_1, filename_gan)
						else:
							gan = Pix2Pix_1D(arch, folder_model, folder_sample, folder_snr, optimizer_, activation_, lambda_1)
						lt_z = False

					with tf.device("/gpu:"+str(gpu)):
						train_gan_model(gan,
										train_noisy_dataset, 
										train_clean_dataset, 
										validation_noisy_dataset, 
										validation_clean_dataset, 
										validation_clean_label,
										epochs=epoch,
										batch_size=batch,
										latent_z=lt_z)
			else:
				raise NotImplementedError
	else:
		print("[Error] The config file need to have a 'training' dict!")
		sys.exit(1)