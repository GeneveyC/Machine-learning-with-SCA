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
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2DTranspose, AveragePooling1D, Input, Reshape, Dropout, Flatten, BatchNormalization, LeakyReLU, concatenate, Conv1DTranspose
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import activations
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
from numpy import load
from numpy import zeros
from numpy import ones

from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

from tensorflow.keras.initializers import RandomNormal, Ones
#weight_init = RandomNormal(mean=0., stddev=0.02)


#from functools import partial
#tf.config.experimental_run_functions_eagerly(True)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



def build_isegan_discriminator(clean_input_shape, noisy_input_shape,
                              n_filters=[32, 32, 64, 64, 128, 128, 256, 256]):

    clean_input = Input(shape=clean_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    x = concatenate([clean_input, noisy_input])

    # convolution layers
    for i in range(len(n_filters)):
        x = Conv1D(n_filters[i], 31, strides=2, kernel_initializer='glorot_uniform',
            use_bias=True, padding="same")(x)
        x = BatchNormalization(axis=2)(x)
        x = LeakyReLU(alpha=0.3)(x)

    x = Conv1D(1, 1, padding="same", use_bias=True, kernel_initializer='glorot_uniform')(x)
    x = Flatten()(x)
    x = Dense(1, activation='linear')(x)

    # create model graph
    D = Model(inputs=[clean_input, noisy_input], outputs=x, name='Discriminator')

    print("\nDiscriminator")
    D.summary()
    return D



def build_isegan_generator(noisy_input_shape,
                        n_filters_enc=[32, 32, 64, 64, 128, 128, 256, 256]):

    n_filters_dec=n_filters_enc[:-1][::-1] + [1]

    # skip connections
    skip_connections = []

    noisy_input = Input(shape=noisy_input_shape)
    
    # encode
    enc_out = noisy_input

    for layernum, numkernels in enumerate(n_filters_enc):      
        enc_out = Conv1D(numkernels, 31, strides=2, kernel_initializer='glorot_uniform', padding="same", use_bias=True)(enc_out)
        
        if layernum < len(n_filters_enc) - 1:
            skip_connections.append(enc_out)
        enc_out = PReLU(alpha_initializer='zero', weights=None)(enc_out)


    num_enc_layers = len(n_filters_enc)
    z_rows = int(1024/ (2 ** num_enc_layers))
    z_cols = n_filters_enc[-1]

    dec_out = enc_out

    # Now to the decoder part
    nrows = z_rows
    ncols = dec_out.get_shape().as_list()[-1]

    # decode
    for declayernum, decnumkernels in enumerate(n_filters_dec):
        indim = dec_out.get_shape().as_list()
        newshape = (indim[1], 1, indim[2])
        dec_out = Reshape(newshape)(dec_out)

        # add the conv2Dtranspose layer
        dec_out = Conv2DTranspose(decnumkernels, [31, 1], strides=[2, 1],
                            kernel_initializer='glorot_uniform', padding='same', use_bias=True)(dec_out)

        nrows *= 2
        ncols = decnumkernels
        dec_out.set_shape([None, nrows, 1, ncols])
        newshape = (nrows, ncols)

        dec_out = Reshape(newshape)(dec_out)

        if declayernum < len(n_filters_dec)-1:
            dec_out = PReLU(alpha_initializer='zero', weights=None)(dec_out)

            skip_ = skip_connections[-(declayernum + 1)]
            dec_out = concatenate([dec_out, skip_])

    dec_out = Activation('tanh')(dec_out)

    # create model graph
    G = Model(inputs=[noisy_input], outputs=[dec_out], name='Generator')

    print("\nGenerator")
    G.summary()
    return G



def mean_absolute_error(y_true, y_pred, denoised_audio, clean_audio):
    return K.mean(K.abs(clean_audio - denoised_audio))


def generate_real_samples(noisy_traces, clean_traces, n_samples):
    # choose random instances
    ix = randint(0, noisy_traces.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = noisy_traces[ix], clean_traces[ix]
    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1), dtype=np.float32)
    return [X1, X2], y


def train(train_noisy_dataset, train_clean_dataset, validation_noisy_dataset, validation_clean_dataset, validation_clean_label, 
        epochs=300, batch_size=200):

    clean_input_shape = (1024, 1)
    noisy_input_shape = (1024, 1)

    D = build_isegan_discriminator(clean_input_shape, noisy_input_shape)
    G = build_isegan_generator(noisy_input_shape)

    clean_input = Input(shape=clean_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    
    denoised_output = G([noisy_input])
    D_fake = D([denoised_output, noisy_input])

    # define D graph and optimizer
    D.compile(optimizer=Adam(0.0002), loss='mean_squared_error')
    G.compile(optimizer=Adam(0.0002), loss='mean_absolute_error')    

    # define D(G(z)) graph and optimizer
    D.trainable = False

    D_of_G = Model(inputs=[clean_input, noisy_input],
                   outputs=[D_fake, denoised_output])

    # reconstruction loss
    #loss_reconstruction = partial(mean_absolute_error,
    #                             denoised_audio=denoised_output,
    #                              clean_audio=clean_input)
    
    #loss_reconstruction.__name__ = "loss_reconstruction"
    #loss_reconstruction.__module__ = mean_absolute_error.__module__
    

    # define Generator's Optimizer
    # D_of_G.compile(RMSprop(lr=lr_g), loss=['mse', loss_reconstruction],
    #                loss_weights=[1, reconstruction_weight])
    D_of_G.compile(Adam(0.0002),
                loss={'Generator': 'mean_absolute_error', 'Discriminator': 'mean_squared_error'},
                loss_weights = {'Generator' : 100., 'Discriminator': 1})


    ones = np.ones((batch_size, 1), dtype=np.float32)
    zeros = np.zeros((batch_size, 1), dtype=np.float32)
    dummy = np.zeros((batch_size, 1), dtype=np.float32)

    batch_per_epochs = int(train_noisy_dataset.shape[0] / batch_size)
    n_iterations = batch_per_epochs * epochs

    for i in range(n_iterations):
        D.trainable = True
        G.trainable = False

        #z = np.random.normal(0, 1, size=(batch_size, ) + z_input_shape)
        #data_batch, cur_batch = next(data_iterator)
        #clean_batch = data_batch[:, 0]
        #noisy_batch = data_batch[:, 1]

        [noisy_batch, clean_batch], _ = generate_real_samples(train_noisy_dataset, train_clean_dataset, batch_size)

        fake_batch = G.predict([noisy_batch])

        loss_real = D.train_on_batch([clean_batch, noisy_batch], ones)
        loss_fake = D.train_on_batch([fake_batch, noisy_batch], zeros)
        loss_d = [loss_real + loss_fake, loss_real, loss_fake]

        D.trainable = False
        G.trainable = True

        [noisy_batch, clean_batch], _ = generate_real_samples(train_noisy_dataset, train_clean_dataset, batch_size)
        
        loss_g = D_of_G.train_on_batch([clean_batch, noisy_batch],
                                       [ones, clean_batch])

        # summarize performance
        print('>%d, g1[%3f], g2[%3f]' % (i+1, loss_g[1], loss_g[2]))

        if (i % batch_per_epochs) == 0:
        	summarize_performance(i, G, validation_noisy_dataset, validation_clean_dataset, validation_clean_label)


def check_file_exists(file_path):
	if os.path.exists(file_path) == False:
			print("Error: provided file path '%s' does not exist!" % file_path)
			sys.exit(-1)
	return


def load_perso_with_validation_for_kfold(database, channels, ntp=1.0, load_metadata=False):
	check_file_exists(database)
	# Open the ASCAD database HDF5 for reading
	try:
		in_file  = h5py.File(database, "r")
	except:
		print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database)
		sys.exit(-1)

	# Load profiling traces
	if len(channels) == 2:
		X_profiling_A = np.array(in_file["LEARN/traces_"+channels[0]])
		X_profiling_A = X_profiling_A[0:int(ntp*X_profiling_A.shape[0]), :]
		
		X_profiling_B = np.array(in_file["LEARN/traces_"+channels[1]])
		X_profiling_B = X_profiling_B[0:int(ntp*X_profiling_B.shape[0]), :]

		X_attack_A = np.array(in_file["ATTACK/traces_"+channels[0]])
		X_attack_B = np.array(in_file["ATTACK/traces_"+channels[1]])
	else:
		raise NotImpelementedError

	# Load profiling labels
	Y_profiling_A = np.array(in_file['LEARN/labels'])
	Y_profiling_A = Y_profiling_A[0:int(ntp*Y_profiling_A.shape[0])]

	Y_profiling_B = np.array(in_file['LEARN/labels'])
	Y_profiling_B = Y_profiling_B[0:int(ntp*Y_profiling_B.shape[0])]

	Y_attack_A = np.array(in_file["ATTACK/labels"])
	Y_attack_B = np.array(in_file["ATTACK/labels"])

	if load_metadata == False:
		return (X_profiling_A, Y_profiling_A), (X_attack_A, Y_attack_A), (X_profiling_B, Y_profiling_B), (X_attack_B, Y_attack_B)
	else:
		return (X_profiling, Y_profiling), (X_attack, Y_attack), (in_file['LEARN/data'], in_file['ATTACK/data'])



# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, validation_noisy_dataset, validation_clean_dataset, validation_clean_label):

    snr = SNR()

    if step == 0:
        #validation_clean_dataset = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], validation_clean_dataset.shape[2], 1))
        snr_real_b = snr.compute_SNR(validation_clean_dataset, np.argmax(validation_clean_label, axis=1))

        plt.figure()
        plt.plot(snr_real_b, label="real")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Time samples', fontsize=12)
        plt.ylabel('SNR value', fontsize=12)
        plt.legend(prop={"size": 12})
        plt.savefig("./samples_isegan1/snr_real_%06d_gan.png" % (step+1))

    #z_input_shape = (1, 4, 256)
    #z = np.random.normal(0, 1, size=(validation_noisy_dataset.shape[0], ) + z_input_shape)
    fake_batch = g_model.predict([validation_noisy_dataset])

    #fake_batch = fake_batch.reshape((fake_batch.shape[0], fake_batch.shape[2], 1))

    snr_fake_b = snr.compute_SNR(fake_batch, np.argmax(validation_clean_label, axis=1))

    plt.figure()
    plt.plot(snr_fake_b, label="fake")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time samples', fontsize=12)
    plt.ylabel('SNR value', fontsize=12)
    plt.legend(prop={"size": 12})
    plt.savefig("./samples_isegan1/snr_fake_%06d_gan.png" % (step+1))

    filename2 = './models_isegan1/model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s' % (filename2))



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

    # parser load
    parser_load = subparsers.add_parser('train', help='create dataset')
    parser_load.add_argument('-load_data_noisy', type=str, dest='load_data_noisy', help='data noisy used for the profiling')
    parser_load.add_argument('-load_data_clean', type=str, dest='load_data_clean', help='data clean used for the profiling')
    parser_load.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")
    #parser_load.add_argument('-channels', type=str, dest="channels", help="channels A/B used for profiling/attack")

    args = parser.parse_args()
    if len(sys.argv) < 2: 
        parser.print_help()
        sys.exit(1)

    if args.command == 'train':
        #database = args.database
        #channels = args.channels.split('-')

        #(X_A, Y_A), (_, _), (X_B, Y_B), (_, _) = load_perso_with_validation_for_kfold(database, channels, 1.0, False)

        #train_noisy_dataset = X_A[0:int(0.8*X_A.shape[0]), :]
        #validation_noisy_dataset = X_A[int(0.8*X_A.shape[0]):, :]
        #train_clean_dataset = X_B[0:int(0.8*X_B.shape[0]), :]
        #validation_clean_dataset = X_B[int(0.8*X_B.shape[0]):, :]

        #print(train_noisy_dataset.shape)
        #print(train_clean_dataset.shape)

        # Normalize for tanh
        #scaler_noisy = MinMaxScaler(feature_range=(-1, 1))
        #train_noisy_dataset = scaler_noisy.fit_transform(train_noisy_dataset)

        #scaler_clean = MinMaxScaler(feature_range=(-1, 1))
        #train_clean_dataset = scaler_clean.fit_transform(train_clean_dataset)

        #validation_noisy_dataset = scaler_noisy.transform(validation_noisy_dataset)
        #validation_clean_dataset = scaler_clean.transform(validation_clean_dataset)

        #train_clean_label = Y_B[0:int(0.8*Y_B.shape[0])]
        #validation_clean_label = Y_B[int(0.8*Y_B.shape[0]):]

        #print(train_clean_label.shape)
        #print(validation_clean_label.shape)

        folder_data_noisy = args.load_data_noisy
        folder_data_clean = args.load_data_clean
        gpu = args.gpu

        train_noisy_dataset = np.load(folder_data_noisy+'test_Xprofiling.npy')
        validation_noisy_dataset = np.load(folder_data_noisy+'test_Xvalidation.npy')
        
        train_clean_dataset = np.load(folder_data_clean+'test_Xprofiling.npy')
        validation_clean_dataset = np.load(folder_data_clean+'test_Xvalidation.npy')
        validation_clean_label = np.load(folder_data_clean+'test_Yvalidation.npy')

        validation_clean_label = to_categorical(validation_clean_label, num_classes=256)

        print(train_noisy_dataset.shape)
        print(train_clean_dataset.shape)
        
        #train_noisy_dataset = train_noisy_dataset.reshape((train_noisy_dataset.shape[0], 1, train_noisy_dataset.shape[1], 1))
        #train_clean_dataset = train_clean_dataset.reshape((train_clean_dataset.shape[0], 1, train_clean_dataset.shape[1], 1))
        #validation_noisy_dataset = validation_noisy_dataset.reshape((validation_noisy_dataset.shape[0], 1, validation_noisy_dataset.shape[1], 1))
        #validation_clean_dataset = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], 1, validation_clean_dataset.shape[1], 1))

        with tf.device("/gpu:"+gpu):
            train(train_noisy_dataset, train_clean_dataset, 
                validation_noisy_dataset, validation_clean_dataset, validation_clean_label)