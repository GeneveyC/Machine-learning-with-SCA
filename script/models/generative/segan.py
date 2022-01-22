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
from tensorflow.keras.layers import Activation, Dense, Concatenate, Conv1DTranspose, AveragePooling1D, Input, Reshape, Dropout, Flatten, BatchNormalization, LeakyReLU, concatenate
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

from tensorflow.keras.layers import Input, Activation, Concatenate, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, UpSampling2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from tensorflow.keras.initializers import RandomNormal, Ones
weight_init = RandomNormal(mean=0., stddev=0.02)


from functools import partial
tf.config.experimental_run_functions_eagerly(True)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def build_segan_discriminator(noisy_input_shape, clean_input_shape,
                              n_filters=[32, 64, 128, 256],
                              kernel_size=(1, 31)):
    clean_input = Input(shape=clean_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    x = Concatenate(-1)([clean_input, noisy_input])

    # convolution layers
    for i in range(len(n_filters)):
        x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
                   strides=(1, 4), padding='same', use_bias=True,
                   kernel_initializer=weight_init)(x)
        x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
        x = PReLU()(x)

    x = Reshape((512, ))(x)

    # dense layers
    x = Dense(256, activation=None, use_bias=True)(x)
    x = PReLU()(x)
    x = Dense(128, activation=None, use_bias=True)(x)
    x = PReLU()(x)
    x = Dense(1, activation=None, use_bias=True)(x)

    # create model graph
    model = Model(inputs=[noisy_input, clean_input], outputs=x, name='Discriminator')

    print("\nDiscriminator")
    model.summary()
    return model


def build_segan_generator(noisy_input_shape, z_input_shape,
                          n_filters=[32, 64, 128, 256],
                          kernel_size=(1, 31), use_upsampling=False):
    noisy_input = Input(shape=noisy_input_shape)
    z_input = Input(shape=z_input_shape)

    # skip connections
    skip_connections = []

    # encode
    x = noisy_input
    for i in range(len(n_filters)):
        x = Conv2D(filters=n_filters[i], kernel_size=kernel_size,
                   strides=(1, 4), padding='same', use_bias=True,
                   kernel_initializer=weight_init)(x)
        x = PReLU()(x)
        skip_connections.append(x)
        #skip_connections.append(ScaleLayer(n_filters[i])(x))

    # prepend single channel filter and remove the last filter size
    n_filters = [1] + n_filters[:-1]

    # update current x input
    x = z_input

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
                                strides=(1, 4), padding='same',
                                kernel_initializer=weight_init)(x)

        x = PReLU()(x) if i > 0 else Activation("tanh")(x)

    # create model graph
    model = Model(inputs=[noisy_input, z_input], outputs=x, name='Generator')

    print("\nGenerator")
    model.summary()
    return model


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


def train_Generator(train_noisy_dataset, train_clean_dataset, validation_noisy_dataset, validation_clean_dataset, validation_clean_label,
        use_upsampling=False,
        lr_g=5e-5, epochs=300, batch_size=200):

    clean_input_shape = (1, 512, 1)
    noisy_input_shape = (1, 512, 1)
    z_input_shape = (1, 2, 256)

    G = build_segan_generator(noisy_input_shape, z_input_shape, use_upsampling=use_upsampling)

    G.compile(optimizer=RMSprop(lr_g), loss='mae')

    batch_per_epochs = int(train_noisy_dataset.shape[0] / batch_size)
    n_iterations = batch_per_epochs * epochs

    for i in range(n_iterations):

        [noisy_batch, clean_batch], _ = generate_real_samples(train_noisy_dataset, train_clean_dataset, batch_size)
        z = np.random.normal(0, 1, size=(batch_size, ) + z_input_shape)

        loss_g = G.train_on_batch([noisy_batch, z], [clean_batch])
        print('>%d, g[%3f]' % (i+1, loss_g))

        if (i % batch_per_epochs) == 0:
            summarize_performance(i, G, validation_noisy_dataset, validation_clean_dataset, validation_clean_label)



def train_SEGAN(train_noisy_dataset, train_clean_dataset, validation_noisy_dataset, validation_clean_dataset, validation_clean_label, 
        folder_samples, folder_models,
        use_upsampling=False, lr_d=5e-5,
        lr_g=5e-5, epochs=100, batch_size=100,
        reconstruction_weight=100):

    clean_input_shape = (1, 512, 1)
    noisy_input_shape = (1, 512, 1)
    z_input_shape = (1, 2, 256)


    D = build_segan_discriminator(noisy_input_shape, clean_input_shape)
    G = build_segan_generator(noisy_input_shape, z_input_shape,
                              use_upsampling=use_upsampling)

    # define D graph and optimizer
    D.compile(optimizer=RMSprop(lr_d), loss='binary_crossentropy')

    # define D(G(z)) graph and optimizer
    D.trainable = False
    z_input = Input(shape=z_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    clean_input = Input(shape=clean_input_shape)
    denoised_output = G([noisy_input, z_input])
    D_fake = D([noisy_input, denoised_output])
    D_of_G = Model(inputs=[noisy_input, z_input, clean_input],
                   outputs=[D_fake, denoised_output])

    # reconstruction loss
    loss_reconstruction = partial(mean_absolute_error,
                                  denoised_audio=denoised_output,
                                  clean_audio=clean_input)
    loss_reconstruction.__name__ = "loss_reconstruction"
    loss_reconstruction.__module__ = mean_absolute_error.__module__
    

    # define Generator's Optimizer
    # D_of_G.compile(RMSprop(lr=lr_g), loss=['mse', loss_reconstruction],
    #                loss_weights=[1, reconstruction_weight])
    D_of_G.compile(RMSprop(lr=lr_g), loss=['binary_crossentropy', 'mae'],
                   loss_weights=[1, reconstruction_weight])

    ones = np.ones((batch_size, 1), dtype=np.float32)
    zeros = np.zeros((batch_size, 1), dtype=np.float32)
    dummy = np.zeros((batch_size, 1), dtype=np.float32)

    # fix a z vector for training evaluation
    #z_fixed = np.random.normal(0, 1, size=(n_checkpoint_audio,) + z_input_shape)
    #data_batch, cur_batch = next(data_iterator)
    #clean_fixed = data_batch[:n_checkpoint_audio, 0]
    #noisy_fixed = data_batch[:n_checkpoint_audio, 1]
    #log_audio(clean_fixed[:, 0, :, 0], logger, 'clean')
    #log_audio(noisy_fixed[:, 0, :, 0], logger, 'noisy')

    batch_per_epochs = int(train_noisy_dataset.shape[0] / batch_size)
    n_iterations = batch_per_epochs * epochs

    for i in range(n_iterations):
        #if cur_batch == 1:
        #    G.trainable = False
        #    fake_audio = G.predict([noisy_fixed, z_fixed])
            #log_audio(fake_audio[:n_checkpoint_audio, 0, :, 0], logger, 'denoised')
        #    epoch += 1
        D.trainable = True
        G.trainable = False

        z = np.random.normal(0, 1, size=(batch_size, ) + z_input_shape)
        #data_batch, cur_batch = next(data_iterator)
        #clean_batch = data_batch[:, 0]
        #noisy_batch = data_batch[:, 1]

        [noisy_batch, clean_batch], _ = generate_real_samples(train_noisy_dataset, train_clean_dataset, batch_size)

        fake_batch = G.predict([noisy_batch, z])
        #fake_batch = fake_batch.reshape((fake_batch.shape[0], 1, fake_batch.shape[1], 1))

        loss_real = D.train_on_batch([noisy_batch, clean_batch], ones)
        loss_fake = D.train_on_batch([noisy_batch, fake_batch], zeros)
        loss_d = [loss_real + loss_fake, loss_real, loss_fake]

        D.trainable = False
        G.trainable = True

        #data_batch, cur_batch = next(data_iterator)
        #clean_batch = data_batch[:, 0]
        #noisy_batch = data_batch[:, 1]

        [noisy_batch, clean_batch], _ = generate_real_samples(train_noisy_dataset, train_clean_dataset, batch_size)

        z = np.random.normal(0, 1, size=(batch_size, ) + z_input_shape)
        loss_g = D_of_G.train_on_batch([noisy_batch, z, clean_batch],
                                       [ones, clean_batch])

        #fake_audio = G.predict([noisy_fixed, z_fixed])
        #log_losses(loss_d, loss_g, i, logger)
        #print("nxt_batch", cur_batch, "min", fake_audio.min(), "max", fake_audio.max())

        # summarize performance
        #print('>%d, g[%.3f]' % (i+1, loss_g))
        print('>%d, g1[%3f], g2[%3f]' % (i+1, loss_g[1], loss_g[2]))

        if (i % batch_per_epochs) == 0:
            print(reconstruction_weight)
            summarize_performance(i, G, validation_noisy_dataset, validation_clean_dataset, validation_clean_label, folder_samples, folder_models)
        print(reconstruction_weight)


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
def summarize_performance(step, g_model, validation_noisy_dataset, validation_clean_dataset, validation_clean_label, folder_samples, folder_models):

    snr = SNR()

    if step == 0:
        validation_clean_dataset = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], validation_clean_dataset.shape[2], 1))

        snr_real_b = snr.compute_SNR(validation_clean_dataset, np.argmax(validation_clean_label, axis=1))
        plt.figure()
        plt.plot(snr_real_b, label="real")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Time samples', fontsize=12)
        plt.ylabel('SNR value', fontsize=12)
        plt.legend(prop={"size": 12})
        plt.savefig(folder_samples+"snr_real_%06d_gan.png" % (step+1))

    z_input_shape = (1, 2, 256)
    z = np.random.normal(0, 1, size=(validation_noisy_dataset.shape[0], ) + z_input_shape)
    fake_batch = g_model.predict([validation_noisy_dataset, z])
    
    fake_batch = fake_batch.reshape((fake_batch.shape[0], fake_batch.shape[2], 1))

    snr_fake_b = snr.compute_SNR(fake_batch, np.argmax(validation_clean_label, axis=1))

    plt.figure()
    plt.plot(snr_fake_b, label="fake")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time samples', fontsize=12)
    plt.ylabel('SNR value', fontsize=12)
    plt.legend(prop={"size": 12})
    plt.savefig(folder_samples+"snr_fake_%06d_gan.png" % (step+1))

    filename2 = folder_models+'model_%06d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s' % (filename2))


def generate_fake(model_folder, scaler_folder, database, channels):

    check_file_exists(database)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file_load = h5py.File(database, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database)
        sys.exit(-1)

    # Load attack traces
    if len(channels) == 1:
        X_attack = np.array(in_file_load["ATTACK/traces_"+channels[0]])
    else:
        raise NotImpelementedError

    # Load attack labels
    Y_attack = np.array(in_file_load["ATTACK/labels"])

    # Load attack data
    attack_data = in_file_load['ATTACK']['data']

    scaler1 = joblib.load(scaler_folder+'scaler.pkl')
    X_attack = scaler1.transform(X_attack)
    X_attack = X_attack.reshape((X_attack.shape[0], 1, X_attack.shape[1], 1))

    z_input_shape = (1, 2, 256)
    z = np.random.normal(0, 1, size=(X_attack.shape[0], ) + z_input_shape)

    g_model = load_model(model_folder+"model_079201.h5")
    #g_model = load_model(model_folder+"model_039601.h5")
    
    fake_attack_dataset = g_model.predict([X_attack, z])

    fake_attack_dataset = fake_attack_dataset.reshape((fake_attack_dataset.shape[0], fake_attack_dataset.shape[2], 1))
    #fake_attack_dataset = fake_attack_dataset.reshape((fake_attack_dataset.shape[0], fake_attack_dataset.shape[2]))
    #scaler2 = joblib.load('./test2/scaler/512PoI/100k/f4pw/scaler.pkl')
    #fake_attack_dataset = scaler2.inverse_transform(fake_attack_dataset)

    file = h5py.File("fake.h5", "w")
    file.create_dataset('FAKE/data', data=attack_data)
    file.create_dataset('FAKE/traces_'+channels[0], data=fake_attack_dataset)
    file.create_dataset('FAKE/labels', data=Y_attack)
    file.close()


def generate_fake_ascad(model_folder, scaler_folder, database, channels):

    check_file_exists(database)
    # Open the ASCAD database HDF5 for reading
    try:
        in_file_load = h5py.File(database, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." % database)
        sys.exit(-1)

    # Load attack traces
    if len(channels) == 1:
        X_attack = np.array(in_file_load["Attack_traces/traces"])
        X_attack = X_attack[:, 188:700]
    else:
        raise NotImpelementedError

    # Load attack labels
    Y_attack = np.array(in_file_load["Attack_traces/labels"])

    # Load attack data
    attack_data = in_file_load['Attack_traces']['metadata']

    scaler = joblib.load(scaler_folder+'scaler.pkl')
    X_attack = scaler.transform(X_attack)
    X_attack = X_attack.reshape((X_attack.shape[0], 1, X_attack.shape[1], 1))

    z_input_shape = (1, 2, 256)
    z = np.random.normal(0, 1, size=(X_attack.shape[0], ) + z_input_shape)

    g_model = load_model(model_folder+"model_039601.h5")
    fake_attack_dataset = g_model.predict([X_attack, z])

    fake_attack_dataset = fake_attack_dataset.reshape((fake_attack_dataset.shape[0], fake_attack_dataset.shape[2]))
    
    file = h5py.File("fake.h5", "w")
    file.create_dataset('Attack_traces/metadata', data=attack_data)
    file.create_dataset('Attack_traces/traces', data=fake_attack_dataset)
    file.create_dataset('Attack_traces/labels', data=Y_attack)
    file.close()


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

    # parser train segan
    parser_train_segan = subparsers.add_parser('train_SEGAN', help='train SEGAN')
    parser_train_segan.add_argument('-load_data_noisy', type=str, dest='load_data_noisy', help='data noisy used for the profiling')
    parser_train_segan.add_argument('-load_data_clean', type=str, dest='load_data_clean', help='data clean used for the profiling')
    parser_train_segan.add_argument('-save_model', type=str, dest='save_folder_model', help='folder to save the model')
    parser_train_segan.add_argument('-save_sample', type=str, dest='save_folder_sample', help='folder to save the sample')
    parser_train_segan.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")
    
    # parser train generator only
    parser_train_g = subparsers.add_parser('train_G', help='train generator only')
    parser_train_g.add_argument('-load_data_noisy', type=str, dest='load_data_noisy', help='data noisy used for the profiling')
    parser_train_g.add_argument('-load_data_clean', type=str, dest='load_data_clean', help='data clean used for the profiling')
    parser_train_g.add_argument('-gpu', type=str, dest="gpu", help="define what gpu to use")

    # parser generate
    parser_generate = subparsers.add_parser('generate', help='generate fake dataset')
    parser_generate.add_argument('-database', type=str, dest='database', help='database use to take attack traces')
    parser_generate.add_argument('-model', type=str, dest='model', help='generator used to generate fake data')
    parser_generate.add_argument('-channels', type=str, dest='channels', help='channels used to generate fake data (first give to G)')
    parser_generate.add_argument('-scaler', type=str, dest='save_scaler', help='scaler to load')


    args = parser.parse_args()
    if len(sys.argv) < 2: 
        parser.print_help()
        sys.exit(1)


    if args.command == 'train_SEGAN':
        folder_data_noisy = args.load_data_noisy
        folder_data_clean = args.load_data_clean
        folder_model = args.save_folder_model
        folder_sample = args.save_folder_sample
        gpu = args.gpu

        train_noisy_dataset = np.load(folder_data_noisy+'test_Xprofiling.npy')
        validation_noisy_dataset = np.load(folder_data_noisy+'test_Xvalidation.npy')
        
        train_clean_dataset = np.load(folder_data_clean+'test_Xprofiling.npy')
        validation_clean_dataset = np.load(folder_data_clean+'test_Xvalidation.npy')
        validation_clean_label = np.load(folder_data_clean+'test_Yvalidation.npy')

        validation_clean_label = to_categorical(validation_clean_label, num_classes=256)

        print(train_noisy_dataset.shape)
        print(train_clean_dataset.shape)
        
        train_noisy_dataset = train_noisy_dataset.reshape((train_noisy_dataset.shape[0], 1, train_noisy_dataset.shape[1], 1))
        train_clean_dataset = train_clean_dataset.reshape((train_clean_dataset.shape[0], 1, train_clean_dataset.shape[1], 1))
        validation_noisy_dataset = validation_noisy_dataset.reshape((validation_noisy_dataset.shape[0], 1, validation_noisy_dataset.shape[1], 1))
        validation_clean_dataset = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], 1, validation_clean_dataset.shape[1], 1))

        with tf.device("/gpu:"+gpu):
            train_SEGAN(train_noisy_dataset, train_clean_dataset, 
                validation_noisy_dataset, validation_clean_dataset, validation_clean_label,
                folder_sample, folder_model)

    elif args.command == 'train_G':
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
        
        train_noisy_dataset = train_noisy_dataset.reshape((train_noisy_dataset.shape[0], 1, train_noisy_dataset.shape[1], 1))
        train_clean_dataset = train_clean_dataset.reshape((train_clean_dataset.shape[0], 1, train_clean_dataset.shape[1], 1))
        validation_noisy_dataset = validation_noisy_dataset.reshape((validation_noisy_dataset.shape[0], 1, validation_noisy_dataset.shape[1], 1))
        validation_clean_dataset = validation_clean_dataset.reshape((validation_clean_dataset.shape[0], 1, validation_clean_dataset.shape[1], 1))

        with tf.device("/gpu:"+gpu):
            train_Generator(train_noisy_dataset, train_clean_dataset, 
                validation_noisy_dataset, validation_clean_dataset, validation_clean_label)

    elif args.command == 'generate':
        database = args.database
        model = args.model
        arg_channels = args.channels
        channels = arg_channels.split('-')
        scaler = args.save_scaler

        generate_fake(model, scaler, database, channels)
        #generate_fake_ascad(model, scaler, database, channels)