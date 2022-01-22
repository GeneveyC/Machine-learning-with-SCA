from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation, Dense, Concatenate, Conv1D, Conv1DTranspose, AveragePooling1D, Input, Reshape, Dropout, Flatten, BatchNormalization, LeakyReLU, ReLU, concatenate
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from numpy.random import randint

from tensorflow.keras.layers import Input, Activation, Concatenate, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, UpSampling2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

from tensorflow.keras.initializers import RandomNormal
weight_init = RandomNormal(mean=0., stddev=0.02)


class SEGAN_1D():
    def __init__(self, arch, save_folder_model, save_folder_samples, save_folder_snr, optimizer_, activation_, lambda_1, load_folder_model=""):
        self.arch = arch
        self.save_folder_model = save_folder_model
        self.save_folder_samples = save_folder_samples
        self.save_folder_snr = save_folder_snr

        if self.arch == "default":
            self.clean_input_shape = (700, 1)
            self.noisy_input_shape = ( 700, 1)
            self.z_input_shape = (2, 256)
            
            self.D = build_segan_discriminator(self.noisy_input_shape, self.clean_input_shape, activation_)
            self.G = build_segan_generator(self.noisy_input_shape, self.z_input_shape, activation_)
            
            if optimizer_ == "rmsprop":
                self.D.compile(optimizer=RMSprop(), loss='binary_crossentropy')
            elif optimizer_ == "adam":
                self.D.compile(optimizer=Adam(), loss='binary_crossentropy')
            elif optimizer_ == "sgd":
                self.D.compile(optimizer=SGD(), loss='binary_crossentropy')
            else:
                raise NotImplementedError

            self.D.trainable = False
            z_input = Input(shape=self.z_input_shape)
            noisy_input = Input(shape=self.noisy_input_shape)
            clean_input = Input(shape=self.clean_input_shape)
            denoised_output = self.G([noisy_input, z_input])
            D_fake = self.D([noisy_input, denoised_output])
            self.D_of_G = Model(inputs=[noisy_input, z_input, clean_input], outputs=[D_fake, denoised_output])
            
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
            self.z_input_shape = (2, 256)

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
            z_input = Input(shape=self.z_input_shape)
            noisy_input = Input(shape=self.noisy_input_shape)
            clean_input = Input(shape=self.clean_input_shape)
            denoised_output = self.G([noisy_input, z_input])
            D_fake = self.D([noisy_input, denoised_output])
            self.D_of_G = Model(inputs=[noisy_input, z_input, clean_input], outputs=[D_fake, denoised_output])
        else:
            raise NotImplementedError



def build_segan_discriminator(noisy_input_shape, clean_input_shape,
                            activation_,
                            n_filters=[32, 64, 128, 256],
                            strides=[2,5,5,7],
                            kernel_size=31):

    clean_input = Input(shape=clean_input_shape)
    noisy_input = Input(shape=noisy_input_shape)
    x = Concatenate(-1)([clean_input, noisy_input])

    # convolution layers
    for i in range(len(n_filters)):
        x = Conv1D(filters=n_filters[i], kernel_size=kernel_size,
                   strides=strides[i], padding='same', use_bias=True,
                   kernel_initializer=weight_init)(x)
        x = BatchNormalization(epsilon=1e-5, momentum=0.1)(x)
        
        if activation_ == "prelu":
            x = PReLU()(x)
        elif activation_ == "lrelu":
            x = LeakyReLU()(x)
        elif activation_ == "tanh":
            x = Activation("tanh")(x)
        else:
            raise NotImpelementedError

    x = Reshape((512, ))(x)

    # dense layers
    x = Dense(256, activation=None, use_bias=True)(x)
    if activation_ == "prelu":
        x = PReLU()(x)
    elif activation_ == "lrelu":
        x = LeakyReLU()(x)
    elif activation_ == "tanh":
        x = Activation("tanh")(x)
    else:
        raise NotImpelementedError

    x = Dense(128, activation=None, use_bias=True)(x)
    
    if activation_ == "prelu":
        x = PReLU()(x)
    elif activation_ == "lrelu":
        x = LeakyReLU()(x)
    elif activation_ == "tanh":
        x = Activation("tanh")(x)
    else:
        raise NotImpelementedError

    x = Dense(1, activation='sigmoid', use_bias=True)(x)

    # create model graph
    model = Model(inputs=[noisy_input, clean_input], outputs=x, name='Discriminator')

    print("\nDiscriminator")
    model.summary()
    return model



def build_segan_generator(noisy_input_shape, z_input_shape,
                        activation_,
                        n_filters=[32, 64, 128, 256],
                        kernel_size=31, 
                        strides=(2,5,5,7),
                        use_upsampling=False):
    noisy_input = Input(shape=noisy_input_shape)
    z_input = Input(shape=z_input_shape)

    # skip connections
    skip_connections = []

    # encode
    x = noisy_input
    for i in range(len(n_filters)):
        x = Conv1D(filters=n_filters[i], kernel_size=kernel_size,
                   strides=strides[i], padding='same', use_bias=True,
                   kernel_initializer=weight_init)(x)
        if activation_ == "prelu":
            x = PReLU()(x)
        elif activation_ == "lrelu":
            x = LeakyReLU()(x)
        elif activation_ == "tanh":
            x = Activation("tanh")(x)
        else:
            raise NotImpelementedError

        skip_connections.append(x)
        
    # prepend single channel filter and remove the last filter size
    n_filters = [1] + n_filters[:-1]

    # update current x input
    x = z_input

    # decode
    for i in range(len(n_filters)-1, -1, -1):
        x = Concatenate(2)([x, skip_connections[i]])
        
        x = Conv1DTranspose(filters=n_filters[i], kernel_size=kernel_size,
                            strides=strides[i], padding='same',
                            kernel_initializer=weight_init)(x)

        if i > 0:
            if activation_ == "prelu":
                x = PReLU()(x) 
            elif activation_ == "lrelu":
                x = LeakyReLU()(x)
            elif activation_ == "tanh":
                x = Activation("tanh")(x)
            else:
                raise NotImpelementedError
        else:
            x = Activation("tanh")(x)

        #x = PReLU()(x) if i > 0 else Activation("tanh")(x)

    # create model graph
    model = Model(inputs=[noisy_input, z_input], outputs=x, name='Generator')

    print("\nGenerator")
    model.summary()
    return model