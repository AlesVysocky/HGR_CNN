import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
import numpy as np


def build(input_size, depth = 1, filters=(32, 32, 64, 64)):
	# initialize the input shape to be "channels last" along with
	# the channels dimension itself
	# channels dimension itself
	inputShape = (*input_size, depth)

	latentDim = 20

	# define the input to the encoder
	inputs = Input(shape=inputShape)
	x = inputs

	# loop over the number of filters
	for f in filters:
		# apply a CONV => RELU => BN operation
		x = Conv2D(f, (3, 3), padding="same", use_bias=False)(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization()(x)

	# flatten the network and then construct our latent vector
	volumeSize = K.int_shape(x)
	x = Flatten()(x)
	latent = Dense(latentDim)(x)

	# build the encoder model
	encoder = Model(inputs, latent, name="encoder")

	# start building the decoder model which will accept the
	# output of the encoder as its inputs
	latentInputs = Input(shape=(latentDim,))
	x = Dense(np.prod(volumeSize[1:]))(latentInputs)
	x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

	# loop over our number of filters again, but this time in
	# reverse order
	for f in filters[::-1]:
		# apply a CONV_TRANSPOSE => RELU => BN operation
		x = Conv2DTranspose(f, 3, padding="same", use_bias=False, kernel_initializer = 'he_uniform')(x)
		x = LeakyReLU(alpha=0.2)(x)
		x = BatchNormalization()(x)

	# apply a single CONV_TRANSPOSE layer used to recover the
	# original depth of the image
	x = Conv2DTranspose(depth, 3, padding="same", kernel_initializer = 'he_uniform')(x)
	outputs = Activation("sigmoid")(x)

	# build the decoder model
	decoder = Model(latentInputs, outputs, name="decoder")

	# our autoencoder is the encoder + decoder
	autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

	autoencoder.summary()
	tf.keras.utils.plot_model(autoencoder, to_file = "arch.png", expand_nested= True, ) #show_shapes=True

	return autoencoder