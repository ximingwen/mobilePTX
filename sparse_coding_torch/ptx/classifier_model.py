from tensorflow import keras
import numpy as np
import torch
import tensorflow as tf
import cv2
import torchvision as tv
import torch
import torch.nn as nn
from sparse_coding_torch.ptx.video_loader import VideoGrayScaler, MinMaxScaler
from sparse_coding_torch.sparse_model import SparseCode

class PTXClassifier(keras.layers.Layer):
    def __init__(self, num_output):
        super(PTXClassifier, self).__init__()

        self.max_pool = keras.layers.MaxPooling2D(pool_size=4, strides=4)
        self.conv_1 = keras.layers.Conv2D(48, kernel_size=8, strides=4, activation='relu', padding='valid')
#         self.conv_2 = keras.layers.Conv2D(24, kernel_size=4, strides=2, activation='relu', padding='valid')

        self.flatten = keras.layers.Flatten()

        self.dropout = keras.layers.Dropout(0.5)

#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
#         self.ff_2 = keras.layers.Dense(500, activation='relu', use_bias=True)
#         self.ff_2 = keras.layers.Dense(20, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(20, activation='relu', use_bias=True)
        self.ff_4 = keras.layers.Dense(num_output)

#     @tf.function
    def call(self, activations):
        activations = tf.squeeze(activations, axis=1)
        x = self.max_pool(activations)
        x = self.conv_1(activations)
#         x = self.conv_2(x)
        x = self.flatten(x)
#         x = self.ff_1(x)
#         x = self.dropout(x)
#         x = self.ff_2(x)
#         x = self.dropout(x)
        x = self.ff_3(x)
        x = self.dropout(x)
        x = self.ff_4(x)

        return x
    
class PTXVAEClassifier(keras.layers.Layer):
    def __init__(self, num_output):
        super(PTXVAEClassifier, self).__init__()

        self.ff_3 = keras.layers.Dense(20, activation='relu', use_bias=True)
        self.ff_4 = keras.layers.Dense(num_output)

#     @tf.function
    def call(self, z):
        x = self.ff_3(z)
        x = self.ff_4(x)

        return x
    
class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
class VAEEncoderPTX(keras.layers.Layer):
    def __init__(self, latent_dim):
        super(VAEEncoderPTX, self).__init__()

        self.conv_1 = keras.layers.Conv3D(24, kernel_size=(5, 16, 16), strides=(1, 4, 4), activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(36, kernel_size=8, strides=2, activation='relu', padding='valid')
        self.conv_3 = keras.layers.Conv2D(48, kernel_size=4, strides=2, activation='relu', padding='valid')

        self.flatten = keras.layers.Flatten()

        self.ff_mean = keras.layers.Dense(latent_dim, activation='relu', use_bias=True)
        self.ff_var = keras.layers.Dense(latent_dim, activation='relu', use_bias=True)
        
        self.sample = Sampling()

#     @tf.function
    def call(self, images):
        x = self.conv_1(images)
        x = tf.squeeze(x, axis=1)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flatten(x)
        z_mean = self.ff_mean(x)
        z_var = self.ff_var(x)
        z = self.sample([z_mean, z_var])
        return z, z_mean, z_var
    
class VAEDecoderPTX(keras.layers.Layer):
    def __init__(self):
        super(VAEDecoderPTX, self).__init__()
        
        self.ff = keras.layers.Dense(9 * 9 * 48, activation='relu', use_bias=True)
        self.reshape = keras.layers.Reshape((9, 9, 48))
        
        self.deconv_1 = keras.layers.Conv2DTranspose(36, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.deconv_2 = keras.layers.Conv2DTranspose(24, kernel_size=8, strides=2, activation='relu', padding='valid')
        self.deconv_3 = keras.layers.Conv3DTranspose(1, kernel_size=(5, 16, 16), strides=(1, 4, 4), activation='relu', padding='valid')
        
        self.padding = keras.layers.ZeroPadding3D((0, 2, 2))

#     @tf.function
    def call(self, images):
        x = self.ff(images)
        x = self.reshape(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = tf.expand_dims(x, axis=1)
        x = self.deconv_3(x)
        x = self.padding(x)
        return x

class MobileModelPTX(keras.Model):
    def __init__(self, sparse_weights, classifier_model, batch_size, image_height, image_width, clip_depth, out_channels, kernel_size, kernel_depth, stride, lam, activation_lr, max_activation_iter, run_2d):
        super().__init__()
        self.sparse_code = SparseCode(batch_size=batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=out_channels, kernel_size=kernel_size, kernel_depth=kernel_depth, stride=stride, lam=lam, activation_lr=activation_lr, max_activation_iter=max_activation_iter, run_2d=run_2d, padding='VALID')
        self.classifier = classifier_model

        self.out_channels = out_channels
        self.stride = stride
        self.lam = lam
        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter
        self.batch_size = batch_size
        self.run_2d = run_2d

        if run_2d:
            weight_list = np.split(sparse_weights, 5, axis=0)
            self.filters_1 = tf.Variable(initial_value=weight_list[0].squeeze(0), dtype='float32', trainable=False)
            self.filters_2 = tf.Variable(initial_value=weight_list[1].squeeze(0), dtype='float32', trainable=False)
            self.filters_3 = tf.Variable(initial_value=weight_list[2].squeeze(0), dtype='float32', trainable=False)
            self.filters_4 = tf.Variable(initial_value=weight_list[3].squeeze(0), dtype='float32', trainable=False)
            self.filters_5 = tf.Variable(initial_value=weight_list[4].squeeze(0), dtype='float32', trainable=False)
        else:
            self.filters = tf.Variable(initial_value=sparse_weights, dtype='float32', trainable=False)

    @tf.function
    def call(self, images):
#         images = tf.squeeze(tf.image.rgb_to_grayscale(images), axis=-1)
        images = tf.transpose(images, perm=[0, 2, 3, 1])
        images = images / 255
        images = (images - 0.2592) / 0.1251

        if self.run_2d:
            activations = self.sparse_code(images, [tf.stop_gradient(self.filters_1), tf.stop_gradient(self.filters_2), tf.stop_gradient(self.filters_3), tf.stop_gradient(self.filters_4), tf.stop_gradient(self.filters_5)])
        else:
            activations = self.sparse_code(images, tf.stop_gradient(self.filters))

        pred = tf.math.sigmoid(self.classifier(tf.expand_dims(activations, axis=1)))

        return pred