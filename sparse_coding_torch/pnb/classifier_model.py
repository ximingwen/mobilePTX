from tensorflow import keras
import numpy as np
import torch
import tensorflow as tf
import cv2
import torchvision as tv
import torch
import torch.nn as nn
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
    
class PNBClassifier(keras.layers.Layer):
    def __init__(self):
        super(PNBClassifier, self).__init__()

#         self.max_pool = keras.layers.MaxPooling2D(pool_size=(8, 8), strides=(2, 2))
        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
#         self.conv_3 = keras.layers.Conv2D(12, kernel_size=4, strides=1, activation='relu', padding='valid')
#         self.conv_4 = keras.layers.Conv2D(16, kernel_size=4, strides=2, activation='relu', padding='valid')

        self.flatten = keras.layers.Flatten()

#         self.dropout = keras.layers.Dropout(0.5)

#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(40, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(20, activation='relu', use_bias=True)
        self.ff_4 = keras.layers.Dense(1)

#     @tf.function
    def call(self, activations):
        x = tf.squeeze(activations, axis=1)
#         x = self.max_pool(x)
#         print(x.shape)
        x = self.conv_1(x)
#         print(x.shape)
        x = self.conv_2(x)
#         print(x.shape)
#         raise Exception
#         x = self.conv_3(x)
#         print(x.shape)
#         x = self.conv_4(x)
#         raise Exception
        x = self.flatten(x)
#         x = self.ff_1(x)
#         x = self.dropout(x)
        x = self.ff_2(x)
#         x = self.dropout(x)
        x = self.ff_3(x)
#         x = self.dropout(x)
        x = self.ff_4(x)

        return x
    
class PNBTemporalClassifier(keras.layers.Layer):
    def __init__(self, sparse_checkpoint):
        super(PNBTemporalClassifier, self).__init__()
#         self.conv_1 = keras.layers.Conv3D(24, kernel_size=(1, 250, 50), strides=(1, 1, 10), activation='relu', padding='valid')
#         self.conv_2 = keras.layers.Conv2D(36, kernel_size=(5, 10), strides=(1, 5), activation='relu', padding='valid')
#         self.conv_3 = keras.layers.Conv1D(48, kernel_size=2, strides=2, activation='relu', padding='valid')
        self.padding = keras.layers.ZeroPadding3D((0,7,7))

        initializer = tf.keras.initializers.HeNormal()
    
#         self.sparse_filters = tf.Variable(keras.models.load_model(sparse_checkpoint).weights[0], trainable=False)
        self.sparse_filters = tf.Variable(initial_value=initializer(shape=(1, 15, 15, 1, 32)), trainable=True)

        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(8, 8), strides=(1, 1), activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(48, kernel_size=(8, 8), strides=(1, 1), activation='relu', padding='valid')
        self.conv_3 = keras.layers.Conv2D(48, kernel_size=(8, 8), strides=(2, 1), activation='relu', padding='valid')
        self.conv_4 = keras.layers.Conv2D(48, kernel_size=(8, 8), strides=(2, 2), activation='relu', padding='valid')
        self.conv_5 = keras.layers.Conv2D(48, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='valid')
        self.conv_6 = keras.layers.Conv2D(48, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='valid')
#         self.conv_7 = keras.layers.Conv3D(32, kernel_size=(5, 1, 1), strides=(1, 1, 1), activation='relu', padding='valid')
        
#         self.ff_1 = keras.layers.Dense(250, activation='relu', use_bias=True)
        
#         self.gru = keras.layers.GRU(250)

        self.flatten = keras.layers.Flatten()

#         self.ff_2 = keras.layers.Dense(250, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(100, activation='relu', use_bias=True)
        self.ff_4 = keras.layers.Dense(10, activation='relu', use_bias=True)
        self.ff_5 = keras.layers.Dense(1)

#     @tf.function
    def call(self, clip):
        width = clip.shape[3]
        height = clip.shape[2]
        depth = clip.shape[1]
        
        x = tf.expand_dims(clip, axis=4)

        x = self.padding(x)
        
        x = tf.nn.conv3d(x, self.sparse_filters, strides=(1, 1, 1, 1, 1), padding='VALID')
        x = tf.nn.relu(x)
        
        x = tf.squeeze(x, axis=1)

#         x = tf.reshape(x, (-1, width, height, 1))
        
        x = self.conv_1(x)

#         x = tf.squeeze(x, axis=2)
#         x = tf.reshape(x, (-1, 5, x.shape[2], x.shape[3]))
        x = self.conv_2(x)
#         print(x.shape)
#         raise Exception
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
#         print(x.shape)
#         x = self.conv_7(x)

        x = self.flatten(x)
#         x = self.ff_1(x)

#         x = tf.reshape(x, (-1, 5, 250))
#         x = self.gru(x)
        
#         x = self.ff_2(x)
        x = self.ff_3(x)
        x = self.ff_4(x)
        x = self.ff_5(x)

        return x
    
class MobileModelPNB(keras.Model):
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

        if self.run_2d:
            activations = self.sparse_code(images, [tf.stop_gradient(self.filters_1), tf.stop_gradient(self.filters_2), tf.stop_gradient(self.filters_3), tf.stop_gradient(self.filters_4), tf.stop_gradient(self.filters_5)])
            activations = tf.expand_dims(activations, axis=1)
        else:
            activations = self.sparse_code(images, tf.stop_gradient(self.filters))

        pred = tf.math.round(tf.math.sigmoid(self.classifier(activations)))
#         pred = tf.math.reduce_sum(activations)

        return pred
