from tensorflow import keras
import numpy as np
import torch
import tensorflow as tf
import cv2
import torchvision as tv
import torch
import torch.nn as nn
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
from sparse_coding_torch.sparse_model import SparseCode
    
class ONSDClassifier(keras.layers.Layer):
    def __init__(self, sparse_checkpoint):
        super(ONSDClassifier, self).__init__()
        
#         self.sparse_filters = tf.squeeze(keras.models.load_model(sparse_checkpoint).weights[0], axis=0)

        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(8, 8), strides=(2), activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
        self.conv_3 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
        self.conv_4 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
        self.conv_5 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
#         self.conv_6 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(12), activation='relu', padding='valid')
#         self.conv_1 = keras.layers.Conv1D(10, kernel_size=3, strides=1, activation='relu', padding='valid')
#         self.conv_2 = keras.layers.Conv1D(10, kernel_size=3, strides=1, activation='relu', padding='valid')

        self.flatten = keras.layers.Flatten()

        self.dropout = keras.layers.Dropout(0.20)
        
        self.ff_dropout = keras.layers.Dropout(0.1)

#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
#         self.ff_2 = keras.layers.Dense(500, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(100, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(20, activation='relu', use_bias=True)
        self.ff_final_1 = keras.layers.Dense(1)
#         self.ff_final_2 = keras.layers.Dense(1)
        self.do_dropout = True

#     @tf.function
    def call(self, activations):
        activations = tf.squeeze(activations, axis=1)
#         activations = tf.transpose(activations, [0, 2, 3, 1])
#         x = tf.nn.conv2d(activations, self.sparse_filters, strides=(1, 4), padding='VALID')
#         x = tf.nn.relu(x)
        x = self.conv_1(activations)
        x = self.conv_2(x)
#         x = self.dropout(x, self.do_dropout)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
#         x = self.conv_6(x)
        x = self.flatten(x)
#         x = self.ff_1(x)
#         x = self.dropout(x)
        x = self.ff_2(x)
        x = self.ff_dropout(x, self.do_dropout)
        x = self.ff_3(x)
#         x = self.dropout(x)
        class_pred = self.ff_final_1(x)
#         width_pred = tf.math.sigmoid(self.ff_final_2(x))

        return class_pred#, width_pred
    
class ONSDConv(keras.layers.Layer):
    def __init__(self, do_regression):
        super(ONSDConv, self).__init__()
        
#         self.ff_dropout = keras.layers.Dropout(0.1)
        self.conv_1 = keras.layers.Conv2D(8, kernel_size=(1, 4), strides=1, activation='relu', padding='valid')
#         self.max_pool = keras.layers.MaxPooling2D(
        self.conv_2 = keras.layers.Conv2D(8, kernel_size=(1, 4), strides=1, activation='relu', padding='valid')

        self.flatten = keras.layers.Flatten()

#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
#         self.ff_2 = keras.layers.Dense(500, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(10, activation='relu', use_bias=True)
#         self.ff_3 = keras.layers.Dense(8, activation='relu', use_bias=True)
        if do_regression:
            self.ff_final_1 = keras.layers.Dense(1)
        else:
            self.ff_final_1 = keras.layers.Dense(1, activation='sigmoid')
        self.do_dropout = True

#     @tf.function
    def call(self, activations):
        print(activations.shape)
        raise Exception
        x = self.conv_1(activations)
        
        x = self.flatten(x)
        
        x = self.ff_2(x)
#         x = self.ff_dropout(x, self.do_dropout)
#         x = self.ff_3(x)
        class_pred = self.ff_final_1(x)

        return class_pred
    
class ONSDMLP(keras.layers.Layer):
    def __init__(self, do_regression):
        super(ONSDMLP, self).__init__()
        
#         self.ff_dropout = keras.layers.Dropout(0.1)

#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
#         self.ff_2 = keras.layers.Dense(500, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(8, activation='relu', use_bias=True)
#         self.ff_3 = keras.layers.Dense(8, activation='relu', use_bias=True)
        if do_regression:
            self.ff_final_1 = keras.layers.Dense(1)
        else:
            self.ff_final_1 = keras.layers.Dense(1, activation='sigmoid')
        self.do_dropout = True

#     @tf.function
    def call(self, activations):
        x = self.ff_2(activations)
#         x = self.ff_dropout(x, self.do_dropout)
#         x = self.ff_3(x)
        class_pred = self.ff_final_1(x)

        return class_pred
    
class ONSDSharpness(keras.Model):
    def __init__(self):
        super().__init__()
#         self.encoder = tf.keras.applications.DenseNet121(include_top=False)
#         self.encoder.trainable = True
        self.conv_1 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.conv_3 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.conv_4 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.conv_5 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.conv_6 = keras.layers.Conv2D(32, kernel_size=2, strides=1, activation='relu', padding='valid')
        
        self.flatten = keras.layers.Flatten()
        
        self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(100, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(1)
        
    @tf.function
    def call(self, images):
#         x = self.encoder(images)
        x = self.conv_1(images)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        
        x = self.flatten(x)
        
        x = self.ff_1(x)
        x = self.ff_2(x)
        x = self.ff_3(x)

        return x
    
class ONSDRegression(keras.Model):
    def __init__(self):
        super().__init__()
#         self.encoder = tf.keras.applications.DenseNet121(include_top=False)
#         self.encoder.trainable = True
        self.conv_1 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
#         self.conv_3 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
#         self.conv_4 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
#         self.conv_5 = keras.layers.Conv2D(32, kernel_size=4, strides=2, activation='relu', padding='valid')
#         self.conv_6 = keras.layers.Conv2D(32, kernel_size=2, strides=1, activation='relu', padding='valid')
        
        self.flatten = keras.layers.Flatten()
        
#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(100, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(1, activation='sigmoid')
        
    @tf.function
    def call(self, images):
#         x = self.encoder(images)
        x = self.conv_1(images)
        x = self.conv_2(x)
#         x = self.conv_3(x)
#         x = self.conv_4(x)
#         x = self.conv_5(x)
#         x = self.conv_6(x)
        
        x = self.flatten(x)
        
#         x = self.ff_1(x)
        x = self.ff_2(x)
        x = self.ff_3(x)

        return x

    
# class MobileModelONSD(keras.Model):
#     def __init__(self, classifier_model):
#         super().__init__()
#         self.classifier = classifier_model

#     @tf.function
#     def call(self, images):
# #         images = tf.squeeze(tf.image.rgb_to_grayscale(images), axis=-1)
#         images = tf.transpose(images, perm=[0, 2, 3, 1])
#         images = images / 255

#         pred = tf.math.sigmoid(self.classifier(images))

#         return pred

class MobileModelONSD(keras.Model):
    def __init__(self, sparse_weights, classifier_model, batch_size, image_height, image_width, clip_depth, out_channels, kernel_size, kernel_depth, stride, lam, activation_lr, max_activation_iter, run_2d):
        super().__init__()
        if run_2d:
            inputs = keras.Input(shape=(image_height, image_width, clip_depth))
        else:
            inputs = keras.Input(shape=(1, image_height, image_width, clip_depth))
        
        if run_2d:
            filter_inputs = keras.Input(shape=(kernel_size, kernel_size, 1, out_channels), dtype='float32')
        else:
            filter_inputs = keras.Input(shape=(1, kernel_size, kernel_size, 1, out_channels), dtype='float32')
        
        output = SparseCode(batch_size=batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=out_channels, kernel_size=kernel_size, kernel_depth=kernel_depth, stride=stride, lam=lam, activation_lr=activation_lr, max_activation_iter=max_activation_iter, run_2d=run_2d)(inputs, filter_inputs)

        self.sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
        self.classifier = classifier_model

        self.out_channels = out_channels
        self.stride = stride
        self.lam = lam
        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter
        self.batch_size = batch_size
        self.run_2d = run_2d
        
        self.sparse_weights = sparse_weights

    @tf.function
    def call(self, images):
#         images = tf.squeeze(tf.image.rgb_to_grayscale(images), axis=-1)
#         images = tf.transpose(images, perm=[0, 2, 3, 1])
        images = images / 255

        activations = tf.stop_gradient(self.sparse_model([images, tf.stop_gradient(self.sparse_weights)]))

        pred = tf.math.sigmoid(self.classifier(tf.expand_dims(activations, axis=1)))
#         pred = tf.math.sigmoid(self.classifier(activations))
#         pred = tf.math.reduce_sum(activations)

        return pred