from tensorflow import keras
import numpy as np
import torch
import tensorflow as tf
import cv2
import torchvision as tv
import torch
import torch.nn as nn
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler

def load_pytorch_weights(file_path):
    pytorch_checkpoint = torch.load(file_path, map_location='cpu')
    weight_tensor = pytorch_checkpoint['model_state_dict']['filters'].swapaxes(1,3).swapaxes(2,4).swapaxes(0,4).numpy()

    return weight_tensor

# @tf.function
# def do_recon(filters_1, filters_2, filters_3, filters_4, filters_5, activations, image_height, image_width, stride, padding='VALID'):
def do_recon(filters, activations, image_height, image_width, stride, padding='VALID'):
    batch_size = tf.shape(activations)[0]
    recon = tf.nn.conv2d_transpose(activations, filters, output_shape=(batch_size, image_height, image_width, 1), strides=stride, padding=padding)
#     out_2 = tf.nn.conv2d_transpose(activations, filters_2, output_shape=(batch_size, image_height, image_width, 1), strides=stride, padding=padding)
#     out_3 = tf.nn.conv2d_transpose(activations, filters_3, output_shape=(batch_size, image_height, image_width, 1), strides=stride, padding=padding)
#     out_4 = tf.nn.conv2d_transpose(activations, filters_4, output_shape=(batch_size, image_height, image_width, 1), strides=stride, padding=padding)
#     out_5 = tf.nn.conv2d_transpose(activations, filters_5, output_shape=(batch_size, image_height, image_width, 1), strides=stride, padding=padding)

#     recon = tf.concat([out_1, out_2, out_3, out_4, out_5], axis=3)

    return recon

# @tf.function
def do_recon_3d(filters, activations, image_height, image_width, clip_depth, stride, padding='VALID'):
#     activations = tf.pad(activations, paddings=[[0,0], [2, 2], [0, 0], [0, 0], [0, 0]])
    batch_size = tf.shape(activations)[0]
    if padding == 'SAME':
        activations = tf.pad(activations, paddings=[[0,0],[2,2],[0,0],[0,0],[0,0]])
    recon = tf.nn.conv3d_transpose(activations, filters, output_shape=(batch_size, clip_depth, image_height, image_width, 1), strides=[1, stride, stride], padding=padding)

    return recon

# @tf.function
def conv_error(filters, e, stride, padding='VALID'):
    g = tf.nn.conv2d(e, filters, strides=stride, padding=padding)

    return g

@tf.function
def conv_error_3d(filters, e, stride, padding='VALID'):
#     e = tf.pad(e, paddings=[[0,0], [0, 0], [7, 7], [7, 7], [0, 0]])
    if padding == 'SAME':
        e = tf.pad(e, paddings=[[0,0], [0,0], [7,7], [7,7], [0,0]])
        g = tf.nn.conv3d(e, filters, strides=[1, 1, stride, stride, 1], padding='VALID')
    else:
        g = tf.nn.conv3d(e, filters, strides=[1, 1, stride, stride, 1], padding=padding)

    return g

# @tf.function
def normalize_weights(filters, out_channels):
    #print('filters shape', tf.shape(filters))
    norms = tf.norm(tf.reshape(tf.transpose(tf.stack(filters), perm=[4, 0, 1, 2, 3]), (out_channels, -1)), axis=1)
    norms = tf.broadcast_to(tf.math.maximum(norms, 1e-12*tf.ones_like(norms)), filters[0].shape)

    adjusted = [f / norms for f in filters]

    #raise Exception('Beep')

    return adjusted

@tf.function
def normalize_weights_3d(filters, out_channels):
    #for f in filters:
    #    print('filters 3d shape', f.shape)
    norms = tf.norm(tf.reshape(tf.transpose(filters[0], perm=[4, 0, 1, 2, 3]), (out_channels, -1)), axis=1)
    # tf.print("norms", norms.shape, norms)
    norms = tf.broadcast_to(tf.math.maximum(norms, 1e-12*tf.ones_like(norms)), filters[0].shape)

    adjusted = [f / norms for f in filters]

    #for i in range(out_channels):
    #    tf.print("after normalization", tf.norm(adjusted[0][:,:,:,0,i]))
    #print()

    #raise Exception('Beep')
    return adjusted

class SparseCode(keras.layers.Layer):
    def __init__(self, batch_size, image_height, image_width, clip_depth, in_channels, out_channels, kernel_height, kernel_width, kernel_depth, stride, lam, activation_lr, max_activation_iter, run_2d, padding='VALID'):
        super(SparseCode, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.lam = lam
        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.clip_depth = clip_depth
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.kernel_depth = kernel_depth
        self.run_2d = run_2d
        self.padding = padding

#     @tf.function
    def do_update(self, images, filters, u, m, v, b1, b2, eps, i):
        activations = tf.nn.relu(u - self.lam)

        if self.run_2d:
            recon = do_recon(filters, activations, self.image_height, self.image_width, self.stride, self.padding)
#             recon = do_recon(filters[0], filters[1], filters[2], filters[3], filters[4], activations, self.image_height, self.image_width, self.stride, self.padding)
        else:
            recon = do_recon_3d(filters, activations, self.image_height, self.image_width, self.clip_depth, self.stride, self.padding)

        e = images - recon
        g = -1 * u

        if self.run_2d:
            g += conv_error(filters, e, self.stride, self.padding)
#             e1, e2, e3, e4, e5 = tf.split(e, 5, axis=3)
#             g += conv_error(filters[0], e1, self.stride, self.padding)
#             g += conv_error(filters[1], e2, self.stride, self.padding)
#             g += conv_error(filters[2], e3, self.stride, self.padding)
#             g += conv_error(filters[3], e4, self.stride, self.padding)
#             g += conv_error(filters[4], e5, self.stride, self.padding)
        else:
            convd_error = conv_error_3d(filters, e, self.stride, self.padding)

            g = g + convd_error

        g = g + activations

        m = b1 * m + (1-b1) * g
        
        v = b2 * v + (1-b2) * tf.math.pow(g, 2)
        
        mh = m / (1 - tf.math.pow(b1, (1+i)))
        
        vh = v / (1 - tf.math.pow(b2, (1+i)))
        du = self.activation_lr * mh / (tf.math.sqrt(vh) + eps)
        
        u += du
        
        return u, m, v

#     @tf.function
    def call(self, images, filters):
        filters = tf.squeeze(filters, axis=0)
        if self.padding == 'SAME':
            if self.run_2d:
                output_shape = (len(images), self.image_height // self.stride, self.image_width // self.stride, self.out_channels)
            else:
                output_shape = (len(images), 1, self.image_height // self.stride, self.image_width // self.stride, self.out_channels)
        else:
            if self.run_2d:
                output_shape = (len(images), (self.image_height - self.kernel_height) // self.stride + 1, (self.image_width - self.kernel_width) // self.stride + 1, self.out_channels)
            else:
                output_shape = (len(images), (self.clip_depth - self.kernel_depth) // 1 + 1, (self.image_height - self.kernel_height) // self.stride + 1, (self.image_width - self.kernel_width) // self.stride + 1, self.out_channels)

        u = tf.stop_gradient(tf.zeros(shape=output_shape))
        m = tf.stop_gradient(tf.zeros(shape=output_shape))
        v = tf.stop_gradient(tf.zeros(shape=output_shape))
#         tf.print('activations before:', tf.reduce_sum(u))

        b1 = tf.constant(0.9, dtype='float32')
        b2 = tf.constant(0.99, dtype='float32')
        eps = tf.constant(1e-8, dtype='float32')
#         i = tf.constant(0, dtype='float32')
#         c = lambda images, filters, u, m, v, b1, b2, eps, i: tf.less(i, self.max_activation_iter)
#         images, filters, u, m, v, b1, b2, eps, i = tf.while_loop(c, self.do_update, [images, filters, u, m, v, b1, b2, eps, i])
        for i in range(self.max_activation_iter):
            u, m, v = self.do_update(images, filters, u, m, v, b1, b2, eps, i)

        u = tf.nn.relu(u - self.lam)

#         tf.print('activations after:', tf.reduce_sum(u))

        return u
    
class ReconSparse(keras.Model):
    def __init__(self, batch_size, image_height, image_width, clip_depth, in_channels, out_channels, kernel_height, kernel_width, kernel_depth, stride, lam, activation_lr, max_activation_iter, run_2d, padding='VALID'):
        super().__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.lam = lam
        self.activation_lr = activation_lr
        self.max_activation_iter = max_activation_iter
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.clip_depth = clip_depth
        self.run_2d = run_2d
        self.padding = padding

        initializer = tf.keras.initializers.HeNormal()
        if run_2d:
            self.filters_1 = tf.Variable(initial_value=initializer(shape=(kernel_height, kernel_width, in_channels, out_channels)), dtype='float32', trainable=True)
            self.filters_2 = tf.Variable(initial_value=initializer(shape=(kernel_height, kernel_width, in_channels, out_channels)), dtype='float32', trainable=True)
            self.filters_3 = tf.Variable(initial_value=initializer(shape=(kernel_height, kernel_width, in_channels, out_channels)), dtype='float32', trainable=True)
            self.filters_4 = tf.Variable(initial_value=initializer(shape=(kernel_height, kernel_width, in_channels, out_channels)), dtype='float32', trainable=True)
            self.filters_5 = tf.Variable(initial_value=initializer(shape=(kernel_height, kernel_width, in_channels, out_channels)), dtype='float32', trainable=True)
        else:
#             pytorch_weights = load_pytorch_weights('sparse.pt')
#             self.filters = tf.Variable(initial_value=pytorch_weights, dtype='float32', trainable=False)
            initial_values = initializer(shape=(1, kernel_height, kernel_width, in_channels, out_channels), dtype='float32')
            initial_values = tf.concat([initial_values]*kernel_depth, axis=0)
            self.filters = tf.Variable(initial_value=initial_values, trainable=True)

        if run_2d:
            weights = normalize_weights(self.get_weights(), out_channels)
        else:
            weights = normalize_weights_3d(self.get_weights(), out_channels)
        self.set_weights(weights)

#     @tf.function
    def call(self, activations):
        if self.run_2d:
#             recon = do_recon(self.filters_1, self.filters_2, self.filters_3, self.filters_4, self.filters_5, activations, self.image_height, self.image_width, self.stride, self.padding)
            recon = do_recon(self.filters_1, self.filters_2, self.filters_3, self.filters_4, self.filters_5, activations, self.image_height, self.image_width, self.stride, self.padding)
        else:
            recon = do_recon_3d(self.filters, activations, self.image_height, self.image_width, self.clip_depth, self.stride, self.padding)
            
        return recon