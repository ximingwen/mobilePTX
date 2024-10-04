import tensorflow.keras as keras
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.onsd.load_data import load_onsd_videos, load_onsd_regression
from sparse_coding_torch.utils import SubsetWeightedRandomSampler, get_sample_weights
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, normalize_weights, normalize_weights_3d
from sparse_coding_torch.onsd.classifier_model import ONSDRegression
from sparse_coding_torch.onsd.video_loader import get_yolo_region_onsd
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
import pickle
# from sparse_coding_torch.onsd.train_sparse_model import sparse_loss
from yolov4.get_bounding_boxes import YoloModel
import torchvision
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
import glob
import cv2
import copy
from tensorflow_addons.image import gaussian_filter2d

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=1, type=int)
    parser.add_argument('--num_kernels', default=32, type=int)
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--max_activation_iter', default=150, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--sparse_checkpoint', default='sparse_coding_torch/output/onsd_frame_level_32/best_sparse.pt/', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--dataset', default='onsd', type=str)
    
    args = parser.parse_args()

    image_height = 100
    image_width = 100
    clip_depth = 1

    batch_size = args.batch_size
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
    
    yolo_model = YoloModel(args.dataset)
#     yolo_model = None

#     all_errors = []
    
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(5),
        keras.layers.RandomBrightness(0.1)
    ])
    
    inputs = keras.Input(shape=(clip_depth, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(clip_depth, args.kernel_size, args.kernel_size, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=False)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
    recon_model = keras.models.load_model(args.sparse_checkpoint)
        
    
    splits, dataset = load_onsd_regression(args.batch_size, input_size=(image_height, image_width), yolo_model=yolo_model, mode='balanced')
    positive_class = 'Positives'
    
    train_idx, test_idx = list(splits)[0]

    train_loader = copy.deepcopy(dataset)
    train_loader.set_indicies(train_idx)
    test_loader = copy.deepcopy(dataset)
    test_loader.set_indicies(test_idx)

    train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels()))
    test_tf = tf.data.Dataset.from_tensor_slices((test_loader.get_frames(), test_loader.get_labels()))

#     classifier_inputs = keras.Input(shape=(image_height, image_width, 1))
    classifier_inputs = keras.Input(shape=((clip_depth - args.kernel_depth) // 1 + 1, (image_height - args.kernel_size) // args.stride + 1, (image_width - args.kernel_size) // args.stride + 1, args.num_kernels))
    classifier_outputs = ONSDRegression()(classifier_inputs)

    classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)

    prediction_optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    
    mse = tf.keras.losses.MeanSquaredError()

    best_so_far = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        t1 = time.perf_counter()

        for images, widths in tqdm(train_tf.shuffle(len(train_tf)).batch(args.batch_size)):
            images = tf.expand_dims(data_augmentation(tf.transpose(images, [0, 2, 3, 1])), axis=1)
            
            activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

            with tf.GradientTape() as tape:
                pred = classifier_model(activations)
                loss = mse(pred, widths)

            epoch_loss += loss * images.shape[0]

            gradients = tape.gradient(loss, classifier_model.trainable_weights)

            prediction_optimizer.apply_gradients(zip(gradients, classifier_model.trainable_weights))

        t2 = time.perf_counter()

        test_loss = 0.0
        predictions = []
        gt = []

        for images, widths in tqdm(test_tf.batch(args.batch_size)):
            images = tf.expand_dims(tf.transpose(images, [0, 2, 3, 1]), axis=1)
            
            activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

            pred = classifier_model(activations)
            loss = mse(pred, widths)
            
            for p, g in zip(pred, widths):
                predictions.append(p * dataset.max_width)
                gt.append(g * dataset.max_width)

            test_loss += loss

        t2 = time.perf_counter()
        
        test_acc = keras.losses.MeanAbsoluteError()(gt, predictions)

        print('epoch={}, time={:.2f}, train_loss={:.2f}, test_loss={:.2f}, test_mae={:.2f}'.format(epoch, t2-t1, epoch_loss, test_loss, test_acc))
#             print(epoch_loss)
        if epoch_loss <= best_so_far:
            print("found better model")
            # Save model parameters
            classifier_model.save(os.path.join(output_dir, "best_classifier.pt"))
#                     recon_model.save(os.path.join(output_dir, "best_sparse_model_{}.pt".format(i_fold)))
            pickle.dump(prediction_optimizer.get_weights(), open(os.path.join(output_dir, 'optimizer.pt'), 'wb+'))
            best_so_far = epoch_loss