import tensorflow.keras as keras
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.onsd.load_data import load_onsd_frames
from sparse_coding_torch.utils import SubsetWeightedRandomSampler, get_sample_weights
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, normalize_weights, normalize_weights_3d
from sparse_coding_torch.onsd.classifier_model import ONSDSharpness
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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--dataset', default='onsd', type=str)
    
    args = parser.parse_args()
    
    crop_height = 512
    crop_width = 512

    image_height = 512
    image_width = 512

    batch_size = args.batch_size
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))

    all_errors = []
    
    yolo_model = YoloModel(args.dataset)
    
#     data_augmentation = keras.Sequential([
# #         keras.layers.RandomFlip('vertical'),
# #         keras.layers.RandomRotation(10),
# #         keras.layers.RandomBrightness(0.1)
#         keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1))
#     ])
        
    
    splits, dataset = load_onsd_frames(args.batch_size, input_size=(image_height, image_width), mode='balanced', yolo_model=None)
    
    train_idx, test_idx = list(splits)[0]

    train_loader = copy.deepcopy(dataset)
    train_loader.set_indicies(train_idx)
    test_loader = copy.deepcopy(dataset)
    test_loader.set_indicies(test_idx)

    train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels()))
    test_tf = tf.data.Dataset.from_tensor_slices((test_loader.get_frames(), test_loader.get_labels()))

    classifier_inputs = keras.Input(shape=(image_height, image_width, 3))
    classifier_outputs = ONSDSharpness()(classifier_inputs)

    classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)

    prediction_optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    
    criterion = keras.losses.BinaryCrossentropy(from_logits=True, reduction=keras.losses.Reduction.SUM)

    best_so_far = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        t1 = time.perf_counter()

        for images, labels in tqdm(train_tf.shuffle(len(train_tf)).batch(args.batch_size)):
            images = tf.cast(tf.transpose(images, [0, 2, 3, 1]), tf.float32)
#             images = data_augmentation(images)
#             images = tf.keras.applications.densenet.preprocess_input(images)

            with tf.GradientTape() as tape:
                pred = classifier_model(images)
                loss = criterion(labels, pred)

            epoch_loss += loss * images.shape[0]

            gradients = tape.gradient(loss, classifier_model.trainable_weights)

            prediction_optimizer.apply_gradients(zip(gradients, classifier_model.trainable_weights))

        t2 = time.perf_counter()

        test_count = 0
        test_correct = 0

        for images, labels in tqdm(test_tf.batch(args.batch_size)):
            images = tf.keras.applications.densenet.preprocess_input(tf.cast(tf.transpose(images, [0, 2, 3, 1]), tf.float32))

            pred = classifier_model(images)
            
            pred = tf.math.sigmoid(pred)
            
            for p, l in zip(pred, labels):
                if round(float(p)) == float(l):
                    test_correct += 1
                test_count += 1

        t2 = time.perf_counter()


        print('epoch={}, time={:.2f}, train_loss={:.4f}, test_acc={:.2f}'.format(epoch, t2-t1, epoch_loss, test_correct / test_count))
#         print('epoch={}, time={:.2f}, train_loss={:.2f}'.format(epoch, t2-t1, epoch_loss))

#             print(epoch_loss)
        if epoch_loss < best_so_far:
            print("found better model")
            # Save model parameters
            classifier_model.save(os.path.join(output_dir, "best_classifier.pt"))
#                     recon_model.save(os.path.join(output_dir, "best_sparse_model_{}.pt".format(i_fold)))
            pickle.dump(prediction_optimizer.get_weights(), open(os.path.join(output_dir, 'optimizer.pt'), 'wb+'))
            best_so_far = epoch_loss