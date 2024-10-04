import tensorflow.keras as keras
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.onsd.load_data import load_onsd_videos
from sparse_coding_torch.utils import SubsetWeightedRandomSampler, get_sample_weights
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, normalize_weights, normalize_weights_3d
from sparse_coding_torch.onsd.classifier_model import ONSDMLP
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
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import normalize

from scikeras.wrappers import KerasClassifier, KerasRegressor

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--kernel_height', default=10, type=int)
    parser.add_argument('--kernel_width', default=150, type=int)
    parser.add_argument('--kernel_depth', default=1, type=int)
    parser.add_argument('--num_kernels', default=10, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=300, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--splits', default=None, type=str, help='k_fold or leave_one_out or all_train or custom')
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--dataset', default='onsd', type=str)
    parser.add_argument('--crop_height', type=int, default=100)
    parser.add_argument('--crop_width', type=int, default=300)
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--clip_depth', type=int, default=1)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--regression', action='store_true')
    
    args = parser.parse_args()
    
    crop_height = args.crop_height
    crop_width = args.crop_width

    image_height = int(crop_height / args.scale_factor)
    image_width = int(crop_width / args.scale_factor)
    clip_depth = args.clip_depth

    batch_size = args.batch_size
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tf.random.set_seed(args.seed)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
    
    yolo_model = YoloModel(args.dataset)

    all_errors = []
    
    inputs = keras.Input(shape=(clip_depth, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(clip_depth, args.kernel_height, args.kernel_width, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_height=args.kernel_height, kernel_width=args.kernel_width, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=False)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
    recon_model = keras.models.load_model(args.sparse_checkpoint)    
    
    splits, dataset = load_onsd_videos(args.batch_size, input_size=(image_height, image_width), crop_size=(crop_height, crop_width), yolo_model=yolo_model, mode=args.splits, n_splits=args.n_splits, do_regression=args.regression)
    positive_class = 'Positives'
    
#     difficult_vids = split_difficult_vids(dataset.get_difficult_vids(), args.n_splits)

    train_pred_all = None
    train_gt_all = None
    
    test_pred_all = None
    test_gt_all = None
    
    video_pred_all = None
    video_gt_all = None
    
    frame_pred_all = None
    frame_gt_all = None

    i_fold = 0
    for train_idx, test_idx in splits:
        with open(os.path.join(output_dir, 'test_ids.txt'), 'a+') as test_id_out:
            test_id_out.write(str(test_idx) + '\n')
        train_loader = copy.deepcopy(dataset)
        train_loader.set_indicies(train_idx)
        test_loader = copy.deepcopy(dataset)
        if args.splits == 'all_train':
            test_loader.set_indicies(train_idx)
        else:
            test_loader.set_indicies(test_idx)

        train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels(), train_loader.get_widths()))
        test_tf = tf.data.Dataset.from_tensor_slices((test_loader.get_frames(), test_loader.get_labels(), test_loader.get_widths()))
        
        print('{} train videos.'.format(len(train_tf)))
        print('{} positive videos.'.format(len(list(train_tf.filter(lambda features, label, width: label==1)))))
        print('{} negative videos.'.format(len(list(train_tf.filter(lambda features, label, width: label==0)))))
        print('-----------------')
        print('{} test videos.'.format(len(test_tf)))
        print('{} positive videos.'.format(len(list(test_tf.filter(lambda features, label, width: label==1)))))
        print('{} negative videos.'.format(len(list(test_tf.filter(lambda features, label, width: label==0)))))
        
#         clf = LogisticRegression(max_iter=1000)
#         clf = RidgeClassifier(alpha=3.0)
#         clf = MLPClassifier(hidden_layer_sizes=(16,))
        if args.flatten:
            classifier_inputs = keras.Input(shape=(args.num_kernels * ((image_height - args.kernel_height) // args.stride + 1)))
        else:
            classifier_inputs = keras.Input(shape=(args.num_kernels))
        classifier_outputs = ONSDMLP()(classifier_inputs)

        classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)
        if args.regression:
            clf = KerasRegressor(classifier_model, loss='mean_squared_error', optimizer='adam', epochs=200, verbose=False)
        else:
            clf = KerasClassifier(classifier_model, loss='binary_crossentropy', optimizer='adam', epochs=200, verbose=False)
        
#         train_filter_activations = [[] for _ in range(args.num_kernels)]
        train_filter_activations = []

        for images, labels, width in tqdm(train_tf.shuffle(len(train_tf)).batch(batch_size)):
            images = tf.expand_dims(tf.transpose(images, [0, 2, 3, 1]), axis=1)

            activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))])).numpy()
            
            activations = tf.squeeze(activations, axis=1)
            activations = tf.squeeze(activations, axis=2)
            if args.flatten:
                activations = tf.reshape(activations, (-1, activations.shape[1] * activations.shape[2]))
            else:
                activations = tf.math.reduce_sum(activations, axis=1)
            
            for b_idx, act in enumerate(activations):
                if args.regression:
                    train_filter_activations.append((act, width[b_idx]))
                else:
                    train_filter_activations.append((act, labels[b_idx]))
            
#             for b_idx in range(activations.shape[0]):
#                 acts = np.squeeze(activations[b_idx])

#                 for i in range(args.num_kernels):
#                     acts_for_filter = acts[:, i]

#                     act_sum = np.sum(acts_for_filter)

#                     train_filter_activations[i].append((act_sum, float(labels[b_idx])))
                
#         test_filter_activations = [[] for _ in range(args.num_kernels)]
        test_filter_activations = []
                
        for images, labels, width in tqdm(test_tf.batch(args.batch_size)):
            images = tf.expand_dims(tf.transpose(images, [0, 2, 3, 1]), axis=1)

            activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))])).numpy()
            
            activations = tf.squeeze(activations, axis=1)
            activations = tf.squeeze(activations, axis=2)
            if args.flatten:
                activations = tf.reshape(activations, (-1, activations.shape[1] * activations.shape[2]))
            else:
                activations = tf.math.reduce_sum(activations, axis=1)
            
            for b_idx, act in enumerate(activations):
                if args.regression:
                    test_filter_activations.append((act, width[b_idx]))
                else:
                    test_filter_activations.append((act, labels[b_idx]))
            
#             for b_idx in range(activations.shape[0]):
#                 acts = np.squeeze(activations[b_idx])

#                 for i in range(args.num_kernels):
#                     acts_for_filter = acts[:, i]

#                     act_sum = np.sum(acts_for_filter)

#                     test_filter_activations[i].append((act_sum, float(labels[b_idx])))
                
        train_X = []
        train_y = []

#         for i in range(len(train_filter_activations[0])):
#             x = np.array([train_filter_activations[j][i][0] for j in range(args.num_kernels)])
#             label = train_filter_activations[0][i][1]
            
#             train_X.append(x)
#             train_y.append(label)
        for x, label in train_filter_activations:
            train_X.append(x)
            train_y.append(label)

        assert len(train_X) == len(train_y)
        assert len(train_X) == len(train_tf)
        
        test_X = []
        test_y = []

#         for i in range(len(test_filter_activations[0])):
#             x = np.array([test_filter_activations[j][i][0] for j in range(args.num_kernels)])
#             label = test_filter_activations[0][i][1]
            
#             test_X.append(x)
#             test_y.append(label)
            
        for x, label in test_filter_activations:
            test_X.append(x)
            test_y.append(label)

        assert len(test_X) == len(test_y)
        
        train_X = normalize(train_X)
        test_X = normalize(test_X)
        
        clf.fit(train_X, train_y)
        
        train_pred = clf.predict(train_X)
        if train_pred_all is None:
            train_pred_all = train_pred
            train_gt_all = train_y
        else:
            train_pred_all = np.concatenate([train_pred_all, train_pred])
            train_gt_all = np.concatenate([train_gt_all, train_y])

        test_pred = clf.predict(test_X)
        if test_pred_all is None:
            test_pred_all = test_pred
            test_gt_all = test_y
        else:
            test_pred_all = np.concatenate([test_pred_all, test_pred])
            test_gt_all = np.concatenate([test_gt_all, test_y])
            
        if args.splits == 'leave_one_out':
            if args.regression:
                video_gt = np.average(test_y)
                if video_gt >= 100 / dataset.max_width:
                    video_gt = np.array([1])
                else:
                    video_gt = np.array([0])
                
                video_pred = np.array([np.average(test_pred)])
                if video_pred >=  100 / dataset.max_width:
                    video_pred = np.array([1])
                else:
                    video_pred = np.array([0])
            else:
                video_gt = np.array([test_y[0]])
                video_pred = np.array([np.round(np.average(test_pred))])
            
            if video_pred_all is None:
                video_pred_all = video_pred
                video_gt_all = video_gt
            else:
                video_pred_all = np.concatenate([video_pred_all, video_pred])
                video_gt_all = np.concatenate([video_gt_all, video_gt])

        sample_idx = random.sample(list(range(len(test_pred))), min([5, len(test_pred)]))
        frame_pred = np.array([pred for idx, pred in enumerate(test_pred) if idx in sample_idx])
        frame_gt = np.array([pred for idx, pred in enumerate(test_y) if idx in sample_idx])
        
        if frame_pred_all is None:
            frame_pred_all = frame_pred
            frame_gt_all = frame_gt
        else:
            frame_pred_all = np.concatenate([frame_pred_all, frame_pred])
            frame_gt_all = np.concatenate([frame_gt_all, frame_gt])

        if args.regression:
            train_acc = metrics.mean_absolute_error(train_pred, train_y)
            test_acc = metrics.mean_absolute_error(test_pred, test_y)
        else:
            train_acc = metrics.accuracy_score(train_pred, train_y)
            test_acc = metrics.accuracy_score(test_pred, test_y)

        print('i_fold={}, train_acc={:.2f}, test_acc={:.2f}'.format(i_fold, train_acc, test_acc))
        
    print('Final Predictions!')
    
    if args.regression:
        train_accuracy = metrics.mean_absolute_error(train_pred_all, train_gt_all)
        test_accuracy = metrics.mean_absolute_error(test_pred_all, test_gt_all)
        frame_accuracy = metrics.mean_absolute_error(frame_pred_all, frame_gt_all)
    else:
        train_accuracy = metrics.accuracy_score(train_pred_all, train_gt_all)
        test_accuracy = metrics.accuracy_score(test_pred_all, test_gt_all)
        frame_accuracy = metrics.accuracy_score(frame_pred_all, frame_gt_all)
    
    if args.splits == 'leave_one_out':
        print(video_pred_all)
        print(video_gt_all)
        video_accuracy = metrics.accuracy_score(video_pred_all, video_gt_all)
        
        print('train_acc={:.2f}, test_acc={:.2f}, frame_acc={:.2f}, video_acc={:.2f}'.format(train_accuracy, test_accuracy, frame_accuracy, video_accuracy))
    else:
        print('train_acc={:.2f}, test_acc={:.2f}, frame_acc={:.2f}'.format(train_accuracy, test_accuracy, frame_accuracy))