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
from sparse_coding_torch.onsd.classifier_model import ONSDMLP, ONSDConv
from sparse_coding_torch.onsd.video_loader import get_yolo_region_onsd, get_participants
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_absolute_error
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
import itertools
import csv
import json

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def calculate_onsd_scores_measured(input_videos, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height):
    frame_path = 'sparse_coding_torch/onsd/onsd_good_for_eval'
    
    all_preds = []
    all_gt = []
    fp = []
    fn = []

    for vid_f in tqdm(input_videos):
        split_path = vid_f.split('/')
        frame_path = '/'.join(split_path[:-1])
        label = split_path[-3]
        f = [png_file for png_file in os.listdir(frame_path) if png_file.endswith('.png')][0]
#     for f in tqdm(os.listdir(os.path.join(frame_path, label))):
#         if not f.endswith('.png'):
#             continue
#         print(split_path)
#         print(frame_path)
#         print(label)
#         print(f)
#         raise Exception

        frame = torch.tensor(cv2.imread(os.path.join(frame_path, f))).swapaxes(2, 1).swapaxes(1, 0)
    
#         print(frame.size())

        frame = get_yolo_region_onsd(yolo_model, frame, crop_width, crop_height, False)
        if not frame:
            continue
        
#         print(frame)

        frame = frame[0]
        
#         print(frame)

        frame = transform(frame).to(torch.float32).unsqueeze(3).unsqueeze(1).numpy()

        activations = tf.stop_gradient(sparse_model([frame, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
            
        activations = tf.squeeze(activations, axis=1)
        activations = tf.squeeze(activations, axis=2)
        activations = tf.math.reduce_sum(activations, axis=1)

        pred = classifier_model.predict(activations)

        pred = tf.math.round(pred)

        final_pred = float(pred)

        all_preds.append(final_pred)

        if label == 'Positives':
            all_gt.append(1.0)
            if final_pred == 0.0:
                fn.append(f)
        elif label == 'Negatives':
            all_gt.append(0.0)
            if final_pred == 1.0:
                fp.append(f)
            
    return np.array(all_preds), np.array(all_gt), fn, fp

def calculate_onsd_scores(input_videos, labels, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height, max_width, flatten, do_regression, activations_2d, use_valid, valid_vids):
    all_predictions = []
    
    numerical_labels = []
    for label in labels:
        if label == 'Positives':
            numerical_labels.append(1.0)
        else:
            numerical_labels.append(0.0)

    final_list = []
    fp_ids = []
    fn_ids = []
    for v_idx, f in tqdm(enumerate(input_videos)):
        if use_valid and not get_participants([f])[0] in valid_vids:
            continue
        
        vc = torchvision.io.read_video(f)[0].permute(3, 0, 1, 2)
        
        all_classes = []
        all_widths = []
        
        all_frames = [vc[:, i, :, :] for i in range(0, vc.size(1), 20)]
        
        all_yolo = [get_yolo_region_onsd(yolo_model, frame, crop_width, crop_height, False) for frame in all_frames]
        
        all_yolo = list(itertools.chain.from_iterable([y for y in all_yolo if y is not None]))
        
#         all_yolo = [yolo[0] for yolo in all_yolo if yolo is not None]
        
        for i in range(0, len(all_yolo), 32):
            batch = torch.stack(all_yolo[i:i+32])
            
            batch = transform(batch).to(torch.float32).unsqueeze(4).numpy()
            
            activations = tf.stop_gradient(sparse_model([batch, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
            
            activations = tf.squeeze(activations, axis=1)
            activations = tf.squeeze(activations, axis=2)
            if flatten:
                activations = tf.reshape(activations, (-1, activations.shape[1] * activations.shape[2]))
            elif activations_2d:
                activations = tf.expand_dims(activations, axis=3)
            else:
                activations = tf.math.reduce_sum(activations, axis=1)
            
            pred = classifier_model.predict(activations)

#             if not do_regression:
#                 pred = tf.math.round(pred)
#             width_pred = tf.math.round(width_pred * max_width)
            
            all_classes.append(pred)
            
        if do_regression:
            final_pred = np.average(np.concatenate(all_classes))
#             raise Exception
#             print(all_classes)
#             print(final_pred)
#             print(max_width)
#             print(100/max_width)
#             raise Exception
            if final_pred >= 100:
                final_pred = np.array([1])
            else:
                final_pred = np.array([0])
        else:
            final_pred = np.round(np.average(np.concatenate(all_classes)))
#         print(all_widths)
#         average_width = np.average(np.array(all_widths))
#         print(average_width)
#         if average_width > 5.0:
#             final_pred = 1
#         else:
#             final_pred = 0
            
        if final_pred != numerical_labels[v_idx]:
            if final_pred == 0:
                fn_ids.append(f)
            else:
                fp_ids.append(f)
            
        final_list.append(final_pred)
        
    return np.array(final_list), np.array(numerical_labels), fn_ids, fp_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=200, type=int)
    parser.add_argument('--kernel_width', default=150, type=int)
    parser.add_argument('--kernel_height', default=10, type=int)
    parser.add_argument('--kernel_depth', default=1, type=int)
    parser.add_argument('--num_kernels', default=16, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=300, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--splits', default=None, type=str, help='k_fold or leave_one_out or all_train or custom')
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--save_train_test_splits', action='store_true')
    parser.add_argument('--balance_classes', action='store_true')
    parser.add_argument('--dataset', default='onsd', type=str)
    parser.add_argument('--crop_height', type=int, default=100)
    parser.add_argument('--crop_width', type=int, default=300)
    parser.add_argument('--scale_factor', type=int, default=2)
    parser.add_argument('--clip_depth', type=int, default=1)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    parser.add_argument('--do_regression', action='store_true')
    parser.add_argument('--flatten', action='store_true')
    parser.add_argument('--activations_2d', action='store_true')
    parser.add_argument('--valid_vids', action='store_true')
    
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
        
    valid_vids = set()
    with open('sparse_coding_torch/onsd/good_frames_onsd.csv', 'r') as valid_in:
        reader = csv.DictReader(valid_in)
        for row in reader:
            vid = row['video'].strip()
            good_frames = row['good_frames'].strip()
            
            if good_frames:
                valid_vids.add(vid)
    
    yolo_model = YoloModel(args.dataset)

    all_errors = []
    
    inputs = keras.Input(shape=(clip_depth, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(clip_depth, args.kernel_height, args.kernel_width, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_height=args.kernel_height, kernel_width=args.kernel_width, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=False)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
    recon_model = keras.models.load_model(args.sparse_checkpoint)
    
    data_augmentation = keras.Sequential([
        keras.layers.Resizing(image_height, image_width)
#         keras.layers.RandomFlip('horizontal'),
# #         keras.layers.RandomFlip('vertical'),
#         keras.layers.RandomRotation(5),
#         keras.layers.RandomBrightness(0.1)
    ])
        
    
    splits, dataset = load_onsd_videos(args.batch_size, crop_size=(crop_height, crop_width), yolo_model=yolo_model, mode=args.splits, n_splits=args.n_splits, do_regression=args.do_regression)
    positive_class = 'Positives'
    
    all_video_labels = [f.split('/')[-3] for f in dataset.get_all_videos()]
    print('{} videos with positive labels.'.format(len([lbl for lbl in all_video_labels if lbl == 'Positives'])))
    print('{} videos with negative labels.'.format(len([lbl for lbl in all_video_labels if lbl == 'Negatives'])))
    
#     difficult_vids = split_difficult_vids(dataset.get_difficult_vids(), args.n_splits)

    print('Processing frames...')
    sparse_codes = []
    total_acts = 0
    total_non_zero = 0
    frames = dataset.get_frames()
    for i in tqdm(range(0, len(frames), 32)):
        frame = tf.stack(frames[i:i+32])
        frame = tf.expand_dims(data_augmentation(tf.transpose(frame, [0, 2, 3, 1])), axis=1)

        activations = tf.stop_gradient(sparse_model([frame, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))])).numpy()
        
        total_non_zero += float(tf.math.count_nonzero(activations))
        total_acts += float(tf.math.reduce_prod(tf.shape(activations)))

        activations = tf.squeeze(activations, axis=1)
        activations = tf.squeeze(activations, axis=2)

        if args.flatten:
            activations = tf.reshape(activations, (-1, activations.shape[1] * activations.shape[2]))
        elif args.activations_2d:
            activations = tf.expand_dims(activations, axis=3)
        else:
            activations = tf.math.reduce_sum(activations, axis=1)
        
        for act in activations:
            sparse_codes.append(act)

    assert len(sparse_codes) == len(frames)
    print('Average sparsity is: {}'.format(total_non_zero / total_acts))

    video_true = []
    video_pred = []
    video_fn = []
    video_fp = []
    
    train_frame_true = []
    train_frame_pred = []
    
    test_frame_true = []
    test_frame_pred = []
    
#     with open(os.path.join(output_dir, 'test_ids.txt'),'w') as f:
#         pass

    i_fold = 0
    fold_to_videos_map = {}
    for train_idx, test_idx in splits:
#         with open(os.path.join(output_dir, 'test_ids.txt'), 'a+') as test_id_out:
#             test_id_out.write(str(test_idx) + '\n')
        train_loader = copy.deepcopy(dataset)
        train_loader.set_indicies(train_idx)
        test_loader = copy.deepcopy(dataset)
        if args.splits == 'all_train':
            test_loader.set_indicies(train_idx)
        else:
            test_loader.set_indicies(test_idx)
            
        train_sparse_codes = [sc for i, sc in enumerate(sparse_codes) if i in train_idx]
        test_sparse_codes = [sc for i, sc in enumerate(sparse_codes) if i in test_idx]
        
        if args.do_regression:
            train_x = tf.stack(train_sparse_codes)
            test_x = tf.stack(test_sparse_codes)
            
            train_y = tf.stack(train_loader.get_widths())
            test_y = tf.stack(test_loader.get_widths())
        else:
            train_x = tf.stack(train_sparse_codes)
            test_x = tf.stack(test_sparse_codes)
            
            train_y = tf.stack(train_loader.get_labels())
            test_y = tf.stack(test_loader.get_labels())
        
#         print('{} train frames.'.format(len(train_x)))
#         print('{} positive frames.'.format(len(list(train_y.filter(lambda features, label, width: label==1)))))
#         print('{} negative frames.'.format(len(list(train_y.filter(lambda features, label, width: label==0)))))
#         print('-----------------')
#         print('{} test frames.'.format(len(test_tf)))
#         print('{} positive frames.'.format(len(list(test_tf.filter(lambda features, label, width: label==1)))))
#         print('{} negative frames.'.format(len(list(test_tf.filter(lambda features, label, width: label==0)))))
        

#         negative_ds = (
#           train_tf
#             .filter(lambda features, label, width: label==0)
#             .repeat())
#         positive_ds = (
#           train_tf
#             .filter(lambda features, label, width: label==1)
#             .repeat())
        
#         balanced_ds = tf.data.Dataset.sample_from_datasets(
#             [negative_ds, positive_ds], [0.5, 0.5])
        
        if args.checkpoint:
            classifier_model = keras.models.load_model(args.checkpoint)
        else:
            if args.flatten:
                classifier_inputs = keras.Input(shape=(args.num_kernels * ((image_height - args.kernel_height) // args.stride + 1)))
            elif args.activations_2d:
                classifier_inputs = keras.Input(shape=(((image_height - args.kernel_height) // args.stride + 1), args.num_kernels, 1))
            else:
                classifier_inputs = keras.Input(shape=(args.num_kernels))
                
            if args.activations_2d:
                classifier_outputs = ONSDConv(args.do_regression)(classifier_inputs)
            else:
                classifier_outputs = ONSDMLP(args.do_regression)(classifier_inputs)

            classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)
            
        if not args.do_regression:
            criterion = keras.losses.BinaryCrossentropy()
        else:
            criterion = keras.losses.MeanSquaredError()
            
            
        classifier_model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr), loss=criterion)
        
        callbacks = [
            keras.callbacks.ModelCheckpoint(os.path.join(args.output_dir, "model_fold_{}.h5".format(i_fold)), save_best_only=False, save_weights_only=True)
        ]
        
        if args.train:
            classifier_model.fit(train_x, train_y, batch_size=args.batch_size, epochs=args.epochs, verbose=False, callbacks=callbacks)
        else:
            classifier_model.load_weights(os.path.join(args.output_dir, "model_fold_{}.h5".format(i_fold)))

        y_true_train = train_y
        if args.do_regression:
            y_pred_train = classifier_model.predict(train_x)
        else:
            y_pred_train = np.round(classifier_model.predict(train_x))
        
        train_frame_true.append(y_true_train)
        train_frame_pred.append(y_pred_train)
        
        y_true_test = test_y
        if args.do_regression:
            y_pred_test = classifier_model.predict(test_x)
        else:
            y_pred_test = np.round(classifier_model.predict(test_x))
        
        test_frame_true.append(y_true_test)
        test_frame_pred.append(y_pred_test)

        t2 = time.perf_counter()

        if args.do_regression:
            f1 = 0.0
            accuracy = mean_absolute_error(y_true_test, y_pred_test)
            train_accuracy = mean_absolute_error(y_true_train, y_pred_train)
        else:
            f1 = f1_score(y_true_test, y_pred_test, average='macro')
            accuracy = accuracy_score(y_true_test, y_pred_test)

            train_accuracy = accuracy_score(y_true_train, y_pred_train)

#         train_accuracies.append(train_accuracy)
#         test_accuracies.append(accuracy)

        pred_dict = {}
        gt_dict = {}

        t1 = time.perf_counter()
        transform = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(1),
         MinMaxScaler(0, 255),
         torchvision.transforms.Resize((image_height, image_width))
        ])

        test_videos = list(test_loader.get_all_videos())# + [v[1] for v in difficult_vids[i_fold]]
        
        fold_to_videos_map[i_fold] = test_videos

        test_labels = [vid_f.split('/')[-3] for vid_f in test_videos]

        classifier_model.do_dropout = False
        max_width = 0
        if args.do_regression:
            max_width = dataset.max_width
        y_pred, y_true, fn, fp = calculate_onsd_scores(test_videos, test_labels, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height, max_width, args.flatten, args.do_regression, args.activations_2d, args.valid_vids, valid_vids)
#         y_pred, y_true, fn, fp = calculate_onsd_scores_measured(test_videos, yolo_model, classifier_model, sparse_model, recon_model, transform, image_width, image_height)
            
        t2 = time.perf_counter()

        print('i_fold={}, time={:.2f}'.format(i_fold, t2-t1))
        
        if np.size(y_pred):

            y_true = tf.cast(y_true, tf.int32)
            y_pred = tf.cast(y_pred, tf.int32)

            f1 = f1_score(y_true, y_pred, average='macro')
            vid_accuracy = accuracy_score(y_true, y_pred)

            video_fn.extend(fn)
            video_fp.extend(fp)

            video_true.extend(y_true)
            video_pred.extend(y_pred)

            print("Test f1={:.2f}, vid acc={:.2f}, train acc={:.2f}, test acc={:.2f}".format(f1, vid_accuracy, train_accuracy, accuracy))

            print(confusion_matrix(y_true, y_pred))
        
#         plt.clf()
#         plt.figure()
        
#         plt.subplot(211)
#         plt.plot(range(len(train_losses)), train_losses)
#         plt.plot(range(len(test_losses)), test_losses)
        
#         plt.subplot(212)
#         plt.plot(range(len(train_accuracies)), train_accuracies)
#         plt.plot(range(len(test_accuracies)), test_accuracies)
#         plt.savefig(os.path.join(args.output_dir, 'loss_acc_graph_{}.png'.format(i_fold)))
            
        i_fold += 1
        
    fp_fn_file = os.path.join(args.output_dir, 'fp_fn.txt')
    with open(fp_fn_file, 'w+') as in_f:
        in_f.write('FP:\n')
        in_f.write(str(video_fp) + '\n\n')
        in_f.write('FN:\n')
        in_f.write(str(video_fn) + '\n\n')
        
    video_true = np.array(video_true)
    video_pred = np.array(video_pred)
            
    final_f1 = f1_score(video_true, video_pred, average='macro')
    final_acc = accuracy_score(video_true, video_pred)
    final_conf = confusion_matrix(video_true, video_pred)
    
    train_frame_true = np.concatenate(train_frame_true)
    train_frame_pred = np.concatenate(train_frame_pred)
    
    test_frame_true = np.concatenate(test_frame_true)
    test_frame_pred = np.concatenate(test_frame_pred)
    
    if args.do_regression:
        train_frame_acc = mean_absolute_error(train_frame_true, train_frame_pred)
        test_frame_acc = mean_absolute_error(test_frame_true, test_frame_pred)
    else:
        train_frame_acc = accuracy_score(train_frame_true, train_frame_pred)
        test_frame_acc = accuracy_score(test_frame_true, test_frame_pred)
            
    print("Final video accuracy={:.2f}, video f1={:.2f}, frame train accuracy={:.2f}, frame test accuracy={:.2f}".format(final_acc, final_f1, train_frame_acc, test_frame_acc))
    print(final_conf)
    
    with open(os.path.join(args.output_dir, 'fold_to_videos.json'), 'w+') as fold_vid_out:
        json.dump(fold_to_videos_map, fold_vid_out)

