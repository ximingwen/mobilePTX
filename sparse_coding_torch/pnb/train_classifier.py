import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.pnb.load_data import load_pnb_videos
from sparse_coding_torch.utils import SubsetWeightedRandomSampler, get_sample_weights
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, normalize_weights, normalize_weights_3d
from sparse_coding_torch.pnb.classifier_model import PNBClassifier, PNBTemporalClassifier
from sparse_coding_torch.pnb.video_loader import classify_nerve_is_right, get_needle_bb, get_yolo_regions
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
import pickle
import tensorflow.keras as keras
import tensorflow as tf
from sparse_coding_torch.pnb.train_sparse_model import sparse_loss
from yolov4.get_bounding_boxes import YoloModel
import torchvision
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
import glob
import cv2
from matplotlib import pyplot as plt
import copy
from matplotlib import cm

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# configproto = tf.compat.v1.ConfigProto()
# configproto.gpu_options.polling_inactive_delay_msecs = 5000
# configproto.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=configproto) 
# tf.compat.v1.keras.backend.set_session(sess)

def calculate_pnb_scores(input_videos, labels, yolo_model, sparse_model, recon_model, classifier_model, image_width, image_height, transform):
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
        vc = torchvision.io.read_video(f)[0].permute(3, 0, 1, 2)
        is_right = classify_nerve_is_right(yolo_model, vc)
        needle_bb = get_needle_bb(yolo_model, vc)
        
        all_preds = []
        for j in range(vc.size(1) - 5, vc.size(1) - 45, -5):
            if j-5 < 0:
                break

            vc_sub = vc[:, j-5:j, :, :]
            
            if vc_sub.size(1) < 5:
                continue
            
            clip = get_yolo_regions(yolo_model, vc_sub, is_right, image_width, image_height)
            
            if not clip:
                continue

            clip = clip[0]
            clip = transform(clip).to(torch.float32)
            clip = tf.expand_dims(clip, axis=4) 
            
            if sparse_model is not None:
                activations = tf.stop_gradient(sparse_model([clip, tf.stop_gradient(tf.expand_dims(recon_model.weights[0], axis=0))]))

                pred = tf.math.round(tf.math.sigmoid(classifier_model(activations)))
            else:
                pred = tf.math.round(tf.math.sigmoid(classifier_model(clip)))

            all_preds.append(pred)
                
        if all_preds:
            final_pred = np.round(np.mean(np.array(all_preds)))
        else:
            final_pred = 1.0
            
        if final_pred != numerical_labels[v_idx]:
            if final_pred == 0:
                fn_ids.append(f)
            else:
                fp_ids.append(f)
            
        final_list.append(final_pred)
        
    return np.array(final_list), np.array(numerical_labels), fn_ids, fp_ids

def calculate_pnb_scores_skipped_frames(input_videos, labels, yolo_model, sparse_model, recon_model, classifier_model, frames_to_skip, image_width, image_height, clip_depth, transform):
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
        vc = torchvision.io.read_video(f)[0].permute(3, 0, 1, 2)
        vc = vc[:,-30:-5, :, :]
        is_right = classify_nerve_is_right(yolo_model, vc)
        needle_bb = get_needle_bb(yolo_model, vc)
        
        all_preds = []
        
        clips = []
        
        for j in range(0, vc.size(1) - 1, 5):
            frames = []
            for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                frames.append(vc[:, k, :, :])
            vc_sub = torch.stack(frames, dim=1)

            if vc_sub.size(1) < 5:
                continue

            clip = get_yolo_regions(yolo_model, vc_sub, is_right, image_width, image_height)
            
            clips.append(clip)

        all_preds = []
        for clip in clips:
            clip = clip[0]
            clip = transform(clip).to(torch.float32)
            clip = tf.expand_dims(clip, axis=4) 

            if sparse_model is not None:
                activations = tf.stop_gradient(sparse_model([clip, tf.stop_gradient(tf.expand_dims(recon_model.weights[0], axis=0))]))

                pred = tf.math.round(tf.math.sigmoid(classifier_model(activations)))
            else:
                pred = tf.math.round(tf.math.sigmoid(classifier_model(clip)))
                
            all_preds.append(pred)
                
        if all_preds:
            final_pred = np.round(np.mean(np.array(all_preds)))
        else:
            final_pred = 0
            
        if final_pred != numerical_labels[v_idx]:
            if final_pred == 0:
                fn_ids.append(f)
            else:
                fp_ids.append(f)

        final_list.append(float(final_pred))
        
    return np.array(final_list), np.array(numerical_labels), fn_ids, fp_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=64, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=150, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--splits', default=None, type=str, help='k_fold or leave_one_out or all_train')
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num_positives', default=100, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--save_train_test_splits', action='store_true')
    parser.add_argument('--run_2d', action='store_true')
    parser.add_argument('--balance_classes', action='store_true')
    parser.add_argument('--dataset', default='pnb', type=str)
    parser.add_argument('--train_sparse', action='store_true')
    parser.add_argument('--mixing_ratio', type=float, default=1.0)
    parser.add_argument('--sparse_lr', type=float, default=0.003)
    parser.add_argument('--crop_height', type=int, default=285)
    parser.add_argument('--crop_width', type=int, default=350)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--clip_depth', type=int, default=5)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    
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
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
    
    yolo_model = YoloModel(args.dataset)

    all_errors = []

    sparse_model = None
    recon_model = None
    
    data_augmentation = keras.Sequential([
        keras.layers.Resizing(image_width//args.scale_factor, image_height//args.scale_factor),
#         keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(5),
        keras.layers.RandomBrightness(0.01),
        keras.layers.RandomTranslation(height_factor=0.1, width_factor=0),
        keras.layers.RandomZoom(height_factor=0.1, width_factor=0),
        keras.layers.RandomContrast(0.05)
    ])
        
    splits, dataset = load_pnb_videos(yolo_model, args.batch_size, input_size=(image_height, image_width, clip_depth), crop_size=(crop_height, crop_width, clip_depth), classify_mode=True, balance_classes=args.balance_classes, mode=args.splits, device=None, n_splits=args.n_splits, sparse_model=None, frames_to_skip=args.frames_to_skip)
    positive_class = 'Positives'

    overall_true = []
    overall_pred = []
    fn_ids = []
    fp_ids = []
    
    i_fold = 0
    for train_idx, test_idx in splits:
        train_loader = copy.deepcopy(dataset)
        train_loader.set_indicies(train_idx)
        test_loader = copy.deepcopy(dataset)
        test_loader.set_indicies(test_idx)

        train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels()))
        test_tf = tf.data.Dataset.from_tensor_slices((test_loader.get_frames(), test_loader.get_labels()))
        
#         negative_ds = (
#           train_tf
#             .filter(lambda features, label: label==0)
#             .repeat())
#         positive_ds = (
#           train_tf
#             .filter(lambda features, label: label==1)
#             .repeat())
        
#         balanced_ds = tf.data.Dataset.sample_from_datasets(
#             [negative_ds, positive_ds], [0.5, 0.5])


#         train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
#         train_sampler = SubsetWeightedRandomSampler(get_sample_weights(train_idx, dataset), train_idx, replacement=True)
#         train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#                                                sampler=train_sampler)
        
        if args.checkpoint:
            classifier_model = keras.models.load_model(args.checkpoint)
        else:
            classifier_inputs = keras.Input(shape=(clip_depth, image_width//args.scale_factor, image_height//args.scale_factor))
            classifier_outputs = PNBTemporalClassifier(args.sparse_checkpoint)(classifier_inputs)

            classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)

        prediction_optimizer = keras.optimizers.Adam(learning_rate=args.lr)
        filter_optimizer = tf.keras.optimizers.SGD(learning_rate=args.sparse_lr)

        best_so_far = float('-inf')

        criterion = keras.losses.BinaryCrossentropy(from_logits=True, reduction=keras.losses.Reduction.SUM)

        if args.train:
            for epoch in range(args.epochs):
                epoch_loss = 0
                t1 = time.perf_counter()

                y_true_train = None
                y_pred_train = None

#                 for images, labels in tqdm(balanced_ds.batch(args.batch_size).take(len(train_tf) // args.batch_size)):
                for images, labels in tqdm(train_tf.shuffle(len(train_tf)).batch(args.batch_size)):
                    images = tf.transpose(images, [0, 2, 3, 4, 1])

                    if args.train_sparse:
                        with tf.GradientTape() as tape:
#                             activations = sparse_model([images, tf.expand_dims(recon_model.trainable_weights[0], axis=0)])
                            pred = classifier_model(activations)
                            loss = criterion(torch_labels, pred)

                            print(loss)
                    else:
                        with tf.GradientTape() as tape:
                            images = tf.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
                            alter = data_augmentation(images)
                            alter = tf.reshape(alter, (-1, clip_depth, alter.shape[1], alter.shape[2], alter.shape[3]))
                            pred = classifier_model(alter)
                            loss = criterion(labels, pred)

                    epoch_loss += loss * images.shape[0]

                    if args.train_sparse:
                        sparse_gradients, classifier_gradients = tape.gradient(loss, [recon_model.trainable_weights, classifier_model.trainable_weights])

                        prediction_optimizer.apply_gradients(zip(classifier_gradients, classifier_model.trainable_weights))

                        filter_optimizer.apply_gradients(zip(sparse_gradients, recon_model.trainable_weights))

                        if args.run_2d:
                            weights = normalize_weights(recon_model.get_weights(), args.num_kernels)
                        else:
                            weights = normalize_weights_3d(recon_model.get_weights(), args.num_kernels)
                        recon_model.set_weights(weights)
                    else:
                        gradients = tape.gradient(loss, classifier_model.trainable_weights)

                        prediction_optimizer.apply_gradients(zip(gradients, classifier_model.trainable_weights))


                    if y_true_train is None:
                        y_true_train = labels
                        y_pred_train = tf.math.round(tf.math.sigmoid(pred))
                    else:
                        y_true_train = tf.concat((y_true_train, labels), axis=0)
                        y_pred_train = tf.concat((y_pred_train, tf.math.round(tf.math.sigmoid(pred))), axis=0)

                t2 = time.perf_counter()

                y_true = None
                y_pred = None
                test_loss = 0.0
                
                eval_loader = test_loader
                if args.splits == 'all_train':
                    eval_loader = train_loader
                for images, labels in tqdm(test_tf.batch(args.batch_size)):
                    images = tf.transpose(images, [0, 2, 3, 4, 1])
                    
                    images = tf.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
                    alter = keras.layers.Resizing(image_width//args.scale_factor, image_height//args.scale_factor)(images)
                    alter = tf.reshape(alter, (-1, clip_depth, alter.shape[1], alter.shape[2], alter.shape[3]))
                    pred = classifier_model(alter)
                    
                    loss = criterion(labels, pred)

                    test_loss += loss

                    if y_true is None:
                        y_true = labels
                        y_pred = tf.math.round(tf.math.sigmoid(pred))
                    else:
                        y_true = tf.concat((y_true, labels), axis=0)
                        y_pred = tf.concat((y_pred, tf.math.round(tf.math.sigmoid(pred))), axis=0)

                t2 = time.perf_counter()

                y_true = tf.cast(y_true, tf.int32)
                y_pred = tf.cast(y_pred, tf.int32)

                y_true_train = tf.cast(y_true_train, tf.int32)
                y_pred_train = tf.cast(y_pred_train, tf.int32)

                f1 = f1_score(y_true, y_pred, average='macro')
                accuracy = accuracy_score(y_true, y_pred)

                train_accuracy = accuracy_score(y_true_train, y_pred_train)

                print('epoch={}, i_fold={}, time={:.2f}, train_loss={:.2f}, test_loss={:.2f}, train_acc={:.2f}, test_f1={:.2f}, test_acc={:.2f}'.format(epoch, i_fold, t2-t1, epoch_loss, test_loss, train_accuracy, f1, accuracy))
    #             print(epoch_loss)
                if f1 >= best_so_far:
                    print("found better model")
                    # Save model parameters
                    classifier_model.save(os.path.join(output_dir, "best_classifier_{}.pt".format(i_fold)))
#                     recon_model.save(os.path.join(output_dir, "best_sparse_model_{}.pt".format(i_fold)))
                    pickle.dump(prediction_optimizer.get_weights(), open(os.path.join(output_dir, 'optimizer_{}.pt'.format(i_fold)), 'wb+'))
                    best_so_far = f1

            classifier_model = keras.models.load_model(os.path.join(output_dir, "best_classifier_{}.pt".format(i_fold)))
#             recon_model = keras.models.load_model(os.path.join(output_dir, 'best_sparse_model_{}.pt'.format(i_fold)))

        epoch_loss = 0

        y_true = None
        y_pred = None

        pred_dict = {}
        gt_dict = {}

        t1 = time.perf_counter()
    #         test_videos = [vid_f for labels, local_batch, vid_f in batch for batch in test_loader]
        transform = torchvision.transforms.Compose(
        [VideoGrayScaler(),
         MinMaxScaler(0, 255),
         torchvision.transforms.Resize((image_width//args.scale_factor, image_height//args.scale_factor))
        ])

        test_videos = test_loader.get_all_videos()

        test_labels = [vid_f.split('/')[-3] for vid_f in test_videos]

        if args.frames_to_skip == 1:
            y_pred, y_true, fn, fp = calculate_pnb_scores(test_videos, test_labels, yolo_model, sparse_model, recon_model, classifier_model, image_width, image_height, transform)
        else:
            y_pred, y_true, fn, fp = calculate_pnb_scores_skipped_frames(test_videos, test_labels, yolo_model, sparse_model, recon_model, classifier_model, args.frames_to_skip, image_width, image_height, clip_depth, transform)
            
        print(fn)
        print(fp)
            
        t2 = time.perf_counter()

        print('i_fold={}, time={:.2f}'.format(i_fold, t2-t1))

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        f1 = f1_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)

        fn_ids.extend(fn)
        fp_ids.extend(fp)

        overall_true.extend(y_true)
        overall_pred.extend(y_pred)

        print("Test f1={:.2f}, vid_acc={:.2f}".format(f1, accuracy))

        print(confusion_matrix(y_true, y_pred))
            
        i_fold += 1

    fp_fn_file = os.path.join(args.output_dir, 'fp_fn.txt')
    with open(fp_fn_file, 'w+') as in_f:
        in_f.write('FP:\n')
        in_f.write(str(fp_ids) + '\n\n')
        in_f.write('FN:\n')
        in_f.write(str(fn_ids) + '\n\n')
        
    overall_true = np.array(overall_true)
    overall_pred = np.array(overall_pred)
            
    final_f1 = f1_score(overall_true, overall_pred, average='macro')
    final_acc = accuracy_score(overall_true, overall_pred)
    final_conf = confusion_matrix(overall_true, overall_pred)
            
    print("Final accuracy={:.2f}, f1={:.2f}".format(final_acc, final_f1))
    print(final_conf)

