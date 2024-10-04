import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.ptx.load_data import load_yolo_clips, load_covid_clips
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, normalize_weights, normalize_weights_3d
from sparse_coding_torch.ptx.classifier_model import PTXClassifier, VAEEncoderPTX, PTXVAEClassifier
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
import pickle
import tensorflow.keras as keras
import tensorflow as tf
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
from yolov4.get_bounding_boxes import YoloModel
import torchvision
import glob
from torchvision.datasets.video_utils import VideoClips
import cv2
import copy

configproto = tf.compat.v1.ConfigProto()
configproto.gpu_options.polling_inactive_delay_msecs = 5000
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)

def calculate_ptx_scores(input_videos, labels, yolo_model, sparse_model, recon_model, classifier_model, image_width, image_height, transform):
    all_predictions = []
    
    numerical_labels = []
    for label in labels:
        if label == 'No_Sliding':
            numerical_labels.append(1.0)
        else:
            numerical_labels.append(0.0)

    final_list = []
    clip_correct = []
    fp_ids = []
    fn_ids = []
    for v_idx, f in tqdm(enumerate(input_videos)):
        clipstride = 15
        
        vc = VideoClips([f],
                        clip_length_in_frames=5,
                        frame_rate=20,
                       frames_between_clips=clipstride)

        clip_predictions = []
        i = 0
        cliplist = []
        countclips = 0
        for i in range(vc.num_clips()):
            clip, _, _, _ = vc.get_clip(i)
            clip = clip.swapaxes(1, 3).swapaxes(0, 1).swapaxes(2, 3).numpy()
            
            bounding_boxes, classes, scores = yolo_model.get_bounding_boxes(clip[:, 2, :, :].swapaxes(0, 2).swapaxes(0, 1))
            bounding_boxes = bounding_boxes.squeeze(0)
            if bounding_boxes.size == 0:
                continue
            #widths = []
            countclips = countclips + len(bounding_boxes)
            
            widths = [(bounding_boxes[i][3] - bounding_boxes[i][1]) for i in range(len(bounding_boxes))]

            ind =  np.argmax(np.array(widths))

            bb = bounding_boxes[ind]
            center_x = (bb[3] + bb[1]) / 2 * 1920
            center_y = (bb[2] + bb[0]) / 2 * 1080

            width=400
            height=400

            lower_y = round(center_y - height / 2)
            upper_y = round(center_y + height / 2)
            lower_x = round(center_x - width / 2)
            upper_x = round(center_x + width / 2)

            trimmed_clip = clip[:, :, lower_y:upper_y, lower_x:upper_x]

            trimmed_clip = torch.tensor(trimmed_clip).to(torch.float)

            trimmed_clip = transform(trimmed_clip)
            trimmed_clip.pin_memory()
            cliplist.append(trimmed_clip)

        if len(cliplist) > 0:
            with torch.no_grad():
                trimmed_clip = torch.stack(cliplist)
                images = trimmed_clip.permute(0, 2, 3, 4, 1).numpy()
                activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.weights[0], axis=0))]))

                pred = classifier_model(activations)

                clip_predictions = tf.math.round(tf.math.sigmoid(pred))

            final_pred = torch.mode(torch.tensor(clip_predictions.numpy()).view(-1))[0].item()
            if len(clip_predictions) % 2 == 0 and tf.math.reduce_sum(clip_predictions) == len(clip_predictions)//2:
                #print("I'm here")
                final_pred = torch.mode(torch.tensor(clip_predictions.numpy()).view(-1))[0].item()
        else:
            final_pred = 1.0
            
        if final_pred != numerical_labels[v_idx]:
            if final_pred == 0.0:
                fn_ids.append(f)
            else:
                fp_ids.append(f)
            
        final_list.append(final_pred)
        
        clip_correct.extend([1 if clip_pred == numerical_labels[v_idx] else 0 for clip_pred in clip_predictions])
        
    return np.array(final_list), np.array(numerical_labels), fn_ids, fp_ids, sum(clip_correct) / len(clip_correct)

def calculate_covid_scores(input_videos, labels, sparse_model, recon_model, classifier_model, image_width, image_height, transform):
    all_predictions = []

    final_list = []
    clip_correct = []
    fp_ids = []
    fn_ids = []
    for v_idx, f in tqdm(enumerate(input_videos)):
        clipstride = 15
        
        vc = VideoClips([f],
                        clip_length_in_frames=5,
                        frame_rate=20,
                       frames_between_clips=clipstride)

        clip_predictions = []
        cliplist = []
        countclips = 0
        for clip_idx in range(vc.num_clips()):
            try:
                clip, _, _, _ = vc.get_clip(clip_idx)
            except Exception:
                continue
            clip = clip.swapaxes(1, 3).swapaxes(0, 1).swapaxes(2, 3)

            clip = transform(clip)
            cliplist.append(clip)

        if len(cliplist) > 0:
            with torch.no_grad():
                clip = torch.stack(cliplist)
                images = clip.permute(0, 2, 3, 4, 1).numpy()
                activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.weights[0], axis=0))]))

                pred = classifier_model(activations)

                clip_predictions = tf.math.argmax(tf.math.softmax(pred, axis=1), axis=1)

            final_pred = torch.mode(torch.tensor(clip_predictions.numpy()).view(-1))[0].item()
        else:
            final_pred = 0.0
            
        if final_pred != labels[v_idx]:
            if final_pred == 0.0:
                fn_ids.append(f)
            else:
                fp_ids.append(f)
            
        final_list.append(final_pred)
        
        clip_correct.extend([1 if clip_pred == labels[v_idx] else 0 for clip_pred in clip_predictions])
        
    return np.array(final_list), labels, fn_ids, fp_ids, sum(clip_correct) / len(clip_correct)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=12, type=int)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=64, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=150, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
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
    parser.add_argument('--train_sparse', action='store_true')
    parser.add_argument('--mixing_ratio', type=float, default=1.0)
    parser.add_argument('--sparse_lr', type=float, default=0.003)
    parser.add_argument('--crop_height', type=int, default=285)
    parser.add_argument('--crop_width', type=int, default=350)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--clip_depth', type=int, default=5)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='bamc')
    
    args = parser.parse_args()
    
    image_height = args.crop_height
    image_width = args.crop_width
    clip_depth = args.clip_depth
        
    batch_size = args.batch_size
    
#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
    
    yolo_model = YoloModel('ptx')

    all_errors = []
    
    if args.run_2d:
        inputs = keras.Input(shape=(image_height, image_width, clip_depth))
    else:
        inputs = keras.Input(shape=(clip_depth, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(args.kernel_depth, args.kernel_size, args.kernel_size, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)

    recon_inputs = keras.Input(shape=((clip_depth - args.kernel_depth) // 1 + 1, (image_height - args.kernel_size) // args.stride + 1, (image_width - args.kernel_size) // args.stride + 1, args.num_kernels))

    recon_outputs = ReconSparse(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(recon_inputs)

    recon_model = keras.Model(inputs=recon_inputs, outputs=recon_outputs)

    if args.sparse_checkpoint:
        recon_model.set_weights(keras.models.load_model(args.sparse_checkpoint).get_weights())
        
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(45)
    ])
    
#     augment_transforms = torchvision.transforms.Compose(
#     [torchvision.transforms.RandomRotation(45),
#      torchvision.transforms.RandomHorizontalFlip(),
#      torchvision.transforms.CenterCrop((100, 200))
#     ])
    
    if args.dataset == 'covid':
        splits, dataset = load_covid_clips(batch_size=args.batch_size, mode=args.splits, clip_width=image_width, clip_height=image_height, clip_depth=clip_depth, n_splits=args.n_splits)
    else:
        splits, dataset = load_yolo_clips(args.batch_size, num_clips=1, num_positives=15, mode=args.splits, device=None, n_splits=args.n_splits, sparse_model=None, whole_video=False, positive_videos='positive_videos.json')

    overall_true = []
    overall_pred = []
    fn_ids = []
    fp_ids = []
    
    i_fold = 0
    for train_idx, test_idx in splits:
        train_loader = copy.deepcopy(dataset)
        train_loader.set_indicies(train_idx)
        if test_idx is not None:
            test_loader = copy.deepcopy(dataset)
            test_loader.set_indicies(test_idx)
            test_tf = tf.data.Dataset.from_tensor_slices((test_loader.get_frames(), test_loader.get_labels()))

        train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels()))
        
        num_output = 1
        if args.dataset == 'covid':
            num_output = len(set(train_loader.get_unique_labels()))
        
        if args.checkpoint:
            classifier_model = keras.models.load_model(args.checkpoint)
        else:
            classifier_inputs = keras.Input(shape=((clip_depth - args.kernel_depth) // 1 + 1, (image_height - args.kernel_size) // args.stride + 1, (image_width - args.kernel_size) // args.stride + 1, args.num_kernels))
            classifier_outputs = PTXClassifier(num_output)(classifier_inputs)

            classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)

        prediction_optimizer = keras.optimizers.Adam(learning_rate=args.lr)
        filter_optimizer = tf.keras.optimizers.SGD(learning_rate=args.sparse_lr)

        best_so_far = float('-inf')

        if args.dataset == 'bamc':
            criterion = keras.losses.BinaryCrossentropy(from_logits=True, reduction=keras.losses.Reduction.SUM)
        else:
            criterion = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=keras.losses.Reduction.SUM)

        if args.train:
            for epoch in range(args.epochs):
                epoch_loss = 0
                t1 = time.perf_counter()

                y_true_train = None
                y_pred_train = None

                for images, labels in tqdm(train_tf.shuffle(len(train_tf)).batch(args.batch_size)):
#                     print(labels)
#                     alter = torch.stack([augment_transforms(t) for t in torch.tensor(np.array(images))])
#                     images = alter.swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4).numpy()
                    images = tf.transpose(images, [0, 2, 3, 4, 1])

                    if args.train_sparse:
                        with tf.GradientTape() as tape:
#                             activations = sparse_model([images, tf.expand_dims(recon_model.trainable_weights[0], axis=0)])
                            pred = classifier_model(activations)
                            loss = criterion(torch_labels, pred)

                            print(loss)
                    else:
                        images = tf.reshape(images, (images.shape[0], images.shape[2], images.shape[3], -1))
                        alter = data_augmentation(images)
                        alter = tf.reshape(alter, (images.shape[0], 5, alter.shape[1], alter.shape[2], 1))
                        activations = tf.stop_gradient(sparse_model([alter, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

                        with tf.GradientTape() as tape:
                            pred = classifier_model(activations)
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
                        if args.dataset == 'bamc':
                            y_pred_train = tf.math.round(tf.math.sigmoid(pred))
                        else:
                            y_pred_train = tf.math.argmax(tf.nn.softmax(pred, axis=1), axis=1)
                    else:
                        y_true_train = tf.concat((y_true_train, labels), axis=0)
                        if args.dataset == 'bamc':
                            y_pred_train = tf.concat((y_pred_train, tf.math.round(tf.math.sigmoid(pred))), axis=0)
                        else:
                            y_pred_train = tf.concat((y_pred_train, tf.math.argmax(tf.nn.softmax(pred, axis=1), axis=1)), axis=0)

                t2 = time.perf_counter()

                y_true = None
                y_pred = None
                test_loss = 0.0
                
                if args.splits == 'all_train':
                    eval_loader = train_tf
                else:
                    eval_loader = test_tf
                for images, labels in tqdm(eval_loader.batch(args.batch_size)):
                    images = tf.transpose(images, [0, 2, 3, 4, 1])
#                     images = images.numpy().swapaxes(1, 2).swapaxes(2, 3).swapaxes(3, 4)
                    
                    activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

                    pred = classifier_model(activations)
                    loss = criterion(labels, pred)

                    test_loss += loss * images.shape[0]

                    if y_true is None:
                        y_true = labels
                        if args.dataset == 'bamc':
                            y_pred = tf.math.round(tf.math.sigmoid(pred))
                        else:
                            y_pred = tf.math.argmax(tf.nn.softmax(pred, axis=1), axis=1)
                    else:
                        y_true = tf.concat((y_true, labels), axis=0)
                        if args.dataset == 'bamc':
                            y_pred = tf.concat((y_pred, tf.math.round(tf.math.sigmoid(pred))), axis=0)
                        else:
                            y_pred = tf.concat((y_pred, tf.math.argmax(tf.nn.softmax(pred, axis=1), axis=1)), axis=0)

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
        
#                 if accuracy == 1.0:
#                     break

            classifier_model = keras.models.load_model(os.path.join(output_dir, "best_classifier_{}.pt".format(i_fold)))
#             recon_model = keras.models.load_model(os.path.join(output_dir, 'best_sparse_model_{}.pt'.format(i_fold)))

        epoch_loss = 0

        y_true = None
        y_pred = None

        pred_dict = {}
        gt_dict = {}

        t1 = time.perf_counter()
        
        if args.dataset == 'bamc':
            transform = torchvision.transforms.Compose(
            [VideoGrayScaler(),
             MinMaxScaler(0, 255),
             torchvision.transforms.Normalize((0.2592,), (0.1251,)),
             torchvision.transforms.CenterCrop((100, 200))
            ])

            test_dir = '/shared_data/bamc_ph1_test_data'
            test_videos = glob.glob(os.path.join(test_dir, '*', '*.*'))
            test_labels = [vid_f.split('/')[-2] for vid_f in test_videos]

            y_pred, y_true, fn, fp, clip_acc = calculate_ptx_scores(test_videos, test_labels, yolo_model, sparse_model, recon_model, classifier_model, image_width, image_height, transform)
        elif args.dataset == 'covid':
            transform = torchvision.transforms.Compose(
            [VideoGrayScaler(),
             MinMaxScaler(0, 255),
             torchvision.transforms.Resize((image_height, image_width))
            ])

            test_videos = test_loader.get_all_videos()

            test_labels = test_loader.get_video_labels()

            y_pred, y_true, fn, fp, clip_acc = calculate_covid_scores(test_videos, test_labels, sparse_model, recon_model, classifier_model, image_width, image_height, transform)
            
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

        print("Test f1={:.2f}, vid_acc={:.2f}, clip_acc={:.2f}".format(f1, accuracy, clip_acc))

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

