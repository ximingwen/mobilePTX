import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.pnb.load_data import load_pnb_videos
from sparse_coding_torch.utils import SubsetWeightedRandomSampler, get_sample_weights
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, normalize_weights, normalize_weights_3d
from sparse_coding_torch.pnb.classifier_model import PNBClassifier
import time
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import random
import pickle
import tensorflow.keras as keras
import tensorflow as tf
from sparse_coding_torch.train_sparse_model import sparse_loss
from sparse_coding_torch.pnb.train_classifier import calculate_pnb_scores
from yolov4.get_bounding_boxes import YoloModel

configproto = tf.compat.v1.ConfigProto()
configproto.gpu_options.polling_inactive_delay_msecs = 5000
configproto.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=configproto) 
tf.compat.v1.keras.backend.set_session(sess)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--kernel_size', default=5, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=40, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=150, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=5e-4, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='sparse_coding_torch/classifier_outputs/needle_model', type=str)
    parser.add_argument('--needle_checkpoint', default='sparse_coding_torch/output/needle_codes/sparse_conv3d_model-best.pt/', type=str)
    parser.add_argument('--non_needle_checkpoint', default='sparse_coding_torch/output/non_needle_codes/sparse_conv3d_model-best.pt/', type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--splits', default='default', type=str, help='k_fold or leave_one_out or all_train')
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num_positives', default=100, type=int)
    parser.add_argument('--n_splits', default=1, type=int)
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
    
    if args.dataset == 'pnb':
        crop_height = args.crop_height
        crop_width = args.crop_width

        image_height = int(crop_height / args.scale_factor)
        image_width = int(crop_width / args.scale_factor)
        clip_depth = args.clip_depth
    elif args.dataset == 'ptx':
        image_height = 100
        image_width = 200
    else:
        raise Exception('Invalid dataset')
        
    batch_size = args.batch_size
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
    
    yolo_model = YoloModel()

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

    if args.needle_checkpoint and args.non_needle_checkpoint:
        weights = tf.concat([keras.models.load_model(args.needle_checkpoint).get_weights(), keras.models.load_model(args.non_needle_checkpoint).get_weights()], axis=5)
        recon_model.set_weights(weights)
        
    positive_class = None
    if args.dataset == 'pnb':
        splits, dataset = load_pnb_videos(yolo_model, args.batch_size, input_size=(image_height, image_width, clip_depth), crop_size=(crop_height, crop_width, clip_depth), classify_mode=True, balance_classes=args.balance_classes, mode=args.splits, device=None, n_splits=args.n_splits, sparse_model=None, frames_to_skip=args.frames_to_skip)
        positive_class = 'Positives'
    elif args.dataset == 'ptx':
        train_loader, test_loader, dataset = load_yolo_clips(args.batch_size, num_clips=1, num_positives=15, mode=args.splits, device=None, n_splits=args.n_splits, sparse_model=None, whole_video=False, positive_videos='positive_videos.json')
        positive_class = 'No_Sliding'
    else:
        raise Exception('Invalid dataset')

    overall_true = []
    overall_pred = []
    fn_ids = []
    fp_ids = []
    
    i_fold = 0
    for train_idx, test_idx in splits:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        train_sampler = SubsetWeightedRandomSampler(get_sample_weights(train_idx, dataset), train_idx, replacement=True)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
        
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=test_sampler)
        
        with open(os.path.join(args.output_dir, 'test_videos_{}.txt'.format(i_fold)), 'w+') as test_videos_out:
            test_set = set([x for tup in test_loader for x in tup[2]])
            test_videos_out.writelines(test_set)
        
        if args.checkpoint:
            classifier_model = keras.models.load_model(args.checkpoint)
        else:
            classifier_inputs = keras.Input(shape=((clip_depth - args.kernel_depth) // 1 + 1, (image_height - args.kernel_size) // args.stride + 1, (image_width - args.kernel_size) // args.stride + 1, args.num_kernels))

            if args.dataset == 'pnb':
                classifier_outputs = PNBClassifier()(classifier_inputs)
            elif args.dataset == 'ptx':
                classifier_outputs = PTXClassifier()(classifier_inputs)
            else:
                raise Exception('No classifier exists for that dataset')

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

                for labels, local_batch, vid_f in tqdm(train_loader):
                    images = local_batch.permute(0, 2, 3, 4, 1).numpy()

                    torch_labels = np.zeros(len(labels))
                    torch_labels[[i for i in range(len(labels)) if labels[i] == positive_class]] = 1
                    torch_labels = np.expand_dims(torch_labels, axis=1)

                    if args.train_sparse:
                        with tf.GradientTape() as tape:
                            activations = sparse_model([images, tf.expand_dims(recon_model.trainable_weights[0], axis=0)])
                            pred = classifier_model(activations)
                            loss = criterion(torch_labels, pred)

                            print(loss)
                    else:
                        activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

                        with tf.GradientTape() as tape:
                            pred = classifier_model(activations)
                            loss = criterion(torch_labels, pred)

                    epoch_loss += loss * local_batch.size(0)

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
                        y_true_train = torch_labels
                        y_pred_train = tf.math.round(tf.math.sigmoid(pred))
                    else:
                        y_true_train = tf.concat((y_true_train, torch_labels), axis=0)
                        y_pred_train = tf.concat((y_pred_train, tf.math.round(tf.math.sigmoid(pred))), axis=0)

                t2 = time.perf_counter()

                y_true = None
                y_pred = None
                test_loss = 0.0
                for labels, local_batch, vid_f in tqdm(test_loader):
                    images = local_batch.permute(0, 2, 3, 4, 1).numpy()

                    torch_labels = np.zeros(len(labels))
                    torch_labels[[i for i in range(len(labels)) if labels[i] == positive_class]] = 1
                    torch_labels = np.expand_dims(torch_labels, axis=1)

                    activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

                    pred = classifier_model(activations)
                    loss = criterion(torch_labels, pred)

                    test_loss += loss

                    if y_true is None:
                        y_true = torch_labels
                        y_pred = tf.math.round(tf.math.sigmoid(pred))
                    else:
                        y_true = tf.concat((y_true, torch_labels), axis=0)
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

        if args.dataset == 'pnb':
            epoch_loss = 0
            
            transform = torchvision.transforms.Compose(
            [VideoGrayScaler(),
             MinMaxScaler(0, 255),
             torchvision.transforms.Resize((image_height, image_width))
            ])

            y_true = None
            y_pred = None

            pred_dict = {}
            gt_dict = {}

            t1 = time.perf_counter()
    #         test_videos = [vid_f for labels, local_batch, vid_f in batch for batch in test_loader]
            test_videos = []
            test_labels = []
            for labels, local_batch, vid_f in test_loader:
                test_videos.extend(vid_f)
                test_labels.extend(labels)
                
            y_pred, y_true, fn, fp = calculate_pnb_scores(test_videos, test_labels, yolo_model, sparse_model, recon_model, classifier_model, 30, image_width, image_height, transform)

            t2 = time.perf_counter()

            print('i_fold={}, time={:.2f}'.format(i_fold, t2-t1))

            y_true = tf.cast(y_true, tf.int32)
            y_pred = tf.cast(y_pred, tf.int32)

            f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            
            fn_ids.extend(fn)
            fp_idx.extend(fp)

            print("Test f1={:.2f}, vid_acc={:.2f}".format(f1, accuracy))

            print(confusion_matrix(y_true, y_pred))
        elif args.dataset == 'ptx':
            epoch_loss = 0

            y_true = None
            y_pred = None

            pred_dict = {}
            gt_dict = {}

            t1 = time.perf_counter()
            for labels, local_batch, vid_f in test_loader:
                images = local_batch.permute(0, 2, 3, 4, 1).numpy()

                torch_labels = np.zeros(len(labels))
                torch_labels[[i for i in range(len(labels)) if labels[i] == positive_class]] = 1
                torch_labels = np.expand_dims(torch_labels, axis=1)

                activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.weights[0], axis=0))]))

                pred = classifier_model(activations)

                loss = criterion(torch_labels, pred)
                epoch_loss += loss * local_batch.size(0)

                for i, v_f in enumerate(vid_f):
                    if v_f not in pred_dict:
                        pred_dict[v_f] = tf.math.round(tf.math.sigmoid(pred[i]))
                    else:
                        pred_dict[v_f] = tf.concat((pred_dict[v_f], tf.math.round(tf.math.sigmoid(pred[i]))), axis=0)

                    if v_f not in gt_dict:
                        gt_dict[v_f] = torch_labels[i]
                    else:
                        gt_dict[v_f] = tf.concat((gt_dict[v_f], torch_labels[i]), axis=0)

                if y_true is None:
                    y_true = torch_labels
                    y_pred = tf.math.round(tf.math.sigmoid(pred))
                else:
                    y_true = tf.concat((y_true, torch_labels), axis=0)
                    y_pred = tf.concat((y_pred, tf.math.round(tf.math.sigmoid(pred))), axis=0)

            t2 = time.perf_counter()

            vid_acc = []
            for k in pred_dict.keys():
                gt_tmp = torch.tensor(gt_dict[k].numpy())
                pred_tmp = torch.tensor(pred_dict[k].numpy())

                gt_mode = torch.mode(gt_tmp)[0].item()
    #             perm = torch.randperm(pred_tmp.size(0))
    #             cutoff = int(pred_tmp.size(0)/4)
    #             if cutoff < 3:
    #                 cutoff = 3
    #             idx = perm[:cutoff]
    #             samples = pred_tmp[idx]
                pred_mode = torch.mode(pred_tmp)[0].item()
                overall_true.append(gt_mode)
                overall_pred.append(pred_mode)
                if pred_mode == gt_mode:
                    vid_acc.append(1)
                else:
                    vid_acc.append(0)
                    if pred_mode == 0:
                        fn_ids.append(k)
                    else:
                        fp_ids.append(k)

            vid_acc = np.array(vid_acc)

            print('----------------------------------------------------------------------------')
            for k in pred_dict.keys():
                print(k)
                print('Predictions:')
                print(pred_dict[k])
                print('Ground Truth:')
                print(gt_dict[k])
                print('Overall Prediction:')
                print(torch.mode(torch.tensor(pred_dict[k].numpy()))[0].item())
                print('----------------------------------------------------------------------------')

            print('loss={:.2f}, time={:.2f}'.format(loss, t2-t1))

            y_true = tf.cast(y_true, tf.int32)
            y_pred = tf.cast(y_pred, tf.int32)

            f1 = f1_score(y_true, y_pred, average='macro')
            accuracy = accuracy_score(y_true, y_pred)
            all_errors.append(np.sum(vid_acc) / len(vid_acc))

            print("Test f1={:.2f}, clip_acc={:.2f}, vid_acc={:.2f}".format(f1, accuracy, np.sum(vid_acc) / len(vid_acc)))

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

