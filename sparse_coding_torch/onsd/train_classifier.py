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
from sparse_coding_torch.onsd.classifier_model import ONSDClassifier
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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# configproto = tf.compat.v1.ConfigProto()
# configproto.gpu_options.polling_inactive_delay_msecs = 5000
# configproto.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=configproto) 
# session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# tf.compat.v1.keras.backend.set_session(sess)
# tf.debugging.set_log_device_placement(True)

def split_difficult_vids(vid_list, num_splits):
    output_array = [[] for _ in range(num_splits)]
    for i, v in enumerate(vid_list):
        output_array[(i + 1) % num_splits].append(v)
        
    return output_array

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

#             print(tf.math.reduce_sum(activations))

        pred, _ = classifier_model(activations)

        final_pred = float(tf.math.round(tf.math.sigmoid(pred)))

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

def calculate_onsd_scores(input_videos, labels, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height, max_width):
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
        
        all_classes = []
        all_widths = []
    
        for i in range(0, vc.size(1), 20):
            frame = vc[:, i, :, :]
                
            nerve = get_yolo_region_onsd(yolo_model, frame, crop_width, crop_height, False)
            
            if not nerve:
                continue

            nerve = nerve[0]

            nerve = transform(nerve).to(torch.float32).unsqueeze(3).unsqueeze(1).numpy()
            
            activations = tf.stop_gradient(sparse_model([nerve, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

            pred = classifier_model(activations)

            pred = float(tf.math.round(tf.math.sigmoid(pred)))
#             width_pred = tf.math.round(width_pred * max_width)
            
            all_classes.append(pred)
#             all_widths.append(width_pred)
            
        final_pred = np.round(np.average(np.array(all_classes)))
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

def calculate_onsd_scores_frame_classifier(input_videos, labels, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height, max):
    good_frame_model = keras.models.load_model('sparse_coding_torch/onsd/valid_frame_model_2/best_classifier.pt/')
    
    resize = torchvision.transforms.Compose(
    [
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize((512, 512))
    ])
    
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
        
        best_frame = None
        best_conf = 0
    
        for i in range(0, vc.size(1)):
            frame = vc[:, i, :, :]

            frame = resize(frame).swapaxes(0, 2).swapaxes(0, 1).numpy()

            prepro_frame = np.expand_dims(frame, axis=0)

#             prepro_frame = tf.keras.applications.densenet.preprocess_input(frame)

            pred = good_frame_model(prepro_frame)

            pred = tf.math.sigmoid(pred)
            
            if pred > best_conf:
                best_conf = pred
                best_frame = vc[:, i, :, :]
                
        frame = get_yolo_region_onsd(yolo_model, best_frame, crop_width, crop_height, False)
            
        if frame is None or len(frame) == 0:
            final_pred = 1.0
        else:
            frame = frame[0]

            frame = transform(frame).to(torch.float32).unsqueeze(3).unsqueeze(1).numpy()
            
            activations = tf.stop_gradient(sparse_model([frame, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

            pred = classifier_model(activations)

            final_pred = float(tf.math.round(tf.math.sigmoid(pred)))
#             final_pred = 1.0
            
        if final_pred != numerical_labels[v_idx]:
            if final_pred == 0:
                fn_ids.append(f)
            else:
                fp_ids.append(f)
            
        final_list.append(final_pred)
        
    return np.array(final_list), np.array(numerical_labels), fn_ids, fp_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--kernel_height', default=30, type=int)
    parser.add_argument('--kernel_width', default=60, type=int)
    parser.add_argument('--kernel_depth', default=1, type=int)
    parser.add_argument('--num_kernels', default=8, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=300, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--checkpoint', default=None, type=str)
    parser.add_argument('--splits', default=None, type=str, help='k_fold or leave_one_out or all_train or custom')
    parser.add_argument('--seed', default=26, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--num_positives', default=100, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--save_train_test_splits', action='store_true')
    parser.add_argument('--run_2d', action='store_true')
    parser.add_argument('--balance_classes', action='store_true')
    parser.add_argument('--dataset', default='onsd', type=str)
    parser.add_argument('--train_sparse', action='store_true')
    parser.add_argument('--mixing_ratio', type=float, default=1.0)
    parser.add_argument('--sparse_lr', type=float, default=0.003)
    parser.add_argument('--crop_height', type=int, default=30)
    parser.add_argument('--crop_width', type=int, default=300)
    parser.add_argument('--image_height', type=int, default=30)
    parser.add_argument('--image_width', type=int, default=250)
    parser.add_argument('--clip_depth', type=int, default=1)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    
    args = parser.parse_args()
    
    crop_height = args.crop_height
    crop_width = args.crop_width

    image_height = args.image_height
    image_width = args.image_width
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
#     yolo_model = None

    all_errors = []
    
    if args.run_2d:
        inputs = keras.Input(shape=(image_height, image_width, clip_depth))
    else:
        inputs = keras.Input(shape=(clip_depth, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(clip_depth, args.kernel_height, args.kernel_width, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_height=args.kernel_height, kernel_width=args.kernel_width, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
    recon_model = keras.models.load_model(args.sparse_checkpoint)
    
    crop_amount = (crop_width - image_width)
    assert crop_amount % 2 == 0
    crop_amount = crop_amount // 2
        
    data_augmentation = keras.Sequential([
        keras.layers.RandomTranslation(0, 0.08),
        keras.layers.Cropping2D((0, crop_amount))
    ])
    
    just_crop = keras.layers.Cropping2D((0, crop_amount))
        
    
    splits, dataset = load_onsd_videos(args.batch_size, crop_size=(crop_height, crop_width), yolo_model=yolo_model, mode=args.splits, n_splits=args.n_splits)
    positive_class = 'Positives'
    
#     difficult_vids = split_difficult_vids(dataset.get_difficult_vids(), args.n_splits)

    overall_true = []
    overall_pred = []
    fn_ids = []
    fp_ids = []
    
    with open(os.path.join(output_dir, 'test_ids.txt'),'w') as f:
        pass

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
            classifier_inputs = keras.Input(shape=output.shape[1:])
            classifier_outputs = ONSDClassifier(args.sparse_checkpoint)(classifier_inputs)

            classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)

        prediction_optimizer = keras.optimizers.Adam(learning_rate=args.lr)
        filter_optimizer = tf.keras.optimizers.SGD(learning_rate=args.sparse_lr)

        best_so_far = float('inf')

        class_criterion = keras.losses.BinaryCrossentropy(from_logits=True, reduction=keras.losses.Reduction.SUM)
#         width_criterion = keras.losses.MeanSquaredError(reduction=keras.losses.Reduction.SUM)


        train_losses = []
        test_losses = []
        
        train_accuracies = []
        test_accuracies = []
        
#         train_mse = []
#         test_mse = []
        if args.train:
            for epoch in range(args.epochs):
                epoch_loss = 0
                t1 = time.perf_counter()

                y_true_train = None
                y_pred_train = None

                classifier_model.do_dropout = True
                for images, labels, width in tqdm(train_tf.shuffle(len(train_tf)).batch(args.batch_size)):
                    images = tf.expand_dims(data_augmentation(tf.transpose(images, [0, 2, 3, 1])), axis=1)

                    width_mask = tf.cast(tf.math.ceil(width), tf.float32)

                    if args.train_sparse:
                        with tf.GradientTape() as tape:
#                             activations = sparse_model([images, tf.expand_dims(recon_model.trainable_weights[0], axis=0)])
                            pred = classifier_model(activations)
                            loss = criterion(torch_labels, pred)

                            print(loss)
                    else:
                        activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
                        with tf.GradientTape() as tape:
                            class_pred = classifier_model(activations)
                            class_loss = class_criterion(labels, class_pred)
#                             width_loss = width_criterion(width, width_pred * width_mask)
                            loss = class_loss# + width_loss

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
                        y_pred_train = tf.math.round(tf.math.sigmoid(class_pred))
                    else:
                        y_true_train = tf.concat((y_true_train, labels), axis=0)
                        y_pred_train = tf.concat((y_pred_train, tf.math.round(tf.math.sigmoid(class_pred))), axis=0)

                t2 = time.perf_counter()

                y_true = None
                y_pred = None
                test_loss = 0.0
                test_width_loss = 0.0
                width_p = []
                width_gt = []
                
#                 eval_loader = test_tf
#                 if args.splits == 'all_train':
#                     eval_loader = train_tf
                classifier_model.do_dropout = False
                for images, labels, width in tqdm(test_tf.batch(args.batch_size)):
                    images = tf.expand_dims(just_crop(tf.transpose(images, [0, 2, 3, 1])), axis=1)
                
                    activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

                    pred = classifier_model(activations)
                    class_loss = class_criterion(labels, pred)
#                     width_loss = width_criterion(width, width_pred)
                    test_loss += (class_loss) * images.shape[0]

#                     test_loss += (class_loss + width_loss) * images.shape[0]
#                     test_width_loss += width_loss * images.shape[0]

                    if y_true is None:
                        y_true = labels
                        y_pred = tf.math.round(tf.math.sigmoid(pred))
                    else:
                        y_true = tf.concat((y_true, labels), axis=0)
                        y_pred = tf.concat((y_pred, tf.math.round(tf.math.sigmoid(pred))), axis=0)
                        
#                     for p, g in zip(width_pred, width):
#                         if g == 0:
#                             continue
#                         width_p.append(p * dataset.max_width)
#                         width_gt.append(g * dataset.max_width)

                t2 = time.perf_counter()

                y_true = tf.cast(y_true, tf.int32)
                y_pred = tf.cast(y_pred, tf.int32)
                
#                 print(tf.math.count_nonzero(y_true))
#                 print(y_true.shape)
#                 raise Exception

                y_true_train = tf.cast(y_true_train, tf.int32)
                y_pred_train = tf.cast(y_pred_train, tf.int32)

                f1 = f1_score(y_true, y_pred, average='macro')
                accuracy = accuracy_score(y_true, y_pred)

                train_accuracy = accuracy_score(y_true_train, y_pred_train)
                
#                 test_mae = keras.losses.MeanAbsoluteError()(width_gt, width_p)
                test_mae = 0.0
                
                train_losses.append(epoch_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(accuracy)

                print('epoch={}, i_fold={}, time={:.2f}, train_loss={:.2f}, test_loss={:.2f}, test_width_loss={:.2f}, train_acc={:.2f}, test_f1={:.2f}, test_acc={:.2f}, test_mae={:.2f}'.format(epoch, i_fold, t2-t1, epoch_loss, test_loss, test_width_loss, train_accuracy, f1, accuracy, test_mae))
#                 print('epoch={}, i_fold={}, time={:.2f}, train_loss={:.2f}, test_loss={:.2f}, train_acc={:.2f}, test_f1={:.2f}, test_acc={:.2f}, test_mae={:.2f}'.format(epoch, i_fold, t2-t1, epoch_loss, test_loss, train_accuracy, f1, accuracy, test_mae))
    #             print(epoch_loss)
                if epoch_loss < best_so_far:
                    print("found better model")
                    # Save model parameters
                    classifier_model.save(os.path.join(output_dir, "best_classifier_{}.pt".format(i_fold)))
#                     recon_model.save(os.path.join(output_dir, "best_sparse_model_{}.pt".format(i_fold)))
                    pickle.dump(prediction_optimizer.get_weights(), open(os.path.join(output_dir, 'optimizer_{}.pt'.format(i_fold)), 'wb+'))
                    best_so_far = epoch_loss

            classifier_model = keras.models.load_model(os.path.join(output_dir, "best_classifier_{}.pt".format(i_fold)))
#             recon_model = keras.models.load_model(os.path.join(output_dir, 'best_sparse_model_{}.pt'.format(i_fold)))

#         epoch_loss = 0

        y_true = None
        y_pred = None

        pred_dict = {}
        gt_dict = {}

        t1 = time.perf_counter()
        transform = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(1),
         MinMaxScaler(0, 255),
         torchvision.transforms.CenterCrop((image_height, image_width))
        ])

        test_videos = list(test_loader.get_all_videos())# + [v[1] for v in difficult_vids[i_fold]]

        test_labels = [vid_f.split('/')[-3] for vid_f in test_videos]

        classifier_model.do_dropout = False
        max_width = 0
        if hasattr(dataset, 'max_width'):
            max_width = dataset.max_width
        y_pred, y_true, fn, fp = calculate_onsd_scores(test_videos, test_labels, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height, max_width)
#         y_pred, y_true, fn, fp = calculate_onsd_scores_measured(test_videos, yolo_model, classifier_model, sparse_model, recon_model, transform, crop_width, crop_height)
            
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
        
        plt.clf()
        plt.figure()
        
        plt.subplot(211)
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(range(len(test_losses)), test_losses)
        
        plt.subplot(212)
        plt.plot(range(len(train_accuracies)), train_accuracies)
        plt.plot(range(len(test_accuracies)), test_accuracies)
        plt.savefig(os.path.join(args.output_dir, 'loss_acc_graph_{}.png'.format(i_fold)))
            
        i_fold += 1

    if args.splits == 'all_train':
        transform = torchvision.transforms.Compose(
        [torchvision.transforms.Grayscale(1),
         MinMaxScaler(0, 255),
         torchvision.transforms.Resize((image_height, image_width))
        ])

        overall_pred, overall_true, fn_ids, fp_ids = calculate_onsd_scores_measured(yolo_model, classifier_model, sparse_model, recon_model, transform, image_width, image_height)
        
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

