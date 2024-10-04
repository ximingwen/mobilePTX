import os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as keras
import random
import numpy as np
import cv2
import glob
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import save_img
from PIL import ImageOps
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
from matplotlib import cm
from unet_models import get_model
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from keras_unet_collection.models import unet_2d
from yolov4.get_bounding_boxes import YoloModel
import torchvision as tv
from sparse_coding_torch.onsd.video_loader import get_yolo_region_onsd
import torch

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def load_videos(input_dir, target_dir):
    target_img_paths = sorted(
    [
        os.path.join(fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
    )

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png") and fname in target_img_paths
        ]
    )

    target_img_paths = [os.path.join(target_dir, path) for path in target_img_paths]

    assert len(input_img_paths) == len(target_img_paths)

    print("Number of training samples:", len(input_img_paths))

    input_data = []
    for input_path, target_path in zip(input_img_paths, target_img_paths):
        input_data.append((input_path, target_path))
        
    return input_data

def get_videos(input_participant):
    all_vids = glob.glob(os.path.join(video_path, '*', '*', '*.mp4'))
    
    out_vids = []
    
    for vid in all_vids:
        vid_name = vid.split('/')[-1][:-4]
        participant = vid.split('/')[-2]
        txt_label = vid.split('/')[-3]
        
        if input_participant == participant:
            out_vids.append(vid)
            
    return out_vids

def get_participants(video_path):
    all_vids = glob.glob(os.path.join(video_path, '*', '*', '*.mp4'))

    participant_to_data = {}

    for vid in all_vids:
        vid_name = vid.split('/')[-1][:-4]
        participant = vid.split('/')[-2]
        txt_label = vid.split('/')[-3]

        for frame in input_data:
            frame_name = frame[0].split('/')[-1][:-4]
            frame_name = frame_name[:frame_name.rfind('_')]

            if frame_name != vid_name:
                continue

            if not participant in participant_to_data:
                participant_to_data[participant] = []

            participant_to_data[participant].append((frame[0], frame[1], txt_label))

    print('{} participants.'.format(len(participant_to_data)))
    
    return participant_to_data

def create_splits(participant_to_data):
    participants = list(participant_to_data.keys())

    random.shuffle(participants)
    
    gss = LeaveOneOut()

    splits = gss.split(participants)
    
    return splits, participants

def make_numpy_arrays(split_participants):
    all_x = []
    all_y = []
    all_txt = []
    
    for participant in split_participants:
        for x, y, txt_label in participant_to_data[participant]:
            x = cv2.resize(cv2.imread(x, cv2.IMREAD_GRAYSCALE), (img_size[1], img_size[0]))
            y = cv2.resize(cv2.imread(y, cv2.IMREAD_GRAYSCALE), (img_size[1], img_size[0]))
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if y[i, j] == 255:
                        y[i, j] = 1.0
                    else:
                        y[i, j] = 0.0

            all_x.append(x)
            all_y.append(y)
            all_txt.append(txt_label)
            
    return np.expand_dims(np.stack(all_x), axis=-1), np.stack(all_y), all_txt

def display_mask_test(model, input_mask):
    """Quick utility to display a model's prediction."""
    test_pred = model.predict(np.expand_dims(np.expand_dims(input_mask, axis=0), axis=-1), verbose=False)[0]
    mask = np.argmax(test_pred, axis=-1)
    mask = np.expand_dims(mask, axis=-1) * 255
    
    return mask

def get_width_measurement(mask):
    x1 = float('inf')
    x2 = float('-inf')
    for i in range(mask.shape[0]//2, (mask.shape[0]//2)+10):
        for j in range(10,mask.shape[1]-10):
            if mask[i, j] == 1 or mask[i, j] == 255:
                x1 = min(x1, j)
                x2 = max(x2, j)
                
    if x1 == float('inf') or x2 == float('-inf'):
        x1 = 0
        x2 = 0
        print('AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH')
                
    return x1, x2

def get_width_predictions(model, X, y, lbls, pos_neg_cutoff):
    all_widths = []
    pred_widths = []
    class_preds = []
    gt_mask_preds = []
    class_gt = []
    
    pred = np.argmax(model.predict(np.expand_dims(X, axis=-1), verbose=False), axis=-1)
    
    for p, gt, lbl in zip(pred, y, lbls):
        x1, x2 = get_width_measurement(gt)

        x1_pred, x2_pred = get_width_measurement(p)

        width = x2 - x1
        pred_width = x2_pred - x1_pred
        all_widths.append(width)
        pred_widths.append(pred_width)
        
        if width >= pos_neg_cutoff:
            gt_mask_preds.append(1)
        else:
            gt_mask_preds.append(0)
        
        if pred_width >= pos_neg_cutoff:
            class_preds.append(1)
        else:
            class_preds.append(0)
            
        if lbl == 'Positives':
            class_gt.append(1)
        else:
            class_gt.append(0)

    return np.array(all_widths), np.array(pred_widths), np.array(gt_mask_preds), np.array(class_preds), np.array(class_gt)

def run_full_eval(model, yolo_model, videos, lbl, img_size, pos_neg_cutoff):
    pred_widths = []
    class_preds = []
    class_gt = []
    
    transforms = tv.transforms.Compose(
    [tv.transforms.Grayscale(1)
    ])
    
    all_regions = []
    for video_path in videos:
        vc = tv.io.read_video(video_path)[0].permute(3, 0, 1, 2)

        all_frames = [vc[:, j, :, :] for j in range(0, vc.size(1), 10)]

        regions = []
        for frame in all_frames:
            yolo_detections = get_yolo_region_onsd(yolo_model, frame, 250, 150, False)
            if yolo_detections is None:
                continue
            for region in yolo_detections:
                region = transforms(region)
                regions.append(region)
        
        regions = [r.numpy().swapaxes(0,1).swapaxes(1,2) for r in regions if r is not None]
        
        if len(regions) == 0:
            continue

        regions = np.stack(regions)
        
        all_regions.append(regions)
        
    all_regions = np.concatenate(all_regions)
    
    X = keras.layers.Resizing(img_size[0], img_size[1])(all_regions)
    
    pred = np.argmax(model.predict(X, verbose=False), axis=-1)
    
    for p in pred:
        x1_pred, x2_pred = get_width_measurement(p)

        pred_width = x2_pred - x1_pred
        pred_widths.append(pred_width)
        
    pred_width = np.average(pred_widths)

    if pred_width >= pos_neg_cutoff:
        class_preds.append(1)
    else:
        class_preds.append(0)
            
    if lbl == 'Positives':
        class_gt.append(1)
    else:
        class_gt.append(0)

    return np.array(class_preds), np.array(class_gt)

def run_full_eval_measured(model, yolo_model, videos, lbl, img_size, pos_neg_cutoff):
    frame_path = 'sparse_coding_torch/onsd/onsd_good_for_eval'
    
    pred_widths = []
    class_preds = []
    class_gt = []
    
    transforms = tv.transforms.Compose(
    [tv.transforms.Grayscale(1)
    ])
    
    all_regions = []
    for vid_f in videos:
        split_path = vid_f.split('/')
        frame_path = '/'.join(split_path[:-1])
        label = split_path[-3]
        f = [png_file for png_file in os.listdir(frame_path) if png_file.endswith('.png')][0]

        frame = torch.tensor(cv2.imread(os.path.join(frame_path, f))).swapaxes(2, 1).swapaxes(1, 0)
        
        yolo_detections = get_yolo_region_onsd(yolo_model, frame, 250, 150, False)
        if yolo_detections is None:
            continue
            
        regions = []
        for region in yolo_detections:
            region = transforms(region)
            regions.append(region)
            
        regions = [r.numpy().swapaxes(0,1).swapaxes(1,2) for r in regions if r is not None]
        
        if len(regions) == 0:
            continue

        regions = np.stack(regions)
        
        all_regions.append(regions)
        
    if len(all_regions) == 0:
        if lbl == 'Positives':
            class_gt.append(1)
        else:
            class_gt.append(0)
        return np.array([1]), np.array(class_gt)
    all_regions = np.concatenate(all_regions)
    
    X = keras.layers.Resizing(img_size[0], img_size[1])(all_regions)
    
    pred = np.argmax(model.predict(X, verbose=False), axis=-1)
    
    for p in pred:
        x1_pred, x2_pred = get_width_measurement(p)

        pred_width = x2_pred - x1_pred
        pred_widths.append(pred_width)
        
    pred_width = np.average(pred_widths)

    if pred_width >= pos_neg_cutoff:
        class_preds.append(1)
    else:
        class_preds.append(0)
            
    if lbl == 'Positives':
        class_gt.append(1)
    else:
        class_gt.append(0)

    return np.array(class_preds), np.array(class_gt)
    

# random.seed(321534)
# np.random.seed(321534)
# tf.random.set_seed(321534)

output_dir = 'sparse_coding_torch/unet_output/unet_6'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_path = "/shared_data/bamc_onsd_data/revised_extended_onsd_data"

input_dir = "segmentation/segmentation_12_15/labeled_frames/"
target_dir = "segmentation/segmentation_12_15/labeled_frames/segmentation/"

yolo_model = YoloModel('onsd')

img_size = (160, 160)
batch_size = 12
pos_neg_cutoff = 74

input_data = load_videos(input_dir, target_dir)

participant_to_data = get_participants(video_path) 

splits, participants = create_splits(participant_to_data)

all_train_frame_pred = []
all_train_frame_gt = []

all_test_frame_pred = []
all_test_frame_gt = []
all_test_video_pred = []
all_test_video_gt = []

all_yolo_gt = []
all_yolo_pred = []

i_fold = 0
for train_idx, test_idx in splits:
    train_participants = [p for i, p in enumerate(participants) if i in train_idx]
    test_participants = [p for i, p in enumerate(participants) if i in test_idx]
    
    assert len(set(train_participants).intersection(set(test_participants))) == 0

    # Instantiate data Sequences for each split
    train_X, train_y, train_txt = make_numpy_arrays(train_participants)
    test_X, test_y, test_txt = make_numpy_arrays(test_participants)

    keras.backend.clear_session()

    # Build model
    model = unet_2d((None, None, 1), [64, 128, 256, 512, 1024], n_labels=2,
                      stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss="sparse_categorical_crossentropy")

    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "best_unet_model_{}.h5".format(i_fold)), save_best_only=True, save_weights_only=True)
    ]

    # Train the model, doing validation at the end of each epoch.
    if os.path.exists(os.path.join(output_dir, "best_unet_model_{}.h5".format(i_fold))):
        model.load_weights(os.path.join(output_dir, "best_unet_model_{}.h5".format(i_fold)))
    else:
        epochs = 1

        model.fit(train_X, train_y, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=callbacks)


    sample_idx = random.randrange(0, len(test_X))
    # Display input image
    cv2.imwrite(os.path.join(output_dir, 'input_image_{}.png'.format(i_fold)), test_X[sample_idx])

    # Display ground-truth target mask
    cv2.imwrite(os.path.join(output_dir, 'input_mask_{}.png'.format(i_fold)), np.expand_dims(test_y[sample_idx] * 255, axis=-1))

    # Display mask predicted by our model
    cv2.imwrite(os.path.join(output_dir, 'pred_mask_{}.png'.format(i_fold)), display_mask_test(model, test_X[sample_idx]))  # Note that the model only sees inputs at 150x150.
             
    final_width_train, final_pred_width_train, class_gt_mask_train, class_pred_train, class_gt_train = get_width_predictions(model, train_X, train_y, train_txt, pos_neg_cutoff)
    
    train_average_width_difference = np.average(np.abs(np.array(final_width_train) - np.array(final_pred_width_train)))
    
    train_gt_mask_class_score = accuracy_score(class_gt_train, class_gt_mask_train)
    
    train_pred_mask_class_score = accuracy_score(class_gt_train, class_pred_train)
    
    print('Training results fold {}: average width difference={:.2f}, ground truth mask classification={:.2f}, predicted mask classification={:.2f}'.format(i_fold, train_average_width_difference, train_gt_mask_class_score, train_pred_mask_class_score))

    final_width_test, final_pred_width_test, class_gt_mask_test, class_pred_test, class_gt_test = get_width_predictions(model, test_X, test_y, test_txt, pos_neg_cutoff)
    
    test_average_width_difference = np.average(np.abs(np.array(final_width_test) - np.array(final_pred_width_test)))
    
    test_gt_mask_class_score = accuracy_score(class_gt_test, class_gt_mask_test)
    
    test_pred_mask_class_score = accuracy_score(class_gt_test, class_pred_test)
    
    video_level_test_width = np.average(final_width_test)
    video_level_test_pred_width = np.average(final_pred_width_test)
    
    if video_level_test_width >= pos_neg_cutoff:
        gt_video_pred = np.array([1])
    else:
        gt_video_pred = np.array([0])
        
    if video_level_test_pred_width >= pos_neg_cutoff:
        pred_video_pred = np.array([1])
    else:
        pred_video_pred = np.array([0])
    
    if test_txt[0] == 'Positives':
        video_class = np.array([1])
    else:
        video_class = np.array([0])
        
    test_video_gt_mask_score = accuracy_score(video_class, gt_video_pred)
    
    test_video_pred_mask_score = accuracy_score(video_class, pred_video_pred)
    
    print('Testing results fold {}: average width difference={:.2f}, ground truth mask classification={:.2f}, predicted mask classification={:.2f}, ground truth mask video-level classification:{:.2f}, predicted mask video-level classification={:.2f}'.format(i_fold, test_average_width_difference, test_gt_mask_class_score, test_pred_mask_class_score, test_video_gt_mask_score, test_video_pred_mask_score))
    
    all_train_frame_pred.append(class_pred_train)
    all_train_frame_gt.append(class_gt_train)
    
    all_test_frame_pred.append(class_pred_test)
    all_test_frame_gt.append(class_gt_test)
    all_test_video_pred.append(pred_video_pred)
    all_test_video_gt.append(video_class)
    
    videos = get_videos(participants[i_fold])
    lbl = test_txt[0]
    yolo_pred, yolo_gt = run_full_eval_measured(model, yolo_model, videos, lbl, img_size, pos_neg_cutoff)
    
    yolo_pred_score = accuracy_score(yolo_gt, yolo_pred)
    
    print('YOLO testing results fold {}: Video Accuracy={:.2f}'.format(i_fold, yolo_pred_score))
    
    all_yolo_gt.append(yolo_gt)
    all_yolo_pred.append(yolo_pred)
    
    i_fold += 1
    
all_train_frame_pred = np.concatenate(all_train_frame_pred)
all_train_frame_gt = np.concatenate(all_train_frame_gt)

all_test_frame_pred = np.concatenate(all_test_frame_pred)
all_test_frame_gt = np.concatenate(all_test_frame_gt)
all_test_video_pred = np.concatenate(all_test_video_pred)
all_test_video_gt = np.concatenate(all_test_video_gt)

final_train_frame_acc = accuracy_score(all_train_frame_gt, all_train_frame_pred)
final_test_frame_acc = accuracy_score(all_test_frame_gt, all_test_frame_pred)
final_test_video_acc = accuracy_score(all_test_video_gt, all_test_video_pred)

all_yolo_gt = np.concatenate(all_yolo_gt)
all_yolo_pred = np.concatenate(all_yolo_pred)

final_yolo_score = accuracy_score(all_yolo_gt, all_yolo_pred)

print('Final results: Train frame-level classification={:.2f}, Test frame-level classification={:.2f}, Test video-level classification={:.2f}, YOLO video-level classification={:.2f}'.format(final_train_frame_acc, final_test_frame_acc, final_test_video_acc, final_yolo_score))