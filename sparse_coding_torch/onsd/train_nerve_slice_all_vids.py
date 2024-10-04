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
from keras_unet_collection.models import unet_2d, transunet_2d, u2net_2d, att_unet_2d, unet_plus_2d, r2_unet_2d, resunet_a_2d
from yolov4.get_bounding_boxes import YoloModel
import torchvision as tv
from sparse_coding_torch.onsd.video_loader import get_yolo_region_onsd
import torch
import math
import csv

from scipy.ndimage import rotate, shift

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def retrieve_outliers(file_loc, fraction_to_discard, pos_neg_cutoff):
    all_rows = []
    with open(file_loc, 'r') as csv_f:
        reader = csv.DictReader(csv_f)
        
        for row in reader:
            all_rows.append((row['file'], int(row['width']), row['label']))
            
    all_rows = [row for row in all_rows if (row[1] >= pos_neg_cutoff and row[2] == 'Negatives') or (row[1] < pos_neg_cutoff and row[2] == 'Positives')]
            
    all_rows_asc = sorted(all_rows, key=lambda tup: tup[1])
    all_rows_asc = [tup[0] for tup in all_rows_asc]
    
    fraction_to_discard = fraction_to_discard // 2
    
    num_to_discard = int(len(all_rows_asc) * fraction_to_discard)
    
    top = all_rows_asc[-num_to_discard:]
    bottom = all_rows_asc[:num_to_discard]
    
    return top + bottom

def load_videos(input_dir, nerve_dir, pos_neg_cutoff, fraction_to_discard=None):
    nerve_img_paths = sorted(
    [
        fname
        for fname in os.listdir(nerve_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
    )
    
    if fraction_to_discard is not None:
        nerve_img_paths = [fname for fname in nerve_img_paths if fname not in retrieve_outliers('segmentation/more_slices/outliers.csv', fraction_to_discard, pos_neg_cutoff)]

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png") and not fname.startswith(".") and fname in nerve_img_paths
        ]
    )
    
    nerve_img_paths = [os.path.join(nerve_dir, fname) for fname in nerve_img_paths]

    assert len(input_img_paths) == len(nerve_img_paths)

    print("Number of training samples:", len(input_img_paths))

    input_data = []
    for input_path, nerve_path in zip(input_img_paths, nerve_img_paths):
        input_data.append((input_path, nerve_path))
        
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

def get_test_videos():
    all_vids = glob.glob(os.path.join('/shared_data/bamc_onsd_test_data', '*', '*.mp4'))
    
    out_vids = []
    out_lbls = []
    
    for vid in all_vids:
        vid_name = vid.split('/')[-1][:-4]
        txt_label = vid.split('/')[-2]

        out_vids.append(vid)
        out_lbls.append(txt_label)
            
    return out_vids, out_lbls

def get_participants(video_path, input_data):
    all_vids = glob.glob(os.path.join(video_path, '*', '*', '*.mp4'))

    participant_to_data = {}

    for vid in all_vids:
        vid_name = vid.split('/')[-1][:-4]
        participant = vid.split('/')[-2]
        txt_label = vid.split('/')[-3]

        for frame in input_data:
            frame_name = frame[0].split('/')[-1].split(' ')[1][:-2]

            if frame_name != participant:
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

def make_numpy_arrays(split_participants, participant_to_data, img_size):
    all_x = []
    all_yolo = []
    all_eye = []
    all_nerve = []
    all_txt = []
    
    for participant in split_participants:
        for x, nerve, txt_label in participant_to_data[participant]:
            yolo = cv2.resize(cv2.imread(x), (img_size[1], img_size[0]))
            x = cv2.resize(cv2.imread(x, cv2.IMREAD_GRAYSCALE), (img_size[1], img_size[0]))
            nerve = cv2.resize(cv2.imread(nerve, cv2.IMREAD_GRAYSCALE), (img_size[1], img_size[0]))
            for i in range(nerve.shape[0]):
                for j in range(nerve.shape[1]):
                    if nerve[i, j] == 255:
                        nerve[i, j] = 1.0
                    else:
                        nerve[i, j] = 0.0

            all_x.append(x)
            all_yolo.append(yolo)
            all_nerve.append(nerve)
            all_txt.append(txt_label)
            
    return np.expand_dims(np.stack(all_x), axis=-1), np.stack(all_yolo), np.stack(all_nerve), all_txt

def display_mask_test(model, input_mask):
    """Quick utility to display a model's prediction."""
    test_pred = model.predict(np.expand_dims(np.expand_dims(input_mask, axis=0), axis=-1), verbose=False)[0]
    mask = np.argmax(test_pred, axis=-1)
    mask = np.expand_dims(mask, axis=-1) * 255
    
    return mask


def get_obj_coordinates_yolo(yolo_model, frame):
    orig_height = frame.shape[0]
    orig_width = frame.shape[1]
    
    bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)
    
    eye_bounding_box = (None, 0.0)
    nerve_bounding_box = (None, 0.0)
    
    for bb, class_pred, score in zip(bounding_boxes, classes, scores):
        if class_pred == 0 and score > nerve_bounding_box[1]:
            nerve_bounding_box = (bb, score)
        elif class_pred == 1 and score > eye_bounding_box[1]:
            eye_bounding_box = (bb, score)
    
    eye_bounding_box = eye_bounding_box[0]
    nerve_bounding_box = nerve_bounding_box[0]
    
    if eye_bounding_box is None or nerve_bounding_box is None:
        return None, None, None, None, None
    
    nerve_center_x = round((nerve_bounding_box[2] + nerve_bounding_box[0]) / 2 * orig_width)
    nerve_center_y = round((nerve_bounding_box[3] + nerve_bounding_box[1]) / 2 * orig_height)
    
    eye_center_x = round((eye_bounding_box[2] + eye_bounding_box[0]) / 2 * orig_width)
    eye_center_y = round((eye_bounding_box[3] + eye_bounding_box[1]) / 2 * orig_height)
    
    dist_to_bottom = round((eye_bounding_box[3] * orig_height) - eye_center_y)
            
    return nerve_center_x, nerve_center_y, eye_center_x, eye_center_y, dist_to_bottom

def get_line(yolo_model, input_frame):
    nerve_center_x, nerve_center_y, eye_center_x, eye_center_y, dist_to_bottom = get_obj_coordinates_yolo(yolo_model, input_frame)
    
    if dist_to_bottom is None:
        return None, None, (None, None), (None, None), None
    
    if eye_center_x != nerve_center_x:
        m = (eye_center_y - nerve_center_y) / (eye_center_x - nerve_center_x)
    else:
        return None, None, (eye_center_x, eye_center_y), (nerve_center_x, nerve_center_y), dist_to_bottom
    
    b = (-1 * (m * nerve_center_x)) + nerve_center_y
    
    assert eye_center_y == round(m*eye_center_x + b, 2)
    assert nerve_center_y == round(m*nerve_center_x + b, 2)
    
    return m, b, (eye_center_x, eye_center_y), (nerve_center_x, nerve_center_y), dist_to_bottom

def get_nerve_slice(yolo_model, input_frames, yolo_frames, nerve_frames, nerve_size):
    all_nerve_slices = []
    all_nerve_masks = []
    
    for input_frame, yolo_frame, nerve_frame in zip(input_frames, yolo_frames, nerve_frames):
        m, b, (eye_center_x, eye_center_y), (nerve_center_x, nerve_center_y), dist_to_bottom = get_line(yolo_model, yolo_frame)

        target_length = (65/1080)*input_frame.shape[0] + dist_to_bottom

        if m is not None and b is not None:
            nerve_start = eye_center_y
            for i in range(round(eye_center_y) + 1, input_frame.shape[0]):
                x_val = round((i - b) / m)
                distance_from_eye = math.sqrt((x_val - eye_center_x)**2 + (i - eye_center_y)**2)
                if distance_from_eye > target_length:
                    nerve_start = i
                    break

            nerve_measure_y = nerve_start
            nerve_measure_x = (nerve_measure_y - b) / m
        else:   
            nerve_measure_y = round(eye_center_y + target_length)
            nerve_measure_x = eye_center_x

        shift_y = (input_frame.shape[0] // 2) - nerve_measure_y
        shift_x = (input_frame.shape[1] // 2) - nerve_measure_x

        shifted_image = shift(input_frame, shift=(shift_y, shift_x, 0))
        shifted_nerve = shift(nerve_frame, shift=(shift_y, shift_x))

        if m is not None:
            angle = 90 + np.degrees(np.arctan(m))

            rotated_image = rotate(shifted_image, angle=angle)
            rotated_nerve = rotate(shifted_nerve, angle=angle)
        else:
            rotated_image = shifted_image
            rotated_nerve = shifted_nerve

        center_y = rotated_image.shape[0] // 2
        center_x = rotated_image.shape[1] // 2

        crop_y = nerve_size[0] // 2
        crop_x = nerve_size[1] // 2

        cropped_image = rotated_image[center_y-crop_y:center_y+crop_y, center_x-crop_x:center_x+crop_x, :]
        cropped_nerve = rotated_nerve[center_y-crop_y:center_y+crop_y, center_x-crop_x:center_x+crop_x]
        
        all_nerve_slices.append(cropped_image)
        all_nerve_masks.append(cropped_nerve)
        
    all_nerve_slices = np.stack(all_nerve_slices)
    all_nerve_masks = np.stack(all_nerve_masks)
    
    return np.expand_dims(all_nerve_slices, axis=-1), all_nerve_masks

def get_nerve_slice_test_time(yolo_model, input_frames, yolo_frames, nerve_size):
    all_nerve_slices = []
    
    for input_frame, yolo_frame in zip(input_frames, yolo_frames):
        m, b, (eye_center_x, eye_center_y), (nerve_center_x, nerve_center_y), dist_to_bottom = get_line(yolo_model, yolo_frame)
        
        if dist_to_bottom is None:
            continue

        target_length = (65/1080)*input_frame.shape[0] + dist_to_bottom

        if m is not None and b is not None:
            nerve_start = eye_center_y
            for i in range(round(eye_center_y) + 1, input_frame.shape[0]):
                x_val = round((i - b) / m)
                distance_from_eye = math.sqrt((x_val - eye_center_x)**2 + (i - eye_center_y)**2)
                if distance_from_eye > target_length:
                    nerve_start = i
                    break

            nerve_measure_y = nerve_start
            nerve_measure_x = (nerve_measure_y - b) / m
        else:   
            nerve_measure_y = round(eye_center_y + target_length)
            nerve_measure_x = eye_center_x

        shift_y = (input_frame.shape[0] // 2) - nerve_measure_y
        shift_x = (input_frame.shape[1] // 2) - nerve_measure_x

        shifted_image = shift(input_frame, shift=(shift_y, shift_x, 0))

        if m is not None:
            angle = 90 + np.degrees(np.arctan(m))

            rotated_image = rotate(shifted_image, angle=angle)
        else:
            rotated_image = shifted_image

        center_y = rotated_image.shape[0] // 2
        center_x = rotated_image.shape[1] // 2

        crop_y = nerve_size[0] // 2
        crop_x = nerve_size[1] // 2

        cropped_image = rotated_image[center_y-crop_y:center_y+crop_y, center_x-crop_x:center_x+crop_x, :]
        
        all_nerve_slices.append(cropped_image)
        
    if not all_nerve_slices:
        return None
        
    all_nerve_slices = np.stack(all_nerve_slices)
    
    return np.expand_dims(all_nerve_slices, axis=-1)

def get_width_measurement(nerve_frame):
    nerve_center_y = nerve_frame.shape[0] // 2
    nerve_center_x = nerve_frame.shape[1] // 2
    
    left_boundary = nerve_center_x
    for j in range(round(nerve_center_x) - 1, 0, -1):
        if nerve_frame[nerve_center_y, j] != 1.0:
            left_boundary = j + 1
            break
    
    right_boundary = nerve_center_x
    for j in range(round(nerve_center_x) + 1, nerve_frame.shape[1]):
        if nerve_frame[nerve_center_y, j] != 1.0:
            right_boundary = j - 1
            break
    
    width = right_boundary - left_boundary

    return width

def get_width_predictions(yolo_model, nerve_model, X, yolo, nerve, lbls, pos_neg_cutoff, nerve_size):
    all_widths = []
    pred_widths = []
    class_preds = []
    gt_mask_preds = []
    class_gt = []
    
    nerve_slices, nerve_masks = get_nerve_slice(yolo_model, X, yolo, nerve, nerve_size)
    
    nerve_pred = np.argmax(nerve_model.predict(nerve_slices, verbose=False), axis=-1)
    
    for nerve_p, nerve_gt, lbl in zip(nerve_pred, nerve_masks, lbls):
        width = get_width_measurement(nerve_gt)

        pred_width = get_width_measurement(nerve_p)

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

def run_full_eval(nerve_model, yolo_model, videos, lbl, pos_neg_cutoff, img_size, nerve_size):
    pred_widths = []
    class_preds = []
    class_gt = []
    
    transforms = tv.transforms.Compose(
    [tv.transforms.Grayscale(1)
    ])
    
    resize = tv.transforms.Resize(img_size)
    
    all_slices = []
    for video_path in videos:
        vc = tv.io.read_video(video_path)[0].permute(3, 0, 1, 2)

        all_frames = [resize(vc[:, j, :, :]) for j in range(0, vc.size(1), 10)]
        
        all_yolo = np.stack([frame.numpy().swapaxes(0,1).swapaxes(1,2) for frame in all_frames])
        
        all_frames = np.stack([transforms(frame).numpy().swapaxes(0,1).swapaxes(1,2) for frame in all_frames])
        
        slices = get_nerve_slice_test_time(yolo_model, all_frames, all_yolo, nerve_size)
        
        if slices is None:
            continue

        all_slices.append(slices)
        
    if not all_slices:
        if lbl == 'Positives':
            class_gt.append(1)
        else:
            class_gt.append(0)
            
        class_preds.append(0)
    else:
        all_slices = np.concatenate(all_slices)

        pred = np.argmax(nerve_model.predict(all_slices, verbose=False), axis=-1)

        for p in pred:
            pred_width = get_width_measurement(p)

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

def run_full_eval_test_set(nerve_model, yolo_model, videos, lbls, pos_neg_cutoff, img_size, nerve_size):
    class_preds = []
    class_gt = []
    
    transforms = tv.transforms.Compose(
    [tv.transforms.Grayscale(1)
    ])
    
    resize = tv.transforms.Resize(img_size)

    for video_path, lbl in zip(videos, lbls):
        pred_widths = []
        
        vc = tv.io.read_video(video_path)[0].permute(3, 0, 1, 2)

        all_frames = [resize(vc[:, j, :, :]) for j in range(0, vc.size(1), 10)]
        
        all_yolo = np.stack([frame.numpy().swapaxes(0,1).swapaxes(1,2) for frame in all_frames])
        
        all_frames = np.stack([transforms(frame).numpy().swapaxes(0,1).swapaxes(1,2) for frame in all_frames])
        
#         cv2.imwrite('onsd_validation/' + video_path.split('/')[-1][:-4] + '_frame.png', all_frames[0])
        
        slices = get_nerve_slice_test_time(yolo_model, all_frames, all_yolo, nerve_size)
        
#         cv2.imwrite('onsd_validation/' + video_path.split('/')[-1][:-4] + '_slice.png', np.squeeze(slices[0], axis=-1))
        
        if slices is None:
            print('Not found')
            class_preds.append(0)

            if lbl == 'Positives':
                class_gt.append(1)
            else:
                class_gt.append(0)
            continue

        slices = np.stack(slices)

        pred = np.argmax(nerve_model.predict(slices, verbose=False), axis=-1)

        for p in pred:
            pred_width = get_width_measurement(p)
            if pred_width == 0:
                continue

            pred_widths.append(pred_width)
            
        print(pred_widths)
        if not pred_widths:
            pred_widths.append(0)

        pred_width = np.average(pred_widths)
        
        print(pred_width)

        if pred_width >= pos_neg_cutoff:
            class_preds.append(1)
        else:
            class_preds.append(0)

        if lbl == 'Positives':
            class_gt.append(1)
        else:
            class_gt.append(0)

    return np.array(class_preds), np.array(class_gt)


random.seed(321534)
np.random.seed(321534)
tf.random.set_seed(321534)

output_dir = 'sparse_coding_torch/unet_output/unet_nerve_all_vids'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_path = "/shared_data/bamc_onsd_data/revised_extended_onsd_data"

input_dir = 'segmentation/more_slices/raw_frames'

nerve_dir = 'segmentation/more_slices/nerve_segmentation'

yolo_model = YoloModel('onsd')

img_size = (416, 416)
nerve_size = (16, 128)
batch_size = 12
# pos_neg_cutoff = (102 / 1080) * img_size[0]
pos_neg_cutoff = 46

input_data = load_videos(input_dir, nerve_dir, pos_neg_cutoff, None)

participant_to_data = get_participants(video_path, input_data) 

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
    train_X, train_yolo, train_nerve, train_txt = make_numpy_arrays(train_participants, participant_to_data, img_size)
    test_X, test_yolo, test_nerve, test_txt = make_numpy_arrays(test_participants, participant_to_data, img_size)
    print(test_txt)

    keras.backend.clear_session()
    
    nerve_inputs = keras.Input(shape=(None, None, 1))
    
    data_preprocessing = keras.Sequential([#keras.layers.RandomFlip('horizontal_and_vertical'),
                                           #keras.layers.RandomBrightness(0.10),
#                                            keras.layers.RandomContrast(0.01),
#                                            keras.layers.RandomRotation(0.10)
    ])(nerve_inputs)

    nerve_outputs = unet_2d((None, None, 1), [64, 128, 256], n_labels=2,
                      stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')(data_preprocessing)
#     nerve_outputs = att_unet_2d((None, None, 1), [64, 128, 256], n_labels=2,
#                            stack_num_down=2, stack_num_up=2,
#                            activation='ReLU', atten_activation='ReLU', attention='add', output_activation='Softmax', 
#                            batch_norm=True, pool=False, unpool='bilinear', name='attunet')(data_preprocessing)
#     nerve_outputs = unet_plus_2d((None, None, 1), [64, 128, 256], n_labels=2,
#                             stack_num_down=2, stack_num_up=2,
#                             activation='LeakyReLU', output_activation='Softmax', 
#                             batch_norm=False, pool='max', unpool=False, deep_supervision=True, name='xnet')(data_preprocessing)
#     nerve_outputs = r2_unet_2d((None, None, 1), [64, 128, 256], n_labels=2,
#                           stack_num_down=2, stack_num_up=1, recur_num=2,
#                           activation='ReLU', output_activation='Softmax', 
#                           batch_norm=True, pool='max', unpool='bilinear', name='r2unet')(data_preprocessing)
#     nerve_outputs = resunet_a_2d(nerve_size + (1,), [32, 64, 128, 256], 
#                             dilation_num=[1, 3, 15, 31], 
#                             n_labels=2, aspp_num_down=256, aspp_num_up=128, 
#                             activation='ReLU', output_activation='Softmax', 
#                             batch_norm=True, pool=False, unpool='nearest', name='resunet')(data_preprocessing)
#     nerve_outputs = u2net_2d((None, None, 1), n_labels=2, 
#                         filter_num_down=[64, 128, 256],
#                         activation='ReLU', output_activation='Softmax', 
#                         batch_norm=True, pool=False, unpool=False, deep_supervision=True, name='u2net')(data_preprocessing)
#     nerve_model = swin_unet_2d(nerve_size + (1,), filter_num_begin=64, n_labels=2, depth=4, stack_num_down=2, stack_num_up=2, 
#                             patch_size=(2, 4), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
#                             output_activation='Softmax', shift_window=True, name='swin_unet')
#     nerve_model = transunet_2d(nerve_size + (1,), filter_num=[64, 128, 256], n_labels=12, stack_num_down=2, stack_num_up=2,
#                                 embed_dim=768, num_mlp=3072, num_heads=12, num_transformer=12,
#                                 activation='ReLU', mlp_activation='GELU', output_activation='Softmax', 
#                                 batch_norm=True, pool=True, unpool='bilinear', name='transunet')
    
    nerve_model = keras.Model(nerve_inputs, nerve_outputs)

    nerve_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss="sparse_categorical_crossentropy")
#     nerve_callbacks = [
#         keras.callbacks.ModelCheckpoint(os.path.join(output_dir, "best_nerve_model_{}.h5".format(i_fold)), save_best_only=True, save_weights_only=True)
#     ]

    # Train the model, doing validation at the end of each epoch.
#     if os.path.exists(os.path.join(output_dir, "best_unet_model_{}.h5".format(i_fold))):
#         model.load_weights(os.path.join(output_dir, "best_unet_model_{}.h5".format(i_fold)))
#     else:
    epochs = 200
    
    train_slices, train_masks = get_nerve_slice(yolo_model, train_X, train_yolo, train_nerve, nerve_size)

    nerve_model.fit(train_slices, train_masks, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
    
    nerve_weights = nerve_model.get_weights()
    
    nerve_model = unet_2d((None, None, 1), [64, 128, 256], n_labels=2,
                      stack_num_down=2, stack_num_up=1,
                      activation='GELU', output_activation='Softmax', 
                      batch_norm=True, pool='max', unpool='nearest', name='unet')
    nerve_model.set_weights(nerve_weights)

    final_width_train, final_pred_width_train, class_gt_mask_train, class_pred_train, class_gt_train = get_width_predictions(yolo_model, nerve_model, train_X, train_yolo, train_nerve, train_txt, pos_neg_cutoff, nerve_size)
    
    train_average_width_difference = np.average(np.abs(np.array(final_width_train) - np.array(final_pred_width_train)))
    
    train_gt_mask_class_score = accuracy_score(class_gt_train, class_gt_mask_train)
    
    train_pred_mask_class_score = accuracy_score(class_gt_train, class_pred_train)
    
    print('Training results fold {}: average width difference={:.2f}, ground truth mask classification={:.2f}, predicted mask classification={:.2f}'.format(i_fold, train_average_width_difference, train_gt_mask_class_score, train_pred_mask_class_score))

#     videos = get_videos(participants[i_fold])
#     lbl = test_txt[0]
#     test_pred, test_gt = run_full_eval(nerve_model, yolo_model, videos, lbl, pos_neg_cutoff, img_size, nerve_size)
    
#     test_pred_mask_class_score = accuracy_score(test_gt, test_pred)
    
#     pred_video_pred = np.array([np.round(np.average(test_pred))])
    
#     if test_txt[0] == 'Positives':
#         video_class = np.array([1])
#     else:
#         video_class = np.array([0])
    
#     test_video_pred_mask_score = accuracy_score(video_class, pred_video_pred)
    
    videos, lbls = get_test_videos()
    test_pred, test_gt = run_full_eval_test_set(nerve_model, yolo_model, videos, lbls, pos_neg_cutoff, img_size, nerve_size)
    
    print(test_pred)
    print(test_gt)
    
    test_video_pred_mask_score = accuracy_score(test_gt, test_pred)
    
    print('Testing results fold {}:  predicted mask video-level classification={:.2f}'.format(i_fold, test_video_pred_mask_score))
    
    all_train_frame_pred.append(class_pred_train)
    all_train_frame_gt.append(class_gt_train)
    
#     all_test_frame_pred.append(test_pred)
#     all_test_frame_gt.append(test_gt)
    all_test_video_pred.append(test_pred)
    all_test_video_gt.append(test_gt)
    
    i_fold += 1
    
all_train_frame_pred = np.concatenate(all_train_frame_pred)
all_train_frame_gt = np.concatenate(all_train_frame_gt)

# all_test_frame_pred = np.concatenate(all_test_frame_pred)
all_test_video_pred = np.concatenate(all_test_video_pred)

final_train_frame_acc = accuracy_score(all_train_frame_gt, all_train_frame_pred)
# final_test_frame_acc = accuracy_score(all_test_frame_gt, all_test_frame_pred)
final_test_video_acc = accuracy_score(all_test_video_gt, all_test_video_pred)

print('Final results: Train frame-level classification={:.2f}, Test frame-level classification={:.2f}, Test video-level classification={:.2f}'.format(final_train_frame_acc, final_test_frame_acc, final_test_video_acc))