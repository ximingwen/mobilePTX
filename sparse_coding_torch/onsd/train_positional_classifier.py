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
from unet_models import ONSDPositionalConv
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sparse_coding_torch.sparse_model import SparseCode
from tqdm import tqdm

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
    all_y1 = []
    all_y2 = []
    all_txt = []
    
    for participant in split_participants:
        for x, y, txt_label in participant_to_data[participant]:
            x = cv2.resize(cv2.imread(x, cv2.IMREAD_GRAYSCALE), (img_size[1], img_size[0]))
            y = cv2.resize(cv2.imread(y, cv2.IMREAD_GRAYSCALE), (img_size[1], img_size[0]))
            y1 = float('inf')
            y2 = float('-inf')
            for i in range(y.shape[0]//2, (y.shape[0]//2)+5):
                for j in range(10,y.shape[1]-10):
                    if y[i, j] == 255:
                        y1 = min(y1, j)
                        y2 = max(y2, j)
                        
            if y1 == float('inf') or y2 == float('-inf'):
                print(participant)
                raise Exception

            all_x.append(x)
            all_y1.append(y1)
            all_y2.append(y2)
            all_txt.append(txt_label)
            
    return np.stack(all_x), np.stack(all_y1), np.stack(all_y2), all_txt

def get_width_predictions(model, sparse_model, recon_model, X, y1, y2, lbls, pos_neg_cutoff):
    all_widths = []
    pred_widths = []
    class_preds = []
    gt_mask_preds = []
    class_gt = []
    
    activations = tf.stop_gradient(sparse_model([np.expand_dims(X, axis=1), tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
    
    y1_pred, y2_pred = model.predict(activations, verbose=False)
    
    for x1_pred, x2_pred, x1, x2, lbl in zip(y1_pred, y2_pred, y1, y2, lbls):
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
    

random.seed(321534)
np.random.seed(321534)
tf.random.set_seed(321534)

output_dir = 'sparse_coding_torch/positional_output/psotional_3'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

video_path = "/shared_data/bamc_onsd_data/revised_extended_onsd_data"

input_dir = "segmentation/segmentation_12_15/labeled_frames/"
target_dir = "segmentation/segmentation_12_15/labeled_frames/segmentation/"

img_size = (160, 160)
batch_size = 12
filter_size = 5
pos_neg_cutoff = 74
kernel_height = 15
kernel_width = 15
num_kernels = 32
sparse_checkpoint = 'sparse_coding_torch/output/onsd_frame_level_32/best_sparse.pt'

inputs = keras.Input(shape=(1, img_size[0], img_size[1], 1))
        
filter_inputs = keras.Input(shape=(1, kernel_height, kernel_width, 1, num_kernels), dtype='float32')

output = SparseCode(batch_size=batch_size, image_height=img_size[0], image_width=img_size[1], clip_depth=1, in_channels=1, out_channels=num_kernels, kernel_height=kernel_height, kernel_width=kernel_width, kernel_depth=1, stride=1, lam=0.05, activation_lr=1e-2, max_activation_iter=200, run_2d=False)(inputs, filter_inputs)

sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
recon_model = keras.models.load_model(sparse_checkpoint)

input_data = load_videos(input_dir, target_dir)

participant_to_data = get_participants(video_path) 

splits, participants = create_splits(participant_to_data)

all_train_frame_pred = []
all_train_frame_gt = []

all_test_frame_pred = []
all_test_frame_gt = []
all_test_video_pred = []
all_test_video_gt = []

i_fold = 0
for train_idx, test_idx in splits:
    train_participants = [p for i, p in enumerate(participants) if i in train_idx]
    test_participants = [p for i, p in enumerate(participants) if i in test_idx]
    
    assert len(set(train_participants).intersection(set(test_participants))) == 0

    # Instantiate data Sequences for each split
    train_X, train_y1, train_y2, train_txt = make_numpy_arrays(train_participants)
    test_X, test_y1, test_y2, test_txt = make_numpy_arrays(test_participants)

    keras.backend.clear_session()

    # Build model
    inputs = keras.Input(shape=output.shape[1:])
    
    outputs = ONSDPositionalConv()(inputs)
    
    classifier_model = keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = keras.optimizers.Adam(learning_rate=1e-5)
    
    criterion = keras.losses.MeanSquaredError()

    # Train the model, doing validation at the end of each epoch.
    if os.path.exists(os.path.join(output_dir, "best_positional_model_{}.h5".format(i_fold))):
        model = keras.models.load_model(os.path.join(output_dir, "best_positional_model_{}.h5".format(i_fold)))
    else:
        epochs = 10
        
        train_tf = tf.data.Dataset.from_tensor_slices((train_X, train_y1, train_y2))

        for _ in tqdm(range(epochs)):
            for images, y1, y2 in train_tf.shuffle(len(train_tf)).batch(batch_size):
                images = tf.expand_dims(images, axis=1)
                
                activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
                        
                with tf.GradientTape() as tape:
                    y1_pred, y2_pred = classifier_model(activations)
                    loss = criterion(y1, y1_pred) + criterion(y2, y2_pred)

                    gradients = tape.gradient(loss, classifier_model.trainable_weights)

                    optimizer.apply_gradients(zip(gradients, classifier_model.trainable_weights))
             
    final_width_train, final_pred_width_train, class_gt_mask_train, class_pred_train, class_gt_train = get_width_predictions(classifier_model, sparse_model, recon_model, train_X, train_y1, train_y2, train_txt, pos_neg_cutoff)
    
    train_average_width_difference = np.average(np.abs(np.array(final_width_train) - np.array(final_pred_width_train)))
    
    train_gt_mask_class_score = accuracy_score(class_gt_train, class_gt_mask_train)
    
    train_pred_mask_class_score = accuracy_score(class_gt_train, class_pred_train)
    
    print('Training results fold {}: average width difference={:.2f}, ground truth mask classification={:.2f}, predicted mask classification={:.2f}'.format(i_fold, train_average_width_difference, train_gt_mask_class_score, train_pred_mask_class_score))

    final_width_test, final_pred_width_test, class_gt_mask_test, class_pred_test, class_gt_test = get_width_predictions(classifier_model, sparse_model, recon_model, test_X, test_y1, test_y2, test_txt, pos_neg_cutoff)
    
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

print('Final results: Train frame-level classification={:.2f}, Test frame-level classification={:.2f}, Test video-level classification={:.2f}'.format(final_train_frame_acc, final_test_frame_acc, final_test_video_acc))