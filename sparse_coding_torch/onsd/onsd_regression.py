import math
from tqdm import tqdm
import glob
from os.path import join, abspath
import random
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
import tensorflow as tf
from yolov4.get_bounding_boxes import YoloModel
import torchvision
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    
def get_width_data(yolo_model, input_videos):
    all_data = []
    for label, path, vid_f in tqdm(input_videos):
        vc = torchvision.io.read_video(path)[0].permute(3, 0, 1, 2)

        orig_height = vc.size(2)
        orig_width = vc.size(3)

        obj_bb = []

        for i in range(vc.size(1) - 1, vc.size(1) - 40, -1):
            frame = vc[:, i, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()

            bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

            obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==0]

            if len(obj_bb) > 0:
                obj_bb = obj_bb[0]
                break

        if len(obj_bb) == 0:
            continue

        obj_width = round((obj_bb[2] + obj_bb[0]) / 2)
        
        if label == 'Positives':
            label = 1.0
        elif label == 'Negatives':
            label = 0.0
        else:
            raise Exception('Bad label')

        all_data.append((obj_width, label, path))
        
    return all_data

def get_splits(videos):
    random.shuffle(videos)
    
    gss = GroupKFold(n_splits=5)

    groups = [vid[2].split('/')[-2] for vid in videos]

    targets = [vid[2].split('/')[-3] for vid in videos]

    return gss.split(np.arange(len(targets)), targets, groups)

yolo_model = YoloModel('onsd')
video_path = "/shared_data/bamc_onsd_data/revised_extended_onsd_data/"

print('Beginning...')

labels = [name for name in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, name))]

videos = []
for label in labels:
    videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])

print('Making splits...')    

splits = get_splits(videos)

preds = []
gt = []

split_count = 0

for train_idx, test_idx in splits:
    print('On split {}...'.format(split_count))

    print('Processing data...')
    train_videos = [ex for i, ex in enumerate(videos) if i in train_idx]
    test_videos = [ex for i, ex in enumerate(videos) if i in test_idx]
    
    train_data = get_width_data(yolo_model, train_videos)
    
    test_data = get_width_data(yolo_model, test_videos)

    train_X = np.array([train_data[i][0] for i in range(len(train_data))]).reshape(-1, 1)
    test_X = np.array([test_data[i][0] for i in range(len(test_data))]).reshape(-1, 1)
    
    train_Y = np.array([train_data[i][1] for i in range(len(train_data))]).reshape(-1, 1)
    test_Y = np.array([test_data[i][1] for i in range(len(test_data))]).reshape(-1, 1)
    
    print('Training models...')
    
    clf = LogisticRegression().fit(train_X, train_Y)
    score = clf.score(test_X, test_Y)
    
    print('Accuracy: {:.2f}'.format(score))
    
    for label, path, _ in tqdm(test_videos):
        vc = torchvision.io.read_video(path)[0].permute(3, 0, 1, 2)

        orig_height = vc.size(2)
        orig_width = vc.size(3)
        
        if label == 'Positives':
            gt.append(1.0)
        elif label == 'Negatives':
            gt.append(0.0)
        else:
            raise Exception('Bad label')

        obj_bb = []

        for i in range(vc.size(1) - 1, vc.size(1) - 40, -1):
            frame = vc[:, i, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()

            bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

            obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==0]

            if len(obj_bb) > 0:
                obj_bb = obj_bb[0]
                break

        if len(obj_bb) > 0:
            width = round((obj_bb[2] + obj_bb[0]) / 2)
            
            pred = clf.predict(np.array([width]).reshape(1, -1))[0]
            preds.append(pred)
        else:
            preds.append(0.0)
            continue
            
    split_count += 1
    
    print('Current results...')
    overall_true = np.array(gt)
    overall_pred = np.array(preds)

    f1 = f1_score(overall_true, overall_pred, average='macro')
    acc = accuracy_score(overall_true, overall_pred)

    print("Current accuracy={:.2f}, f1={:.2f}".format(acc, f1))
    
overall_true = np.array(gt)
overall_pred = np.array(preds)

final_f1 = f1_score(overall_true, overall_pred, average='macro')
final_acc = accuracy_score(overall_true, overall_pred)
final_conf = confusion_matrix(overall_true, overall_pred)

print("Final accuracy={:.2f}, f1={:.2f}".format(final_acc, final_f1))
print(final_conf)