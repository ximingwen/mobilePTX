from sparse_coding_torch.pnb.video_loader import classify_nerve_is_right, load_pnb_region_labels
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
import pickle as pkl

def get_distance_data_sme_labels(yolo_model, input_videos, yolo_class):
    region_labels = load_pnb_region_labels('sme_region_labels.csv')
    
    all_data = []
    for label_str, path, vid_f in tqdm(input_videos):
        vc = torchvision.io.read_video(path)[0].permute(3, 0, 1, 2)
        is_right = classify_nerve_is_right(yolo_model, vc)
        
        orig_height = vc.size(2)
        orig_width = vc.size(3)
        
        if label_str == 'Positives':
            label = 1.0
        elif label_str == 'Negatives':
            label = 0.0
        
        person_idx = path.split('/')[-1].split(' ')[1]
        
        if label == 1.0 and person_idx in region_labels:
            negative_regions, positive_regions = region_labels[person_idx]
            for sub_region in negative_regions.split(','):
                sub_region = sub_region.split('-')
                start_loc = int(sub_region[0])
                end_loc = int(sub_region[1]) + 1
                for j in range(start_loc, end_loc, 1):
                    frame = vc[:, j, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()
                    
                    bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

                    obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==yolo_class]
                    needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]
                    
                    if len(obj_bb) == 0 or len(needle_bb) == 0:
                        continue
                        
                    obj_bb = obj_bb[0]
                    needle_bb = needle_bb[0]
                    
                    obj_x = round((obj_bb[2] + obj_bb[0]) / 2 * orig_width)
                    obj_y = round((obj_bb[3] + obj_bb[1]) / 2 * orig_height)

                    needle_x = needle_bb[2] * orig_width
                    needle_y = needle_bb[3] * orig_height

                    if not is_right:
                        needle_x = needle_bb[0] * orig_width

                    all_data.append((math.sqrt((obj_x - needle_x)**2 + (obj_y - needle_y)**2), 0.0, path))
                    
            if positive_regions:
                for sub_region in positive_regions.split(','):
                    sub_region = sub_region.split('-')
#                                 start_loc = int(sub_region[0]) + 15
                    start_loc = int(sub_region[0])
                    if len(sub_region) == 1 and vc.size(1) > start_loc:
                        frame = vc[:, start_loc, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()
                    
                        bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

                        obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==yolo_class]
                        needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

                        if len(obj_bb) == 0 or len(needle_bb) == 0:
                            continue

                        obj_bb = obj_bb[0]
                        needle_bb = needle_bb[0]

                        obj_x = round((obj_bb[2] + obj_bb[0]) / 2 * orig_width)
                        obj_y = round((obj_bb[3] + obj_bb[1]) / 2 * orig_height)

                        needle_x = needle_bb[2] * orig_width
                        needle_y = needle_bb[3] * orig_height

                        if not is_right:
                            needle_x = needle_bb[0] * orig_width

                        all_data.append((math.sqrt((obj_x - needle_x)**2 + (obj_y - needle_y)**2), 1.0, path))
                            
                    elif vc.size(1) > start_loc:
                        end_loc = sub_region[1]
                        if end_loc.strip().lower() == 'end':
                            end_loc = vc.size(1)
                        else:
                            end_loc = int(end_loc)
                        for j in range(start_loc, end_loc, 1):
                            frame = vc[:, j, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()
                    
                            bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

                            obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==yolo_class]
                            needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

                            if len(obj_bb) == 0 or len(needle_bb) == 0:
                                continue

                            obj_bb = obj_bb[0]
                            needle_bb = needle_bb[0]

                            obj_x = round((obj_bb[2] + obj_bb[0]) / 2 * orig_width)
                            obj_y = round((obj_bb[3] + obj_bb[1]) / 2 * orig_height)

                            needle_x = needle_bb[2] * orig_width
                            needle_y = needle_bb[3] * orig_height

                            if not is_right:
                                needle_x = needle_bb[0] * orig_width

                            all_data.append((math.sqrt((obj_x - needle_x)**2 + (obj_y - needle_y)**2), 1.0, path))
                            
        elif label == 1.0:
            frames = []
            for k in range(vc.size(1) - 1, vc.size(1) - 40, -1):
                frame = vc[:, k, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()
                    
                bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

                obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==yolo_class]
                needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

                if len(obj_bb) == 0 or len(needle_bb) == 0:
                    continue

                obj_bb = obj_bb[0]
                needle_bb = needle_bb[0]

                obj_x = round((obj_bb[2] + obj_bb[0]) / 2 * orig_width)
                obj_y = round((obj_bb[3] + obj_bb[1]) / 2 * orig_height)

                needle_x = needle_bb[2] * orig_width
                needle_y = needle_bb[3] * orig_height

                if not is_right:
                    needle_x = needle_bb[0] * orig_width

                all_data.append((math.sqrt((obj_x - needle_x)**2 + (obj_y - needle_y)**2), 1.0, path))
            
        elif label == 0.0:
            for j in range(0, vc.size(1), 1):
                frame = vc[:, j, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()
                    
                bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

                obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==yolo_class]
                needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

                if len(obj_bb) == 0 or len(needle_bb) == 0:
                    continue

                obj_bb = obj_bb[0]
                needle_bb = needle_bb[0]

                obj_x = round((obj_bb[2] + obj_bb[0]) / 2 * orig_width)
                obj_y = round((obj_bb[3] + obj_bb[1]) / 2 * orig_height)

                needle_x = needle_bb[2] * orig_width
                needle_y = needle_bb[3] * orig_height

                if not is_right:
                    needle_x = needle_bb[0] * orig_width

                all_data.append((math.sqrt((obj_x - needle_x)**2 + (obj_y - needle_y)**2), 0.0, path))
        
    return all_data

    
def get_distance_data(yolo_model, input_videos, yolo_class):
    all_data = []
    for label, path, vid_f in tqdm(input_videos):
        vc = torchvision.io.read_video(path)[0].permute(3, 0, 1, 2)
        is_right = classify_nerve_is_right(yolo_model, vc)
        
        orig_height = vc.size(2)
        orig_width = vc.size(3)

        obj_bb = []
        needle_bb = []

        for i in range(vc.size(1) - 1, vc.size(1) - 40, -1):
            frame = vc[:, i, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()

            bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

            obj_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==yolo_class]
            needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

            if len(obj_bb) > 0 and len(needle_bb) > 0:
                obj_bb = obj_bb[0]
                needle_bb = needle_bb[0]
                break

        if len(obj_bb) == 0 or len(needle_bb) == 0:
            continue

        obj_x = round((obj_bb[2] + obj_bb[0]) / 2 * orig_width)
        obj_y = round((obj_bb[3] + obj_bb[1]) / 2 * orig_height)

        needle_x = needle_bb[2] * orig_width
        needle_y = needle_bb[3] * orig_height

        if not is_right:
            needle_x = needle_bb[0] * orig_width
        
        if label == 'Positives':
            label = 1.0
        elif label == 'Negatives':
            label = 0.0
        else:
            raise Exception('Bad label')

        all_data.append((math.sqrt((obj_x - needle_x)**2 + (obj_y - needle_y)**2), label, path))
        
    return all_data

def get_splits(videos):
    gss = GroupKFold(n_splits=5)

    groups = [vid[2].split('/')[-2] for vid in videos]

    targets = [vid[2].split('/')[-3] for vid in videos]

    return gss.split(np.arange(len(targets)), targets, groups)

def get_splits_all_train(videos):
    return [(range(len(videos)), [])]

yolo_model = YoloModel('pnb')
video_path = "/shared_data/bamc_pnb_data/revised_training_data/"

print('Beginning...')

labels = [name for name in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, name))]

videos = []
for label in labels:
    videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])

print('Making splits...')    

splits = get_splits_all_train(videos)

preds = []
gt = []
fp_vids = []
fn_vids = []

split_count = 0

for train_idx, test_idx in splits:
    print('On split {}...'.format(split_count))

    print('Processing data...')
    train_videos = [ex for i, ex in enumerate(videos) if i in train_idx]
    if len(test_idx) > 0:
        test_videos = [ex for i, ex in enumerate(videos) if i in test_idx]
        assert not set(train_videos).intersection(set(test_videos))
    else:
        test_videos = train_videos
    
#     nerve_train_data = get_distance_data(yolo_model, train_videos, 1)
    if not os.path.exists('sparse_coding_torch/pnb/regression_train.pkl'):
        vessel_train_data = get_distance_data_sme_labels(yolo_model, train_videos, 0)
        pkl.dump(vessel_train_data, open('sparse_coding_torch/pnb/regression_train.pkl', 'wb+'))
    else:
        vessel_train_data = pkl.load(open('sparse_coding_torch/pnb/regression_train.pkl', 'rb'))
    
#     nerve_test_data = get_distance_data(yolo_model, test_videos, 1)
    if not os.path.exists('sparse_coding_torch/pnb/regression_test.pkl'):
        vessel_test_data = get_distance_data_sme_labels(yolo_model, test_videos, 0)
        pkl.dump(vessel_test_data, open('sparse_coding_torch/pnb/regression_test.pkl', 'wb+'))
    else:
        vessel_test_data = pkl.load(open('sparse_coding_torch/pnb/regression_test.pkl', 'rb'))

#     train_nerve_X = np.array([nerve_train_data[i][0] for i in range(len(nerve_train_data))]).reshape(-1, 1)
#     test_nerve_X = np.array([nerve_test_data[i][0] for i in range(len(nerve_test_data))]).reshape(-1, 1)
    
#     train_nerve_Y = np.array([nerve_train_data[i][1] for i in range(len(nerve_train_data))]).reshape(-1, 1)
#     test_nerve_Y = np.array([nerve_test_data[i][1] for i in range(len(nerve_test_data))]).reshape(-1, 1)
    
    train_vessel_X = np.array([vessel_train_data[i][0] for i in range(len(vessel_train_data))]).reshape(-1, 1)
    test_vessel_X = np.array([vessel_test_data[i][0] for i in range(len(vessel_test_data))]).reshape(-1, 1)
    
    train_vessel_Y = np.array([vessel_train_data[i][1] for i in range(len(vessel_train_data))]).reshape(-1, 1)
    test_vessel_Y = np.array([vessel_test_data[i][1] for i in range(len(vessel_test_data))]).reshape(-1, 1)
    
    print('Training models...')
    
#     nerve_clf = LogisticRegression().fit(train_nerve_X, train_nerve_Y)
#     nerve_score = nerve_clf.score(test_nerve_X, test_nerve_Y)
    
#     print('Nerve accuracy: {:.2f}'.format(nerve_score))
    
    vessel_clf = LogisticRegression().fit(train_vessel_X, train_vessel_Y)
    vessel_score = vessel_clf.score(test_vessel_X, test_vessel_Y)
    
#     print(vessel_clf.get_params(deep=True))

    print(vessel_clf.intercept_, vessel_clf.coef_)
#     random.shuffle(train_vessel_X)
    for j in range(len(train_vessel_X)):
        if train_vessel_Y[j][0] == 1:
            print(vessel_clf.predict_proba(train_vessel_X[j].reshape(-1, 1)))
            print(tf.math.sigmoid(vessel_clf.intercept_ + vessel_clf.coef_[0][0] * train_vessel_X[j]))
            print(train_vessel_X[j])
            print(train_vessel_Y[j])
            print('---------------------------------------')
            raise Exception
    
    print('Vessel accuracy: {:.2f}'.format(vessel_score))
    
    print('Running predictions on test videos...')
    
    for label, path, vid_f in tqdm(test_videos):
        vc = torchvision.io.read_video(path)[0].permute(3, 0, 1, 2)
        is_right = classify_nerve_is_right(yolo_model, vc)
        
        orig_height = vc.size(2)
        orig_width = vc.size(3)
        
        if label == 'Positives':
            gt.append(1.0)
        elif label == 'Negatives':
            gt.append(0.0)
        else:
            raise Exception('Bad label')

#         nerve_bb = []
#         needle_bb = []

#         for i in range(vc.size(1) - 1, vc.size(1) - 40, -1):
#             frame = vc[:, i, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()

#             bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

#             nerve_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==1]
#             needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

#             if len(nerve_bb) > 0 and len(needle_bb) > 0:
#                 nerve_bb = nerve_bb[0]
#                 needle_bb = needle_bb[0]
#                 break

#         if len(nerve_bb) > 0 and len(needle_bb) > 0:
#             nerve_x = round((nerve_bb[2] + nerve_bb[0]) / 2 * orig_width)
#             nerve_y = round((nerve_bb[3] + nerve_bb[1]) / 2 * orig_height)

#             needle_x = needle_bb[2] * orig_width
#             needle_y = needle_bb[3] * orig_height

#             if not is_right:
#                 needle_x = needle_bb[0] * orig_width
                
#             distance = math.sqrt((nerve_x - needle_x)**2 + (nerve_y - needle_y)**2)
            
#             pred = nerve_clf.predict(np.array([distance]).reshape(1, -1))[0]
#             preds.append(pred)
#         else:
        vessel_bb = []

        for i in range(vc.size(1) - 1, vc.size(1) - 40, -1):
            frame = vc[:, i, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy()

            bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame)

            vessel_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==0]
            needle_bb = [bb for bb, class_pred, score in zip(bounding_boxes, classes, scores) if class_pred==2]

            if len(vessel_bb) > 0 and len(needle_bb) > 0:
                vessel_bb = vessel_bb[0]
                needle_bb = needle_bb[0]
                break

        if len(vessel_bb) == 0 or len(needle_bb) == 0:
            preds.append(0.0)
            if label == 'Positives':
                fn_vids.append(vid_f)
            continue

        vessel_x = round((vessel_bb[2] + vessel_bb[0]) / 2 * orig_width)
        vessel_y = round((vessel_bb[3] + vessel_bb[1]) / 2 * orig_height)

        needle_x = needle_bb[2] * orig_width
        needle_y = needle_bb[3] * orig_height

        if not is_right:
            needle_x = needle_bb[0] * orig_width

        distance = math.sqrt((vessel_x - needle_x)**2 + (vessel_y - needle_y)**2)

        pred = vessel_clf.predict(np.array([distance]).reshape(1, -1))[0]
        preds.append(pred)
        
        if pred == 0.0 and label == 'Positives':
            fn_vids.append(vid_f)
        elif pred == 1.0 and label == 'Negatives':
            fp_vids.append(vid_f)
            
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

print('False Negative Videos:')
print(fn_vids)
print('False Positive Videos:')
print(fp_vids)