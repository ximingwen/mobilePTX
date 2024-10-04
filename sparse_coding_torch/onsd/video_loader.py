from os import listdir
from os.path import isfile
from os.path import join
from os.path import isdir
from os.path import abspath
from os.path import exists
import json
import glob

from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.datasets.video_utils import VideoClips
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_video
import torchvision as tv
from torch import nn
import torchvision.transforms.functional as tv_f
import csv
import random
import cv2
from yolov4.get_bounding_boxes import YoloModel
import tensorflow as tf
import torchvision

from matplotlib import pyplot as plt
from matplotlib import cm

def get_participants(filenames):
    return [f.split('/')[-2] for f in filenames]

def three_mm(yolo_model, frame):
    orig_height = frame.size(1)
    orig_width = frame.size(2)
    
    bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())
    
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
        return None
    
    nerve_center_x = round((nerve_bounding_box[2] + nerve_bounding_box[0]) / 2 * orig_width)
    nerve_center_y = round((nerve_bounding_box[3] + nerve_bounding_box[1]) / 2 * orig_height)
    
    
    eye_center_x = round((eye_bounding_box[2] + eye_bounding_box[0]) / 2 * orig_width)
#     eye_center_y = round((eye_bounding_box[3] + eye_bounding_box[1]) / 2 * orig_height)
    eye_center_y = round(eye_bounding_box[3] * orig_height)
    
    crop_center_x = nerve_center_x
    crop_center_y = eye_center_y + 65

    return crop_center_y
    
def get_yolo_region_onsd(yolo_model, frame, crop_width, crop_height, do_augmentation, label=''):
    orig_height = frame.size(1)
    orig_width = frame.size(2)
    
    bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())
    
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
        return None
    
    nerve_center_x = round((nerve_bounding_box[2] + nerve_bounding_box[0]) / 2 * orig_width)
    nerve_center_y = round((nerve_bounding_box[3] + nerve_bounding_box[1]) / 2 * orig_height)
    
    
    eye_center_x = round((eye_bounding_box[2] + eye_bounding_box[0]) / 2 * orig_width)
#     eye_center_y = round((eye_bounding_box[3] + eye_bounding_box[1]) / 2 * orig_height)
    eye_center_y = round(eye_bounding_box[3] * orig_height)
    
    crop_center_x = nerve_center_x
    crop_center_y = eye_center_y + 65
    
    all_frames = []
    if do_augmentation:
        NUM_AUGMENTED_SAMPLES=10
        frame_center_y = int(orig_height / 2)
        frame_center_x = int(orig_width / 2)
        
        shift_x = (frame_center_x - crop_center_x)
        shift_y = (frame_center_y - crop_center_y)
        
#         print(shift_x)
#         print(shift_y)
        
#         cv2.imwrite('onsd_not_translated.png', frame.numpy().swapaxes(0,1).swapaxes(1,2))
        frame = torchvision.transforms.functional.affine(frame, angle=0, translate=(shift_x, shift_y), scale=1.0, shear=0.0)
#         cv2.imwrite('onsd_translated.png', frame.numpy().swapaxes(0,1).swapaxes(1,2))
#         raise Exception
        
        transform_list = []
#         print(label)
        if label == 'Positives':
            transform_list.append(torchvision.transforms.RandomAffine(degrees=5, scale=(1.0, 1.7)))
        elif label == 'Negatives':
            transform_list.append(torchvision.transforms.RandomAffine(degrees=5, scale=(0.5, 1.0)))
        transform = torchvision.transforms.Compose(transform_list)
        for i in range(NUM_AUGMENTED_SAMPLES):
            aug_frame = transform(frame)
            aug_frame = aug_frame[:, frame_center_y:frame_center_y + crop_height, frame_center_x - int(crop_width/2):frame_center_x + int(crop_width/2)]
#             normal_crop = frame[:, frame_center_y:frame_center_y + crop_height, frame_center_x - int(crop_width/2):frame_center_x + int(crop_width/2)]
#             cv2.imwrite('onsd_zoomed.png', aug_frame.numpy().swapaxes(0,1).swapaxes(1,2))
#             cv2.imwrite('onsd_not_zoomed.png', normal_crop.numpy().swapaxes(0,1).swapaxes(1,2))
#             print(aug_frame.size())
#             print(frame.size())
#             raise Exception
            all_frames.append(aug_frame)
    else:
#         print(frame.size())
#         print(crop_center_y)
#         print(crop_center_x)
        trimmed_frame = frame[:, crop_center_y - int(crop_height / 2):crop_center_y + int(crop_height / 2), max(crop_center_x - int(crop_width/2), 0):crop_center_x + int(crop_width/2)]
#         print(trimmed_frame.size())
        all_frames.append(trimmed_frame)
        
#     cv2.imwrite('test_onsd_orig_w_eye.png', frame.numpy().swapaxes(0,1).swapaxes(1,2))
#     plt.clf()
#     plt.imshow(frame.numpy().swapaxes(0,1).swapaxes(1,2), cmap=cm.Greys_r)
#     plt.scatter([crop_center_x], [crop_center_y], color=["red"])
#     plt.savefig('test_onsd_orig_w_eye_dist.png')
#     cv2.imwrite('test_onsd_orig_trimmed_slice.png', trimmed_frame.numpy().swapaxes(0,1).swapaxes(1,2))
#     raise Exception
        
    return all_frames

class ONSDGoodFramesLoader:
    def __init__(self, video_path, clip_width, clip_height, transform=None, yolo_model=None):
        self.transform = transform
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        
        self.count = 0
        
        valid_frames = {}
        invalid_frames = {}
        with open('sparse_coding_torch/onsd/good_frames_onsd.csv', 'r') as valid_in:
            reader = csv.DictReader(valid_in)
            for row in reader:
                vid = row['video'].strip()
                good_frames = row['good_frames'].strip()
                bad_frames = row['bad_frames'].strip()
                
                valid_frames[vid] = []
                if good_frames:
                    for subrange in good_frames.split(';'):
                        splitrange = subrange.split('-')
                        valid_frames[vid].append((int(splitrange[0]), int(splitrange[1])))
                if bad_frames:
                    for subrange in bad_frames.split(';'):
                        splitrange = subrange.split('-')
                        invalid_frames[vid] = (int(splitrange[0]), int(splitrange[1]))
        
#         onsd_widths = {}
#         with open(join(video_path, 'onsd_widths.csv'), 'r') as width_in:
#             reader = csv.reader(width_in)
#             for row in reader:
#                 width_vals = [float(val) for val in row[3:] if val != '']
#                 onsd_widths[row[2]] = round(sum(width_vals) / len(width_vals), 2)

        onsd_widths = {}
        with open("sparse_coding_torch/onsd/individual_frames_cleaned/onsd_labeled_widths_classifier.csv", 'r') as width_in:
            reader = csv.reader(width_in)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                onsd_widths[row[0]] = float(row[1])
        
        clip_cache_file = 'clip_cache_onsd_{}_{}.pt'.format(clip_width, clip_height)
        difficult_cache_file = 'difficult_vid_cache_onsd_{}_{}.pt'.format(clip_width, clip_height)
        
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])
            
        self.difficult_vids = []
            
        self.clips = []
        
        if exists(clip_cache_file):
            self.clips = torch.load(open(clip_cache_file, 'rb'))
            self.difficult_vids = torch.load(open(difficult_cache_file, 'rb'))
        else:
            vid_idx = 0
            for txt_label, path, _ in tqdm(self.videos):
                vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
#                 width = 0.0
                
                frame_key = path.split('/')[-2]
                if frame_key in valid_frames:
                    ranges = valid_frames[frame_key]
                    
                    for start_range, end_range in ranges:
                        for j in range(start_range, end_range, 5):
                            if j >= vc.size(1):
                                break
                            frame = vc[:, j, :, :]

                            width_key = path.split('/')[-1]
                            width_key = width_key[:width_key.rfind('.')]
                            width_key = txt_label + '/' + width_key
                            width_key = width_key + '/' + str(j) + '.png'
                            if width_key not in onsd_widths:
                                continue
                            else:
                                width = onsd_widths[width_key]

                            if yolo_model is not None:
                                all_frames = get_yolo_region_onsd(yolo_model, frame, clip_width, clip_height, False, txt_label)
                            else:
                                all_frames = [frame]

                            if all_frames is None or len(all_frames) == 0:
                                continue

                            if self.transform:
                                all_frames = [self.transform(frm) for frm in all_frames]

                            label = self.videos[vid_idx][0]
                            if label == 'Positives':
                                label = np.array(1.0)
                            elif label == 'Negatives':
                                label = np.array(0.0)

#                             width = np.round(width / 30)

                            for frm in all_frames:
                                self.clips.append((label, frm.numpy(), self.videos[vid_idx][2], width))
                else:
                    label = self.videos[vid_idx][0]
                    if label == 'Positives':
                        label = np.array(1.0)
                    elif label == 'Negatives':
                        label = np.array(0.0)
                    self.difficult_vids.append((label, self.videos[vid_idx][2]))

                vid_idx += 1
                
            torch.save(self.clips, open(clip_cache_file, 'wb+'))
            torch.save(self.difficult_vids, open(difficult_cache_file, 'wb+'))
            
        num_positive = len([clip[0] for clip in self.clips if clip[0] == 1.0])
        num_negative = len([clip[0] for clip in self.clips if clip[0] == 0.0])
        
        self.max_width = max([clip[3] for clip in self.clips])
        
        random.shuffle(self.clips)
        
#         print('Loaded', num_positive, 'positive examples.')
#         print('Loaded', num_negative, 'negative examples.')
        
    def get_difficult_vids(self):
        return self.difficult_vids
        
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
    
    def get_all_videos(self):
        return set([self.clips[i][2] for i in range(len(self.clips))])
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def set_indicies(self, iter_idx):
        new_clips = []
        for i, clip in enumerate(self.clips):
            if i in iter_idx:
                new_clips.append(clip)
                
        self.clips = new_clips
        
    def get_frames(self):
        return [frame for _, frame, _, _ in self.clips]
    
    def get_widths(self):
        return [width for _, _, _, width in self.clips]
    
    def __next__(self):
        if self.count < len(self.clips):
            label, frame, vid_f, widths = self.clips[self.count]
            self.count += 1
            return label, frame, widths
        else:
            raise StopIteration
            
    def __iter__(self):
        return self
    
class ONSDAllFramesLoader:
    def __init__(self, video_path, clip_width, clip_height, transform=None, yolo_model=None):
        self.transform = transform
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        
        self.count = 0
        
        clip_cache_file = 'clip_cache_all_onsd_{}_{}.pt'.format(clip_width, clip_height)
        
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])
            
        self.clips = []
        
        if exists(clip_cache_file):
            self.clips = torch.load(open(clip_cache_file, 'rb'))
        else:
            vid_idx = 0
            for txt_label, path, _ in tqdm(self.videos):
                vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
#                 width = 0.0
                
                for j in range(0, vc.size(1), 5):
                    frame = vc[:, j, :, :]

                    width = 0

                    if yolo_model is not None:
                        all_frames = get_yolo_region_onsd(yolo_model, frame, clip_width, clip_height, False, txt_label)
                    else:
                        all_frames = [frame]

                    if all_frames is None or len(all_frames) == 0:
                        continue

                    if self.transform:
                        all_frames = [self.transform(frm) for frm in all_frames]

                    label = self.videos[vid_idx][0]
                    if label == 'Positives':
                        label = np.array(1.0)
                    elif label == 'Negatives':
                        label = np.array(0.0)

                    width = np.round(width / 30)

                    for frm in all_frames:
                        self.clips.append((label, frm.numpy(), self.videos[vid_idx][2], width))

                vid_idx += 1
                
            torch.save(self.clips, open(clip_cache_file, 'wb+'))
            
        num_positive = len([clip[0] for clip in self.clips if clip[0] == 1.0])
        num_negative = len([clip[0] for clip in self.clips if clip[0] == 0.0])
        
        random.shuffle(self.clips)
        
    def get_difficult_vids(self):
        return self.difficult_vids
        
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
    
    def get_all_videos(self):
        return set([self.clips[i][2] for i in range(len(self.clips))])
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def set_indicies(self, iter_idx):
        new_clips = []
        for i, clip in enumerate(self.clips):
            if i in iter_idx:
                new_clips.append(clip)
                
        self.clips = new_clips
        
    def get_frames(self):
        return [frame for _, frame, _, _ in self.clips]
    
    def get_widths(self):
        return [width / 1 for _, _, _, width in self.clips]
    
    def __next__(self):
        if self.count < len(self.clips):
            label, frame, vid_f, widths = self.clips[self.count]
            self.count += 1
            return label, frame, widths
        else:
            raise StopIteration
            
    def __iter__(self):
        return self

class RegressionLoader:
    def __init__(self, csv_path, clip_width, clip_height, transform=None, yolo_model=None):
        self.transform = transform
        
        self.count = 0
        
        onsd_widths = {}
        with open(csv_path, 'r') as width_in:
            reader = csv.reader(width_in)
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                onsd_widths[row[0]] = float(row[1])
        
        self.frames = []
        self.max_width = 0.0
        for f_path, width in tqdm(onsd_widths.items()):
            frame = torch.tensor(cv2.imread(f_path)).swapaxes(2, 1).swapaxes(1, 0)
                    
            if yolo_model is not None:
                all_frames = get_yolo_region_onsd(yolo_model, frame, clip_width, clip_height, False)
            else:
                all_frames = [frame]
                        
            if all_frames is None or len(all_frames) == 0:
                continue

            if self.transform:
                all_frames = [self.transform(frm) for frm in all_frames]
                
            if width > self.max_width:
                self.max_width = width
                
            width = np.round(width / 5) * 5
                        
            for frm in all_frames:
                self.frames.append((width, frm.numpy(), f_path))
        
        random.shuffle(self.frames)
        
    def get_filenames(self):
        return ['/'.join(self.frames[i][2].split('/')[:-1]) for i in range(len(self.frames))]
        
    def get_labels(self):
        return [self.frames[i][0] / self.max_width for i in range(len(self.frames))]
    
    def set_indicies(self, iter_idx):
        new_frames = []
        for i, frame in enumerate(self.frames):
            if i in iter_idx:
                new_frames.append(frame)
                
        self.frames = new_frames
        
    def get_frames(self):
        return [frame for _, frame, _ in self.frames]
    
    def __next__(self):
        if self.count < len(self.frames):
            width, frame, vid_f = self.frames[self.count]
            self.count += 1
            return width, frame
        else:
            raise StopIteration
            
    def __iter__(self):
        return self

class FrameLoader:
    def __init__(self, video_path, clip_width, clip_height, transform=None, yolo_model=None):
        self.transform = transform
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        
        self.count = 0
        
        valid_frames = {}
        invalid_frames = {}
        with open('sparse_coding_torch/onsd/good_frames_onsd.csv', 'r') as valid_in:
            reader = csv.DictReader(valid_in)
            for row in reader:
                vid = row['video'].strip()
                good_frames = row['good_frames'].strip()
                bad_frames = row['bad_frames'].strip()
                if good_frames:
                    for subrange in good_frames.split(';'):
                        splitrange = subrange.split('-')
                        valid_frames[vid] = (int(splitrange[0]), int(splitrange[1]))
                if bad_frames:
                    for subrange in bad_frames.split(';'):
                        splitrange = subrange.split('-')
                        invalid_frames[vid] = (int(splitrange[0]), int(splitrange[1]))
        
        clip_cache_file = 'clip_cache_onsd_frames_{}_{}.pt'.format(clip_width, clip_height)
        
        self.videos = []
        for label in self.labels:
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])
            
        self.clips = []
        
        if exists(clip_cache_file):
            self.clips = torch.load(open(clip_cache_file, 'rb'))
        else:
            vid_idx = 0
            for txt_label, path, _ in tqdm(self.videos):
                vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
                
                frame_key = path.split('/')[-2]
                if frame_key in valid_frames:
                    start_range, end_range = valid_frames[frame_key]
                
                    for j in range(start_range, end_range):
                        if j == vc.size(1):
                            break
                        
                        frame = vc[:, j, :, :]

                        if yolo_model is not None:
                            all_frames = get_yolo_region_onsd(yolo_model, frame, clip_width, clip_height, True, txt_label)
                        else:
                            all_frames = [frame]

                        if all_frames is None or len(all_frames) == 0:
                            continue
                            
                        all_frames = [frm[:, 70:frm.size(1)-200, :] for frm in all_frames]

                        if self.transform:
                            all_frames = [self.transform(frm) for frm in all_frames if frm.size(1) > 0 and frm.size(2) > 0]
                        
                        label = np.array(1.0)
                        
                        for frm in all_frames:
#                             cv2.imwrite('onsd_full_frame_clean.png', frm.swapaxes(0,1).swapaxes(1,2).numpy())
#                             print(frm.size())
#                             raise Exception
                            self.clips.append((label, frm.numpy(), self.videos[vid_idx][2]))

                if frame_key in invalid_frames:
                    start_range, end_range = invalid_frames[frame_key]
                
                    for j in range(start_range, end_range):
                        if j == vc.size(1):
                            break
                        frame = vc[:, j, :, :]

                        if yolo_model is not None:
                            all_frames = get_yolo_region_onsd(yolo_model, frame, clip_width, clip_height, True, txt_label)
                        else:
                            all_frames = [frame]

                        if all_frames is None or len(all_frames) == 0:
                            continue
                            
                        all_frames = [frm[:, 70:frm.size(1)-200, :] for frm in all_frames]

                        if self.transform:
                            all_frames = [self.transform(frm) for frm in all_frames if frm.size(1) > 0 and frm.size(2) > 0]
                        
                        label = np.array(0.0)
                        
                        for frm in all_frames:
                            self.clips.append((label, frm.numpy(), self.videos[vid_idx][2]))
                    
#                     negative_frames = [i for i in range(vc.size(1)) if i < start_range or i > end_range]
#                     random.shuffle(negative_frames)
                    
#                     negative_frames = negative_frames[:end_range - start_range]
#                     for i in negative_frames:
#                         frame = vc[:, i, :, :]

#                         if self.transform:
#                             frame = self.transform(frame)
                        
#                         label = np.array(0.0)
                        
#                         self.clips.append((label, frame.numpy(), self.videos[vid_idx][2]))
#                 else:
#                     for j in random.sample(range(vc.size(1)), 50):
#                         frame = vc[:, j, :, :]

#                         if self.transform:
#                             frame = self.transform(frame)
                        
#                         label = np.array(0.0)
                        
#                         self.clips.append((label, frame.numpy(), self.videos[vid_idx][2]))

                vid_idx += 1
                
            torch.save(self.clips, open(clip_cache_file, 'wb+'))
            
        num_positive = len([clip[0] for clip in self.clips if clip[0] == 1.0])
        num_negative = len([clip[0] for clip in self.clips if clip[0] == 0.0])
        
        random.shuffle(self.clips)
        
        print('Loaded', num_positive, 'positive examples.')
        print('Loaded', num_negative, 'negative examples.')
        
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
    
    def get_all_videos(self):
        return set([self.clips[i][2] for i in range(len(self.clips))])
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def set_indicies(self, iter_idx):
        new_clips = []
        for i, clip in enumerate(self.clips):
            if i in iter_idx:
                new_clips.append(clip)
                
        self.clips = new_clips
        
    def get_frames(self):
        return [frame for _, frame, _ in self.clips]
    
    def __next__(self):
        if self.count < len(self.clips):
            label, frame, vid_f = self.clips[self.count]
            self.count += 1
            return label, frame
        else:
            raise StopIteration
            
    def __iter__(self):
        return self