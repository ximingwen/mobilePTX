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
from skimage.transform import warp_polar
from skimage.io import imsave

from matplotlib import pyplot as plt
from matplotlib import cm

import math

def get_participants(filenames):
    return [f.split('/')[-2] for f in filenames]
    
def load_pnb_region_labels(file_path):
    all_regions = {}
    with open(file_path, newline='') as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            idx = row['idx']
            positive_regions = row['positive_regions'].strip()
            negative_regions = row['negative_regions'].strip()
            
            all_regions[idx] = (negative_regions, positive_regions)
            
        return all_regions
    
def get_yolo_regions(yolo_model, clip, is_right, crop_width, crop_height):
    orig_height = clip.size(2)
    orig_width = clip.size(3)
    bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(clip[:, clip.shape[1]//2, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy())
    
    needle_bb = None
    for bb, class_pred in zip(bounding_boxes, classes):
        if class_pred == 2:
            needle_bb = bb
            
    if needle_bb is None:
        return []
    
    rotate_box = False
    
    if 1 not in classes:
        return []
    
    all_clips = []
    for bb, class_pred, score in zip(bounding_boxes, classes, scores):
        if class_pred != 1:
            continue
        center_x = round((bb[2] + bb[0]) / 2 * orig_width)
        center_y = round((bb[3] + bb[1]) / 2 * orig_height)
        
        if not is_right:
            clip = tv.transforms.functional.hflip(clip)
            center_x = orig_width - center_x
            needle_bb[0] = orig_width - needle_bb[0]
            needle_bb[2] = orig_width - needle_bb[2]
        
#         lower_y = round((bb[0] * orig_height))
#         upper_y = round((bb[2] * orig_height))
#         lower_x = round((bb[1] * orig_width))
#         upper_x = round((bb[3] * orig_width))
        
#         if is_right:
        angle = calculate_angle(needle_bb, center_x, center_y, orig_height, orig_width)
#         else:
#             angle = calculate_angle(needle_bb, lower_x, center_y, orig_height, orig_width)
        
#         lower_y = center_y - (crop_height // 2)
#         upper_y = center_y + (crop_height // 2) 
        
#         if is_right:
#             lower_x = center_x - crop_width
#             upper_x = center_x
#         else:
#             lower_x = center_x
#             upper_x = center_x + crop_width
            
#         if lower_x < 0:
#             lower_x = 0
#         if upper_x < 0:
#             upper_x = 0
#         if lower_y < 0:
#             lower_y = 0
#         if upper_y < 0:
#             upper_y = 0
        clip = tv.transforms.functional.rotate(clip, angle=angle, center=[center_x, center_y])
    
#         test_img = clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2)
        
#         kernel = np.array([[-1.0, -1.0], 
#                    [2.0, 2.0],
#                    [-1.0, -1.0]])

#         kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)

# #filter the source image
#         filtered_img = cv2.filter2D(test_img,-1,kernel)
            
# #         plt.clf()
#         plt.imshow(filtered_img, cmap=cm.Greys_r)
# #         # plt.scatter([214], [214], color="red")
# #         plt.scatter([center_x, int(needle_bb[0]*orig_width)], [center_y, int(needle_bb[1] * orig_height)], color=["red", 'red'])
# # #         cv2.imwrite('test_normal.png', clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#         plt.savefig('test_normal.png')
#         raise Exception
            
#         if rotate_box:
# #             cv2.imwrite('test_1.png', clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#             if is_right:
#         clip = tv.transforms.functional.rotate(clip, angle=angle, center=[center_x, center_y])
#             else:
# #                 cv2.imwrite('test_1.png', clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#                 clip = tv.transforms.functional.rotate(clip, angle=-angle, center=[center_x, center_y])
#                 cv2.imwrite('test_2.png', clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))

#         plt.imshow(clip[0, 0, :, :], cmap=cm.Greys_r)
#         # plt.annotate('25, 50', xy=(25, 50), xycoords='data',
#         #             xytext=(0.5, 0.5), textcoords='figure fraction',
#         #             arrowprops=dict(arrowstyle="->"))
#         plt.scatter([center_x], [center_y], color="red")
#         plt.savefig('red_dot.png')
#         clip = clip[:, :, :upper_y, :]

        ro,col=clip[0, 0, :, :].shape
        max_radius = int(np.sqrt(ro**2+col**2)/2)
# # #         print(upper_y)
# # #         print(bb[0])
# # #         print(center_x)
# # #         print(center_y)
        trimmed_clip = []
        for i in range(clip.shape[0]):
            sub_clip = []
            for j in range(clip.shape[1]):
                sub_clip.append(cv2.linearPolar(clip[i, j, :, :].numpy(), (center_x, center_y), max_radius, cv2.WARP_FILL_OUTLIERS))
# #                 sub_clip.append(warp_polar(clip[i, j, :, :].numpy(), center=(center_x, center_y), radius=max_radius, preserve_range=True))
            trimmed_clip.append(np.stack(sub_clip))
        trimmed_clip = np.stack(trimmed_clip)
        
        approximate_needle_position = int(((angle+150)/360)*orig_height)
        
#         plt.clf()
#         plt.imshow(trimmed_clip[:, 0, :, :].swapaxes(0,1).swapaxes(1,2), cmap=cm.Greys_r)
# #         # plt.scatter([214], [214], color="red")
#         plt.scatter([center_x], [approximate_needle_position], color=["red"])
# # #         cv2.imwrite('test_normal.png', clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#         plt.savefig('test_polar.png')
        
        trimmed_clip = trimmed_clip[:, :, approximate_needle_position - (crop_height//2):approximate_needle_position + (crop_height//2), :crop_width]
#         trimmed_clip = clip[:, :, center_y - (crop_height//2):center_y + (crop_height//2), center_x - crop_width:center_x]
                
#         trimmed_clip=cv2.linearPolar(clip[0, 0, :, :].numpy(), (center_x, center_y), max_radius, cv2.WARP_FILL_OUTLIERS)
#         trimmed_clip = warp_polar(clip[0, 0, :, :].numpy(), center=(center_x, center_y), radius=max_radius)

#         trimmed_clip = clip[:, :, lower_y:upper_y, lower_x:upper_x]
        
#         if orig_width - center_x >= center_x:
#         if not is_right:
#         print(angle)
#         if not is_right:
#         cv2.imwrite('test_polar.png', trimmed_clip[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#         plt.clf()
#         plt.imshow(clip[:, 0, :, :].swapaxes(0,1).swapaxes(1,2), cmap=cm.Greys_r)
#         plt.scatter([center_x], [center_y], color="red")
# # #         plt.scatter([center_x], [approximate_needle_position], color=["red"])
# #         cv2.imwrite('test_normal.png', clip.numpy()[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#         plt.savefig('test_normal.png')
#         raise Exception

#         if not is_right:
#             trimmed_clip = tv.transforms.functional.hflip(trimmed_clip)
#             cv2.imwrite('test_polar.png', trimmed_clip)
#         cv2.imwrite('test_yolo.png', trimmed_clip[:, 0, :, :].swapaxes(0,1).swapaxes(1,2))
#         raise Exception
        
        if trimmed_clip.shape[2] == 0 or trimmed_clip.shape[3] == 0:
            continue
        all_clips.append(torch.tensor(trimmed_clip))

    return all_clips

def classify_nerve_is_right(yolo_model, video):
    orig_height = video.size(2)
    orig_width = video.size(3)

    all_preds = []
    if video.size(1) < 10:
        return 1

    for frame in range(0, video.size(1), round(video.size(1) / 10)):
        frame = video[:, frame, :, :]
        bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())
    
        for bb, class_pred in zip(bounding_boxes, classes):
            if class_pred != 2:
                continue
            center_x = (bb[2] + bb[0]) / 2 * orig_width
            center_y = (bb[3] + bb[1]) / 2 * orig_height

            if orig_width - center_x < center_x:
                all_preds.append(0)
            else:
                all_preds.append(1)
        
        if not all_preds:
            for bb, class_pred in zip(bounding_boxes, classes):
                if class_pred != 1:
                    continue
                center_x = (bb[2] + bb[0]) / 2 * orig_width
                center_y = (bb[3] + bb[1]) / 2 * orig_height

                if orig_width - center_x < center_x:
                    all_preds.append(1)
                else:
                    all_preds.append(0)
                    
        if not all_preds:
            all_preds.append(1)
                
    final_pred = round(sum(all_preds) / len(all_preds))

    return final_pred == 1

def calculate_angle(needle_bb, vessel_x, vessel_y, orig_height, orig_width):
    needle_x = needle_bb[0] * orig_width
    needle_y = needle_bb[1] * orig_height

    return np.abs(np.degrees(np.arctan((needle_y-vessel_y)/(needle_x-vessel_x))))

def get_needle_bb(yolo_model, video):
    orig_height = video.size(2)
    orig_width = video.size(3)

    for frame in range(0, video.size(1), 1):
        frame = video[:, frame, :, :]
        
        bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())

        for bb, class_pred in zip(bounding_boxes, classes):
            if class_pred == 2:
                return bb
        
    return None
        
def calculate_angle_video(yolo_model, video):
    orig_height = video.size(2)
    orig_width = video.size(3)

    all_preds = []
    if video.size(1) < 10:
        return 30

    for frame in range(0, video.size(1), video.size(1) // 10):
        frame = video[:, frame, :, :]
        
        bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())

        vessel_x = 0
        vessel_y = 0
        needle_x = 0
        needle_y = 0

        for bb, class_pred in zip(bounding_boxes, classes):
            if class_pred == 0 and vessel_x == 0:
                vessel_x = (bb[2] + bb[0]) / 2 * orig_width
                vessel_y = (bb[3] + bb[1]) / 2 * orig_height
            elif class_pred == 2 and needle_x == 0:
                needle_x = bb[0] * orig_width
                needle_y = bb[1] * orig_height

            if needle_x != 0 and vessel_x != 0:
                break

        if vessel_x > 0 and needle_x > 0:
            all_preds.append(np.abs(np.degrees(np.arctan((needle_y-vessel_y)/(needle_x-vessel_x)))))
        
    return np.mean(np.array(all_preds))
                
    
class PNBLoader(Dataset):
    
    def __init__(self, yolo_model, video_path, clip_width, clip_height, clip_depth, classify_mode=False, balance_classes=False, num_frames=5, frames_to_skip=1, transform=None):
        self.transform = transform
        self.labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        self.count = 0
        
        if classify_mode:
            clip_cache_file = 'clip_cache_pnb_{}_{}_{}_{}.pt'.format(clip_width, clip_height, clip_depth, frames_to_skip)
            clip_cache_final_file = 'clip_cache_pnb_{}_{}_{}_{}_final.pt'.format(clip_width, clip_height, clip_depth, frames_to_skip)
        else:
            clip_cache_file = 'clip_cache_pnb_{}_{}_sparse.pt'.format(clip_width, clip_height)
            clip_cache_final_file = 'clip_cache_pnb_{}_{}_final_sparse.pt'.format(clip_width, clip_height)
        
        region_labels = load_pnb_region_labels('sme_region_labels.csv')
        
        self.videos = []
        for label in self.labels:
#             self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])
            self.videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])
        
#         self.videos = list(filter(lambda x: x[1].split('/')[-2] in ['67', '94', '134', '193', '222', '240'], self.videos))
#         self.videos = list(filter(lambda x: x[1].split('/')[-2] in ['67'], self.videos))
            
        self.clips = []
        
        if exists(clip_cache_file):
            self.clips = torch.load(open(clip_cache_file, 'rb'))
        else:
            vid_idx = 0
            for label, path, _ in tqdm(self.videos):
                vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
                is_right = classify_nerve_is_right(yolo_model, vc)
#                 needle_bb = get_needle_bb(yolo_model, vc)
                
                label = self.videos[vid_idx][0]
                if label == 'Positives':
                    label = np.array(1.0)
                elif label == 'Negatives':
                    label = np.array(0.0)

                if classify_mode:
#                     person_idx = path.split('/')[-2]
                    person_idx = path.split('/')[-1].split(' ')[1]

                    if vc.size(1) < clip_depth:
                        continue

                    if label == 1.0 and person_idx in region_labels:
                        negative_regions, positive_regions = region_labels[person_idx]
                        for sub_region in negative_regions.split(','):
                            sub_region = sub_region.split('-')
                            start_loc = int(sub_region[0])
#                             end_loc = int(sub_region[1]) - 50
                            end_loc = int(sub_region[1]) + 1
                            for j in range(start_loc, end_loc - clip_depth * frames_to_skip, 1):
                                frames = []
                                for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                                    frames.append(vc[:, k, :, :])
                                vc_sub = torch.stack(frames, dim=1)

                                if vc_sub.size(1) < clip_depth:
                                    continue

                                for clip in get_yolo_regions(yolo_model, vc_sub, is_right, clip_width, clip_height):
                                    if self.transform:
                                        clip = self.transform(clip)

                                    self.clips.append((np.array(0.0), clip.numpy(), self.videos[vid_idx][2]))

                        if positive_regions:
                            for sub_region in positive_regions.split(','):
                                sub_region = sub_region.split('-')
#                                 start_loc = int(sub_region[0]) + 15
                                start_loc = int(sub_region[0])
                                if len(sub_region) == 1 and vc.size(1) >= start_loc + clip_depth * frames_to_skip:
                                    frames = []
                                    for k in range(start_loc, start_loc + clip_depth * frames_to_skip, frames_to_skip):
                                        frames.append(vc[:, k, :, :])
                                    vc_sub = torch.stack(frames, dim=1)

                                    if vc_sub.size(1) < clip_depth:
                                        continue
                                        
                                    for clip in get_yolo_regions(yolo_model, vc_sub, is_right, clip_width, clip_height):
                                        if self.transform:
                                            clip = self.transform(clip)

                                        self.clips.append((np.array(1.0), clip.numpy(), self.videos[vid_idx][2]))
                                elif vc.size(1) >= start_loc + clip_depth * frames_to_skip:
                                    end_loc = sub_region[1]
                                    if end_loc.strip().lower() == 'end':
                                        end_loc = vc.size(1)
                                    else:
                                        end_loc = int(end_loc)
                                    for j in range(start_loc, end_loc - clip_depth * frames_to_skip, 1):
                                        frames = []
                                        for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                                            frames.append(vc[:, k, :, :])
                                        vc_sub = torch.stack(frames, dim=1)

                                        if vc_sub.size(1) < clip_depth:
                                            continue
                                        for clip in get_yolo_regions(yolo_model, vc_sub, is_right, clip_width, clip_height):
                                            if self.transform:
                                                clip = self.transform(clip)

                                            self.clips.append((np.array(1.0), clip.numpy(), self.videos[vid_idx][2]))
                                else:
                                    continue
                    elif label == 1.0:
                        frames = []
                        for k in range(0, -1 * clip_depth * frames_to_skip, frames_to_skip):
                            frames.append(vc[:, k, :, :])
                        if not frames:
                            continue
                        vc_sub = torch.stack(frames, dim=1)
                        if vc_sub.size(1) < clip_depth:
                            continue
                        for clip in get_yolo_regions(yolo_model, vc_sub, is_right, clip_width, clip_height):
                            if self.transform:
                                clip = self.transform(clip)

                            self.clips.append((label, clip.numpy(), self.videos[vid_idx][2]))
                    elif label == 0.0:
                        for j in range(0, vc.size(1) - clip_depth * frames_to_skip, 1):
                            frames = []
                            for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                                frames.append(vc[:, k, :, :])
                            vc_sub = torch.stack(frames, dim=1)
                            if vc_sub.size(1) < clip_depth:
                                continue
                            for clip in get_yolo_regions(yolo_model, vc_sub, is_right, clip_width, clip_height):
                                if self.transform:
                                    clip = self.transform(clip)

                                self.clips.append((label, clip.numpy(), self.videos[vid_idx][2]))
                    else:
                        raise Exception('Invalid label')
                else:
                    for j in range(0, vc.size(1) - clip_depth * frames_to_skip, clip_depth):
                        frames = []
                        for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                            frames.append(vc[:, k, :, :])
                        vc_sub = torch.stack(frames, dim=1)
                        
                        if vc_sub.size(1) != clip_depth:
                            continue
#                         for clip in get_yolo_regions(yolo_model, vc_sub, is_right, clip_width, clip_height):
                        if self.transform:
                            vc_sub = self.transform(vc_sub)

                        self.clips.append((label, vc_sub.numpy(), self.videos[vid_idx][2]))

                vid_idx += 1
                
            torch.save(self.clips, open(clip_cache_file, 'wb+'))
            
        num_positive = len([clip[0] for clip in self.clips if clip[0] == 1.0])
        num_negative = len([clip[0] for clip in self.clips if clip[0] == 0.0])
        
        random.shuffle(self.clips)
        
        if balance_classes:
            new_clips = []
            count_negative = 0
            for clip in self.clips:
                if clip[0] == 0.0:
                    if count_negative < num_positive:
                        new_clips.append(clip)
                    count_negative += 1
                else:
                    new_clips.append(clip)
                    
            self.clips = new_clips
            num_positive = len([clip[0] for clip in self.clips if clip[0] == 1.0])
            num_negative = len([clip[0] for clip in self.clips if clip[0] == 0.0])
        
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
    
    def get_labels(self):
        return [label for label, _, _ in self.clips]
    
    def __next__(self):
        if self.count < len(self.clips):
            label, frame, vid_f = self.clips[self.count]
            self.count += 1
            return label, frame
        else:
            raise StopIteration
            
    def __iter__(self):
        return self
    
    
class NeedleLoader(Dataset):
    def __init__(self, video_path, transform=None, augmentation=None):
        self.transform = transform
        self.augmentation = augmentation

        self.videos = [(abspath(join(video_path, f)), f) for f in glob.glob(join(video_path, '*.avi'))]
            
        self.clips = []

        for path, vid_idx in tqdm(self.videos):
            clip = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
            
            if clip.size(1) != 5:
                continue
            
            if self.transform:
                clip = self.transform(clip)

            self.clips.append((0, clip, vid_idx))
        
        random.shuffle(self.clips)
        
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
    
    def __getitem__(self, index):
        label, clip, vid_f = self.clips[index]
        if self.augmentation:
            clip = clip.swapaxes(0, 1)
            clip = self.augmentation(clip)
            clip = clip.swapaxes(0, 1)
        return (label, clip, vid_f)
        
    def __len__(self):
        return len(self.clips)
    