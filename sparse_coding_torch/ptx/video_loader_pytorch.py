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

def get_ptx_participants():
    video_to_participant = {}
    with open('/shared_data/bamc_data/bamc_video_info.csv', 'r') as csv_in:
        reader = csv.DictReader(csv_in)
        for row in reader:
            key = row['Filename'].split('.')[0].lower().replace('_clean', '')
            if key == '37 (mislabeled as 38)':
                key = '37'
            video_to_participant[key] = row['Participant_id']
            
    return video_to_participant

class MinMaxScaler(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __init__(self, min_val=0, max_val=254):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor):
        return (tensor - self.min_val) / (self.max_val - self.min_val)

class VideoGrayScaler(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.grayscale = tv.transforms.Grayscale(num_output_channels=1)
        
    def forward(self, video):
        # shape = channels, time, width, height
        video = self.grayscale(video.swapaxes(-4, -3).swapaxes(-2, -1))
        video = video.swapaxes(-4, -3).swapaxes(-2, -1)
        # print(video.shape)
        return video
    
class YoloClipLoader(Dataset):
    
    def __init__(self, yolo_output_path, num_frames=5, frames_between_clips=None,
                 transform=None, augment_transform=None, num_clips=1, num_positives=1, positive_videos=None, sparse_model=None, device=None):
        if (num_frames % 2) == 0:
            raise ValueError("Num Frames must be an odd number, so we can extract a clip centered on each detected region")
        
        clip_cache_file = 'clip_cache_pytorch.pt'
        
        self.num_clips = num_clips
        
        self.num_frames = num_frames
        if frames_between_clips is None:
            self.frames_between_clips = num_frames
        else:
            self.frames_between_clips = frames_between_clips

        self.transform = transform
        self.augment_transform = augment_transform
         
        self.labels = [name for name in listdir(yolo_output_path) if isdir(join(yolo_output_path, name))]
        self.clips = []
        if exists(clip_cache_file):
            self.clips = torch.load(open(clip_cache_file, 'rb'))
        else:
            for label in self.labels:
                print("Processing videos in category: {}".format(label))
                videos = list(listdir(join(yolo_output_path, label)))
                for vi in tqdm(range(len(videos))):
                    video = videos[vi]
                    counter = 0
                    all_trimmed = []
                    with open(abspath(join(yolo_output_path, label, video, 'result.json'))) as fin:
                        results = json.load(fin)
                        max_frame = len(results)

                        for i in range((num_frames-1)//2, max_frame - (num_frames-1)//2 - 1, self.frames_between_clips):
                        # for frame in results:
                            frame = results[i]
                            # print('loading frame:', i, frame['frame_id'])
                            frame_start = int(frame['frame_id']) - self.num_frames//2
                            frames = [abspath(join(yolo_output_path, label, video, 'frame{}.png'.format(frame_start+fid)))
                                      for fid in range(num_frames)]
                            # print(frames)
                            frames = torch.stack([ToTensor()(Image.open(f).convert("RGB")) for f in frames]).swapaxes(0, 1)

                            for region in frame['objects']:
                                # print(region)
                                if region['name'] != "Pleural_Line":
                                    continue

                                center_x = region['relative_coordinates']["center_x"] * 1920
                                center_y = region['relative_coordinates']['center_y'] * 1080

                                # width = region['relative_coordinates']['width'] * 1920
                                # height = region['relative_coordinates']['height'] * 1080
                                width=200
                                height=100

                                lower_y = round(center_y - height / 2)
                                upper_y = round(center_y + height / 2)
                                lower_x = round(center_x - width / 2)
                                upper_x = round(center_x + width / 2)

                                final_clip = frames[:, :, lower_y:upper_y, lower_x:upper_x]

                                if self.transform:
                                    final_clip = self.transform(final_clip)

                                if sparse_model:
                                    with torch.no_grad():
                                        final_clip = final_clip.unsqueeze(0).to(device)
                                        final_clip = sparse_model(final_clip)
                                        final_clip = final_clip.squeeze(0).detach().cpu()

                                self.clips.append((label, final_clip, video))

            torch.save(self.clips, open(clip_cache_file, 'wb+'))
            
            
#         random.shuffle(self.clips)
            
#         video_to_clips = {}
        if positive_videos:
            vids_to_keep = json.load(open(positive_videos))[:num_positives]
            
            self.clips = [clip_tup for clip_tup in self.clips if clip_tup[2] in vids_to_keep or clip_tup[0] == 'Sliding']
        else:
            video_to_labels = {}

            for lbl, clip, video in self.clips:
                video = video.lower().replace('_clean', '')
                if video not in video_to_labels:
    #                 video_to_clips[video] = []
                    video_to_labels[video] = []

    #             video_to_clips[video].append(clip)
                video_to_labels[video].append(lbl)

            video_to_participants = get_ptx_participants()
            participants_to_video = {}
            for k, v in video_to_participants.items():
                if video_to_labels[k][0] == 'Sliding':
                    continue
                if not v in participants_to_video:
                    participants_to_video[v] = []

                participants_to_video[v].append(k)

            participants_to_video = dict(sorted(participants_to_video.items(), key=lambda x: len(x[1]), reverse=True))

            num_to_remove = len([k for k,v in video_to_labels.items() if v[0] == 'No_Sliding']) - num_positives
            vids_to_remove = set()
            while num_to_remove > 0:
                vids_to_remove.add(participants_to_video[list(participants_to_video.keys())[0]].pop())
                participants_to_video = dict(sorted(participants_to_video.items(), key=lambda x: len(x[1]), reverse=True))
                num_to_remove -= 1
                    
            self.clips = [clip_tup for clip_tup in self.clips if clip_tup[2].lower().replace('_clean', '') not in vids_to_remove]
        
        video_to_clips = {}
        video_to_labels = {}

        for lbl, clip, video in self.clips:
            if video not in video_to_clips:
                video_to_clips[video] = []
                video_to_labels[video] = []

            video_to_clips[video].append(clip)
            video_to_labels[video].append(lbl)
            
        print([k for k,v in video_to_labels.items() if v[0] == 'No_Sliding'])
            
        print('Num positive:', len([k for k,v in video_to_labels.items() if v[0] == 'No_Sliding']))
        print('Num negative:', len([k for k,v in video_to_labels.items() if v[0] == 'Sliding']))

        self.videos = None
        self.max_video_clips = 0
        if num_clips > 1:
            self.videos = []

            for video in video_to_clips.keys():
                clip_list = video_to_clips[video]
                lbl_list = video_to_labels[video]
                
                for i in range(0, len(clip_list) - num_clips, 1):
                    video_stack = torch.stack(clip_list[i:i+num_clips])
                
                    self.videos.append((max(set(lbl_list[i:i+num_clips]), key=lbl_list[i:i+num_clips].count), video_stack, video))
            
            self.clips = None

            
    def get_labels(self):
        if self.num_clips > 1:
            return [self.videos[i][0] for i in range(len(self.videos))]
        else:
            return [self.clips[i][0] for i in range(len(self.clips))]
    
    def get_filenames(self):
        if self.num_clips > 1:
            return [self.videos[i][2] for i in range(len(self.videos))]
        else:
            return [self.clips[i][2] for i in range(len(self.clips))]
    
    def __getitem__(self, index): 
        if self.num_clips > 1:
            label = self.videos[index][0]
            video = self.videos[index][1]
            filename = self.videos[index][2]
            
            video = video.squeeze(2)
            video = video.permute(1, 0, 2, 3)

            if self.augment_transform:
                video = self.augment_transform(video)
                
            video = video.unsqueeze(2)
            video = video.permute(1, 0, 2, 3, 4)
#             video = video.permute(4, 1, 2, 3, 0)
#             video = torch.nn.functional.pad(video, (0), 'constant', 0)
#             video = video.permute(4, 1, 2, 3, 0)

            orig_len = video.size(0)

#             if orig_len < self.max_video_clips:
#                 video = torch.cat([video, torch.zeros(self.max_video_clips - len(video), video.size(1), video.size(2), video.size(3), video.size(4))])

            return label, video, filename, orig_len
        else:
            label = self.clips[index][0]
            video = self.clips[index][1]
            filename = self.clips[index][2]

            if self.augment_transform:
                video = self.augment_transform(video)

            return label, video, filename
        
    def __len__(self):
        return len(self.clips)
    
def get_yolo_regions(yolo_model, clip):
    orig_height = clip.size(2)
    orig_width = clip.size(3)
    bounding_boxes, classes, scores = yolo_model.get_bounding_boxes(clip[:, 2, :, :].swapaxes(0, 2).swapaxes(0, 1).numpy())
    bounding_boxes = bounding_boxes.squeeze(0)
    classes = classes.squeeze(0)
    scores = scores.squeeze(0)
    
    all_clips = []
    for bb, class_pred, score in zip(bounding_boxes, classes, scores):
        lower_y = round((bb[0] * orig_height))
        upper_y = round((bb[2] * orig_height))
        lower_x = round((bb[1] * orig_width))
        upper_x = round((bb[3] * orig_width))

        trimmed_clip = clip[:, :, lower_y:upper_y, lower_x:upper_x]
        
        if trimmed_clip.shape[2] == 0 or trimmed_clip.shape[3] == 0:
            continue
        all_clips.append(torch.tensor(trimmed_clip))

    return all_clips

class COVID19Loader(Dataset):
    def __init__(self, yolo_model, video_path, clip_depth, classify_mode=False, transform=None, augmentation=None):
        self.transform = transform
        self.augmentation = augmentation
        
        self.videos = glob.glob(join(video_path, '*', '*.*'))
        
        vid_to_label = {}
        with open('/home/dwh48@drexel.edu/covid19_ultrasound/data/dataset_metadata.csv') as csv_in:
            reader = csv.DictReader(csv_in)
            for row in reader:
                vid_to_label[row['Filename']] = row['Label']
            
        self.clips = []
        
        vid_idx = 0
        for path in tqdm(self.videos):
            vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
            label = vid_to_label[path.split('/')[-1].split('.')[0]]
            
            if classify_mode:
                for j in range(0, vc.size(1) - clip_depth, clip_depth):
                    vc_sub = vc[:, j:j+clip_depth, :, :]
                    if vc_sub.size(1) < clip_depth:
                        continue
                    for clip in get_yolo_regions(yolo_model, vc_sub):
                        if self.transform:
                            clip = self.transform(clip)

                        self.clips.append((label, clip, path))
            else:
                for j in range(0, vc.size(1) - clip_depth, clip_depth):
                    vc_sub = vc[:, j:j+clip_depth, :, :]
                    if vc_sub.size(1) != clip_depth:
                        continue
                    if self.transform:
                        vc_sub = self.transform(vc_sub)

                    self.clips.append((label, vc_sub, path))

            vid_idx += 1
        
        random.shuffle(self.clips)
        
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def __getitem__(self, index):
        label, clip, vid_f = self.clips[index]
        if self.augmentation:
            clip = clip.swapaxes(0, 1)
            clip = self.augmentation(clip)
            clip = clip.swapaxes(0, 1)
        return (label, clip, vid_f)
        
    def __len__(self):
        return len(self.clips)