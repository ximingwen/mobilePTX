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
                 transform=None, num_clips=1, num_positives=1, positive_videos=None, sparse_model=None, device=None):
        if (num_frames % 2) == 0:
            raise ValueError("Num Frames must be an odd number, so we can extract a clip centered on each detected region")
        
        clip_cache_file = 'clip_cache.pt'
        
        self.num_clips = num_clips
        
        self.num_frames = num_frames
        if frames_between_clips is None:
            self.frames_between_clips = num_frames
        else:
            self.frames_between_clips = frames_between_clips

        self.transform = transform
         
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
                                        
                                if label == 'No_Sliding':
                                    y = np.array(1.0)
                                elif label == 'Sliding':
                                    y = np.array(0.0)

                                self.clips.append((y, final_clip.numpy(), video))

            torch.save(self.clips, open(clip_cache_file, 'wb+'))
            
            
#         random.shuffle(self.clips)
            
#         video_to_clips = {}
        if positive_videos:
            vids_to_keep = json.load(open(positive_videos))
            
            self.clips = [clip_tup for clip_tup in self.clips if clip_tup[2] in vids_to_keep or clip_tup[0] == 0.0]
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
                if video_to_labels[k][0] == 0.0:
                    continue
                if not v in participants_to_video:
                    participants_to_video[v] = []

                participants_to_video[v].append(k)

            participants_to_video = dict(sorted(participants_to_video.items(), key=lambda x: len(x[1]), reverse=True))

            num_to_remove = len([k for k,v in video_to_labels.items() if v[0] == 1.0]) - num_positives
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
            
        print([k for k,v in video_to_labels.items() if v[0] == 1.0])
            
        print('Num positive:', len([k for k,v in video_to_labels.items() if v[0] == 1.0]))
        print('Num negative:', len([k for k,v in video_to_labels.items() if v[0] == 0.0]))

#         self.videos = None
#         self.max_video_clips = 0
#         if num_clips > 1:
#             self.videos = []

#             for video in video_to_clips.keys():
#                 clip_list = video_to_clips[video]
#                 lbl_list = video_to_labels[video]
                
#                 for i in range(0, len(clip_list) - num_clips, 1):
#                     video_stack = torch.stack(clip_list[i:i+num_clips])
                
#                     self.videos.append((max(set(lbl_list[i:i+num_clips]), key=lbl_list[i:i+num_clips].count), video_stack, video))
            
#             self.clips = None
            
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

            
#     def get_labels(self):
#         if self.num_clips > 1:
#             return [self.videos[i][0] for i in range(len(self.videos))]
#         else:
#             return [self.clips[i][0] for i in range(len(self.clips))]
    
#     def get_filenames(self):
#         if self.num_clips > 1:
#             return [self.videos[i][2] for i in range(len(self.videos))]
#         else:
#             return [self.clips[i][2] for i in range(len(self.clips))]
    
#     def __getitem__(self, index): 
#         if self.num_clips > 1:
#             label = self.videos[index][0]
#             video = self.videos[index][1]
#             filename = self.videos[index][2]
            
#             video = video.squeeze(2)
#             video = video.permute(1, 0, 2, 3)

#             if self.augment_transform:
#                 video = self.augment_transform(video)
                
#             video = video.unsqueeze(2)
#             video = video.permute(1, 0, 2, 3, 4)
# #             video = video.permute(4, 1, 2, 3, 0)
# #             video = torch.nn.functional.pad(video, (0), 'constant', 0)
# #             video = video.permute(4, 1, 2, 3, 0)

#             orig_len = video.size(0)

# #             if orig_len < self.max_video_clips:
# #                 video = torch.cat([video, torch.zeros(self.max_video_clips - len(video), video.size(1), video.size(2), video.size(3), video.size(4))])

#             return label, video, filename, orig_len
#         else:
#             label = self.clips[index][0]
#             video = self.clips[index][1]
#             filename = self.clips[index][2]

#             if self.augment_transform:
#                 video = self.augment_transform(video)

#             return label, video, filename
        
#     def __len__(self):
#         return len(self.clips)
    
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
    def __init__(self, video_path, clip_depth, transform=None):
        self.transform = transform
        
        self.videos = glob.glob(join(video_path, '*', '*.*'))
        
        self.vid_to_label = {}
        self.vid_to_patient = {}
        with open('/home/dwh48@drexel.edu/covid19_ultrasound/data/dataset_metadata.csv') as csv_in:
            reader = csv.DictReader(csv_in)
            for row in reader:
                self.vid_to_label[row['Filename']] = row['Label'].lower()
                self.vid_to_patient[row['Filename']] = row['Patient ID / Name']
                if self.vid_to_patient[row['Filename']].strip() == '':
                    self.vid_to_patient[row['Filename']] = row['Filename']
                
        all_labels = set(self.vid_to_label.values())
        self.label_to_id = {v: i for i, v in enumerate(all_labels)}
            
        self.clips = []
        
        vid_idx = 0
        for path in tqdm(self.videos):
            vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
            patient = self.vid_to_patient[path.split('/')[-1].split('.')[0]]
            label = self.vid_to_label[path.split('/')[-1].split('.')[0]]
            label = self.label_to_id[label]
            
            for j in range(0, vc.size(1) - clip_depth, clip_depth):
                vc_sub = vc[:, j:j+clip_depth, :, :]
                if vc_sub.size(1) != clip_depth:
                    continue
                if self.transform:
                    vc_sub = self.transform(vc_sub)

                self.clips.append((np.array(label), vc_sub.numpy(), path, patient))

            vid_idx += 1
        
        random.shuffle(self.clips)
        
    def get_groups(self):
        return [self.clips[i][3] for i in range(len(self.clips))]
        
    def get_filenames(self):
        return [self.clips[i][2] for i in range(len(self.clips))]
    
    def get_all_videos(self):
        return set([self.clips[i][2] for i in range(len(self.clips))])
    
    def get_video_labels(self):
        return [self.label_to_id[self.vid_to_label[vid.split('/')[-1].split('.')[0]]] for vid in self.get_all_videos()]
        
    def get_labels(self):
        return [self.clips[i][0] for i in range(len(self.clips))]
    
    def get_unique_labels(self):
        return [int(self.clips[i][0]) for i in range(len(self.clips))]
    
    def set_indicies(self, iter_idx):
        new_clips = []
        for i, clip in enumerate(self.clips):
            if i in iter_idx:
                new_clips.append(clip)
                
        self.clips = new_clips
        
    def get_frames(self):
        return [frame for _, frame, _, _ in self.clips]
    
    def __next__(self):
        if self.count < len(self.clips):
            label, frame, vid_f, patient = self.clips[self.count]
            self.count += 1
            return label, frame
        else:
            raise StopIteration
            
    def __iter__(self):
        return self