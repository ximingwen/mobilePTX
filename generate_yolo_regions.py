import torch
import os
import time
import numpy as np
import torchvision
from sparse_coding_torch.video_loader import VideoGrayScaler, MinMaxScaler, get_yolo_regions, classify_nerve_is_right, load_pnb_region_labels, calculate_angle, calculate_angle_video, get_needle_bb
from torchvision.datasets.video_utils import VideoClips
import torchvision as tv
import csv
from datetime import datetime
from yolov4.get_bounding_boxes import YoloModel
import argparse
import tensorflow as tf
import scipy.stats
import cv2
import tensorflow.keras as keras
from sparse_coding_torch.keras_model import SparseCode, PNBClassifier, PTXClassifier, ReconSparse
import glob
from sparse_coding_torch.train_sparse_model import plot_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', required=True, type=str, help='Path to input video.')
    parser.add_argument('--output_dir', default='yolo_output', type=str, help='Location where yolo clips should be saved.')
    parser.add_argument('--num_frames', default=5, type=int)
    parser.add_argument('--stride', default=5, type=int)
    parser.add_argument('--image_height', default=300, type=int)
    parser.add_argument('--image_width', default=400, type=int)
    
    args = parser.parse_args()
    
    path = args.input_video
        
    region_labels = load_pnb_region_labels('sme_region_labels.csv')
                        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_height = args.image_height
    image_width = args.image_width
    clip_depth = args.num_frames
    frames_to_skip = args.stride
    
    # For some reason the size has to be even for the clips, so it will add one if the size is odd
    transforms = torchvision.transforms.Compose([
     torchvision.transforms.Resize(((image_height//2)*2, (image_width//2)*2))
    ])
        
    yolo_model = YoloModel()
        
    vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
    is_right = classify_nerve_is_right(yolo_model, vc)
#     video_angle = calculate_angle_video(yolo_model, vc)
    needle_bb = get_needle_bb(yolo_model, vc)
    person_idx = path.split('/')[-2]
    label = path.split('/')[-3]
    
    output_count = 0
    
    if label == 'Positives' and person_idx in region_labels:
        negative_regions, positive_regions = region_labels[person_idx]
        for sub_region in negative_regions.split(','):
            sub_region = sub_region.split('-')
            start_loc = int(sub_region[0])
#                             end_loc = int(sub_region[1]) - 50
            end_loc = int(sub_region[1]) + 1
            for j in range(start_loc, end_loc - clip_depth * frames_to_skip, clip_depth):
                frames = []
                for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                    frames.append(vc[:, k, :, :])
                vc_sub = torch.stack(frames, dim=1)

                if vc_sub.size(1) < clip_depth:
                    continue

                for clip in get_yolo_regions(yolo_model, vc_sub, is_right, needle_bb, image_width, image_height):
                    clip = transforms(clip)
                    ani = plot_video(clip)
                    ani.save(os.path.join(args.output_dir, 'negative_yolo' + str(output_count) + '.mp4'))
                    print(output_count)
                    output_count += 1

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

                    for clip in get_yolo_regions(yolo_model, vc_sub, is_right, needle_bb, image_width, image_height):
                        clip = transforms(clip)
                        ani = plot_video(clip)
                        ani.save(os.path.join(args.output_dir, 'positive_yolo' + str(output_count) + '.mp4'))
                        print(output_count)
                        output_count += 1

                elif vc.size(1) >= start_loc + clip_depth * frames_to_skip:
                    end_loc = sub_region[1]
                    if end_loc.strip().lower() == 'end':
                        end_loc = vc.size(1)
                    else:
                        end_loc = int(end_loc)
                    for j in range(start_loc, end_loc - clip_depth * frames_to_skip, clip_depth):
                        frames = []
                        for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                            frames.append(vc[:, k, :, :])
                        vc_sub = torch.stack(frames, dim=1)

                        if vc_sub.size(1) < clip_depth:
                            continue

                        for clip in get_yolo_regions(yolo_model, vc_sub, is_right, needle_bb, image_width, image_height):
                            clip = transforms(clip)
                            ani = plot_video(clip)
                            ani.save(os.path.join(args.output_dir, 'positive_yolo' + str(output_count) + '.mp4'))
                            print(output_count)
                            output_count += 1
                else:
                    continue
    elif label == 'Positives':
        frames = []
        for k in range(j, -1 * clip_depth * frames_to_skip, frames_to_skip):
            frames.append(vc[:, k, :, :])
        if frames:
            vc_sub = torch.stack(frames, dim=1)
            if vc_sub.size(1) >= clip_depth:
                for clip in get_yolo_regions(yolo_model, vc_sub, is_right, needle_bb, image_width, image_height):
                    clip = transforms(clip)
                    ani = plot_video(clip)
                    ani.save(os.path.join(args.output_dir, 'positive_yolo' + str(output_count) + '.mp4'))
                    print(output_count)
                    output_count += 1
    elif label == 'Negatives':
        for j in range(0, vc.size(1) - clip_depth * frames_to_skip, clip_depth):
            frames = []
            for k in range(j, j + clip_depth * frames_to_skip, frames_to_skip):
                frames.append(vc[:, k, :, :])
            vc_sub = torch.stack(frames, dim=1)

            if vc_sub.size(1) >= clip_depth:
                for clip in get_yolo_regions(yolo_model, vc_sub, is_right, needle_bb, image_width, image_height):
                    clip = transforms(clip)
                    ani = plot_video(clip)
                    ani.save(os.path.join(args.output_dir, 'negative_yolo' + str(output_count) + '.mp4'))
                    print(output_count)
                    output_count += 1
    else:
        raise Exception('Invalid label')