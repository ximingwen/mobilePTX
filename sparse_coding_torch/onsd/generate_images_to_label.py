from os import listdir
from os.path import isfile
from os.path import join
from os.path import isdir
from os.path import abspath
from os.path import exists
import csv
import glob
import os
from tqdm import tqdm
import torchvision as tv
import cv2
import random
from yolov4.get_bounding_boxes import YoloModel
from matplotlib import pyplot as plt
from matplotlib import cm

yolo_model = YoloModel('onsd')

video_path = "/shared_data/bamc_onsd_data/revised_onsd_data"

labels = [name for name in listdir(video_path) if isdir(join(video_path, name))]
        
count = 0

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

videos = []
for label in labels:
    videos.extend([(label, abspath(join(video_path, label, f)), f) for f in glob.glob(join(video_path, label, '*', '*.mp4'))])

if not os.path.exists('sparse_coding_torch/onsd/individual_frames_cleaned'):
    os.makedirs('sparse_coding_torch/onsd/individual_frames_cleaned')
    
files_to_write = []

vid_idx = 0
for txt_label, path, f_name in tqdm(videos):
    vc = tv.io.read_video(path)[0].permute(3, 0, 1, 2)
    
    label = videos[vid_idx][0]
    f_name = f_name.split('/')[-1]
    
#     print(f_name)
    write_path = os.path.join('sparse_coding_torch/onsd/individual_frames_cleaned', label, f_name[:f_name.rfind('.')])
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    frame_key = path.split('/')[-2]
    if frame_key in valid_frames:
        start_range, end_range = valid_frames[frame_key]

        for j in range(start_range, end_range, 1):
            if j == vc.size(1):
                break
            frame = vc[:, j, :, :]
            
            files_to_write.append((os.path.join(write_path, str(j) + '.png'), frame, label))

#             cv2.imwrite(os.path.join(write_path, str(j) + '.png'), frame.numpy().swapaxes(0,1).swapaxes(1,2))

    vid_idx += 1
    
num_positive = 100
num_negative = 100

curr_positive = 0
curr_negative = 0

random.shuffle(files_to_write)

for path, frame, label in files_to_write:
    if label == 'Positives':
        curr_positive += 1
    else:
        curr_negative += 1
        
print(curr_positive)
print(curr_negative)

# with open('sparse_coding_torch/onsd/individual_frames_cleaned/onsd_labeled_widths_empty.csv', 'w+') as csv_out:
#     out_write = csv.writer(csv_out)
    
#     out_write.writerow(['Video', 'Distance'])
    
#     for path, frame, label in files_to_write:
#         if os.path.exists(path):
#             continue
            
#         orig_height = frame.size(1)
#         orig_width = frame.size(2)

#         bounding_boxes, classes, scores = yolo_model.get_bounding_boxes_v5(frame.swapaxes(0, 2).swapaxes(0, 1).numpy())

#         eye_bounding_box = (None, 0.0)
#         nerve_bounding_box = (None, 0.0)

#         for bb, class_pred, score in zip(bounding_boxes, classes, scores):
#             if class_pred == 0 and score > nerve_bounding_box[1]:
#                 nerve_bounding_box = (bb, score)
#             elif class_pred == 1 and score > eye_bounding_box[1]:
#                 eye_bounding_box = (bb, score)

#         eye_bounding_box = eye_bounding_box[0]
#         nerve_bounding_box = nerve_bounding_box[0]

#         if eye_bounding_box is None or nerve_bounding_box is None:
#             continue

#         nerve_center_x = round((nerve_bounding_box[2] + nerve_bounding_box[0]) / 2 * orig_width)

#         eye_center_y = round(eye_bounding_box[3] * orig_height)

#         crop_center_x = nerve_center_x
#         crop_center_y = eye_center_y + 65

#         if label == 'Positives' and curr_positive < num_positive:
# #             plt.clf()
# #             plt.imshow(frame.numpy().swapaxes(0,1).swapaxes(1,2), cmap=cm.Greys_r)
# #             plt.scatter([crop_center_x], [crop_center_y], color=["red"])
# #             plt.savefig(path)
#             cv2.imwrite(path, frame.numpy().swapaxes(0,1).swapaxes(1,2))
#             out_write.writerow([path])
#             curr_positive += 1
#         elif label == 'Negatives' and curr_negative < num_positive:
# #             plt.clf()
# #             plt.imshow(frame.numpy().swapaxes(0,1).swapaxes(1,2), cmap=cm.Greys_r)
# #             plt.scatter([crop_center_x], [crop_center_y], color=["red"])
# #             plt.savefig(path)
#             cv2.imwrite(path, frame.numpy().swapaxes(0,1).swapaxes(1,2))
#             out_write.writerow([path])
#             curr_negative += 1
            
        