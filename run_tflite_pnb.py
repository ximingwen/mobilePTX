import torch
import os
import time
import numpy as np
import torchvision
from sparse_coding_torch.video_loader import VideoGrayScaler, MinMaxScaler, get_yolo_regions, classify_nerve_is_right
from torchvision.datasets.video_utils import VideoClips
import csv
from datetime import datetime
from yolov4.get_bounding_boxes import YoloModel
import argparse
import tensorflow as tf
import scipy.stats
import cv2
import glob
import torchvision as tv

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Python program for processing PNB data')
    parser.add_argument('--classifier', type=str, default='sparse_coding_torch/mobile_output/pnb.tflite')
    parser.add_argument('--input_dir', default='/shared_data/bamc_pnb_data/revised_training_data', type=str)
    parser.add_argument('--stride', default=30, type=int)
    parser.add_argument('--image_width', default=400, type=int)
    parser.add_argument('--image_height', default=285, type=int)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(args.classifier)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    yolo_model = YoloModel()

    transform = torchvision.transforms.Compose(
    [VideoGrayScaler(),
#      MinMaxScaler(0, 255),
     torchvision.transforms.Resize((args.image_height, args.image_width))
    ])

    all_predictions = []

    all_files = glob.glob(pathname=os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)

    for f in all_files:
        print('Processing', f)
        
        vc = tv.io.read_video(f)[0].permute(3, 0, 1, 2)
        is_right = classify_nerve_is_right(yolo_model, vc)
        
        all_preds = []
        
        for j in range(0, vc.size(1) - 5, args.stride):
            vc_sub = vc[:, j:j+5, :, :]
            
            if vc_sub.size(1) < 5:
                continue
            
            ### START time after loading video ###
            start_time = time.time()
            
            clip = get_yolo_regions(yolo_model, vc_sub, is_right, args.image_width, args.image_height)
            
            if not clip:
                continue

            clip = clip[0]
            clip = transform(clip).to(torch.float32)

            interpreter.set_tensor(input_details[0]['index'], clip)

            interpreter.invoke()

            output_array = np.array(interpreter.get_tensor(output_details[0]['index']))

            pred = output_array[0][0]

            final_pred = pred.round()
            
            all_preds.append(final_pred)
            
        print(all_preds)
            
            
        if all_preds[-5:-2]:
            video_pred = np.round(sum(all_preds[-5:-2]) / len(all_preds[-5:-2]))

            if video_pred == 1:
                str_pred = 'Positive'
            else:
                str_pred = 'Negative'
        else:
            str_pred = "Positive"

        end_time = time.time()

        print(str_pred)

        all_predictions.append({'FileName': f, 'Prediction': str_pred, 'TotalTimeSec': end_time - start_time})

    with open('output_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', 'w+', newline='') as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=all_predictions[0].keys())

        writer.writeheader()
        writer.writerows(all_predictions)
