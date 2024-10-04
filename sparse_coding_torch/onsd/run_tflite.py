import torch
import os
import time
import numpy as np
import torchvision
import csv
from datetime import datetime
from yolov4.get_bounding_boxes import YoloModel
from sparse_coding_torch.onsd.video_loader import get_yolo_region_onsd
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
import argparse
import tensorflow as tf
import scipy.stats
import cv2
import glob
import torchvision as tv
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Python program for processing ONSD data')
    parser.add_argument('--classifier', type=str, default='sparse_coding_torch/mobile_output/onsd.tflite')
    parser.add_argument('--input_dir', default='sparse_coding_torch/onsd/onsd_good_for_eval', type=str)
    parser.add_argument('--image_width', default=200, type=int)
    parser.add_argument('--image_height', default=200, type=int)
    parser.add_argument('--run_2d', default=True, type=bool)
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(args.classifier)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    yolo_model = YoloModel('onsd')

    transform = torchvision.transforms.Compose(
    [torchvision.transforms.Grayscale(1),
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize((args.image_height, args.image_width))
    ])
    
    all_gt = []
    all_preds = []

    for label in ['Positives', 'Negatives']:
        for f in tqdm(os.listdir(os.path.join(args.input_dir, label))):
            if not f.endswith('.png'):
                continue

            frame = torch.tensor(cv2.imread(os.path.join(args.input_dir, label, f))).swapaxes(2, 1).swapaxes(1, 0)

            frame = get_yolo_region_onsd(yolo_model, frame, args.image_width, args.image_height)

            frame = frame[0]

            if args.run_2d:
                frame = transform(frame).to(torch.float32).squeeze().unsqueeze(0).unsqueeze(3).numpy()
            else:
                frame = transform(frame).to(torch.float32).squeeze().unsqueeze(0).unsqueeze(0).unsqueeze(4).numpy()
            
#             cv2.imwrite('testing_tflite_onsd.png', frame[0])
#             print(frame.shape)

            interpreter.set_tensor(input_details[0]['index'], frame)

            interpreter.invoke()

            output_array = np.array(interpreter.get_tensor(output_details[0]['index']))

            pred = output_array[0][0]

            final_pred = float(tf.math.round(pred))
            
            all_preds.append(final_pred)

            if label == 'Positives':
                all_gt.append(1.0)
            elif label == 'Negatives':
                all_gt.append(0.0)
            
    overall_pred = np.array(all_preds)
    overall_true = np.array(all_gt)

    overall_true = np.array(overall_true)
    overall_pred = np.array(overall_pred)
            
    final_f1 = f1_score(overall_true, overall_pred, average='macro')
    final_acc = accuracy_score(overall_true, overall_pred)
    
    print("Final accuracy={:.2f}, f1={:.2f}".format(final_acc, final_f1))