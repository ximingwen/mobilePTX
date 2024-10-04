import torch
import os
import time
import numpy as np
import torchvision
from sparse_coding_torch.video_loader import VideoGrayScaler, MinMaxScaler, get_yolo_regions, classify_nerve_is_right
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/shared_data/bamc_pnb_data/revised_training_data', type=str)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=48, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=150, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--sparse_checkpoint', default='sparse_coding_torch/output/sparse_pnb_48/sparse_conv3d_model-best.pt/', type=str)
    parser.add_argument('--checkpoint', default='sparse_coding_torch/classifier_outputs/48_filters_grouped/best_classifier.pt/', type=str)
    parser.add_argument('--run_2d', action='store_true')
    
    args = parser.parse_args()
    #print(args.accumulate(args.integers))
    batch_size = 1

    image_height = 285
    image_width = 235
    clip_depth = 5

    if args.run_2d:
        inputs = keras.Input(shape=(image_height, image_width, 5))
    else:
        inputs = keras.Input(shape=(5, image_height, image_width, 1))

    filter_inputs = keras.Input(shape=(5, args.kernel_size, args.kernel_size, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d, padding='VALID')(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)

    recon_model = keras.models.load_model(args.sparse_checkpoint)
        
    classifier_model = keras.models.load_model(args.checkpoint)
        
    yolo_model = YoloModel()

    transform = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     MinMaxScaler(0, 255),
     torchvision.transforms.Resize((image_height, image_width))
    ])

    all_predictions = []

    all_files = glob.glob(pathname=os.path.join(args.input_dir, '**', '*.mp4'), recursive=True)

    for f in all_files:
        print('Processing', f)
        
        vc = tv.io.read_video(f)[0].permute(3, 0, 1, 2)
        is_right = classify_nerve_is_right(yolo_model, vc)
        
        all_preds = []
        for i in range(1, 5):
            vc_sub = vc[:, -5*(i+1):-5*i, :, :]
            if vc_sub.size(1) < 5:
                print(f + ' does not contain enough frames for processing')
                continue
            
            ### START time after loading video ###
            start_time = time.time()
            clip = None
        
            clip = get_yolo_regions(yolo_model, vc_sub, is_right, crop_width=image_width, crop_height=image_height)
            
            if clip:
                clip = clip[0]
                clip = transform(clip)
                clip = tf.expand_dims(clip, axis=4) 

                activations = tf.stop_gradient(sparse_model([clip, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))

                pred = tf.math.sigmoid(classifier_model(activations))
                
                all_preds.append(pred)
                
        if all_preds:
            final_pred = np.round(np.mean(np.array(all_preds)))
            if final_pred == 1:
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
