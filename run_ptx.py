import torch
import os
from sparse_coding_torch.keras_model import SparseCode, PNBClassifier, PTXClassifier, ReconSparse
import time
import numpy as np
import torchvision
from sparse_coding_torch.video_loader import VideoGrayScaler, MinMaxScaler
from torchvision.datasets.video_utils import VideoClips
import csv
from datetime import datetime
from yolov4.get_bounding_boxes import YoloModel
import argparse
import tensorflow as tf
import tensorflow.keras as keras


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/shared_data/bamc_data/PTX_Sliding', type=str)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=64, type=int)
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--max_activation_iter', default=100, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--sparse_checkpoint', default='ptx_tensorflow/sparse.pt', type=str)
    parser.add_argument('--checkpoint', default='ptx_tensorflow/best_classifier.pt', type=str)
    parser.add_argument('--run_2d', action='store_true')
    parser.add_argument('--dataset', default='ptx', type=str)
    
    args = parser.parse_args()
    #print(args.accumulate(args.integers))
    batch_size = 1
    
    if args.dataset == 'ptx':
        image_height = 100
        image_width = 200
        clip_depth = 5
    else:
        raise Exception('Invalid dataset')

    if args.run_2d:
        inputs = keras.Input(shape=(image_height, image_width, 5))
    else:
        inputs = keras.Input(shape=(5, image_height, image_width, 1))

    filter_inputs = keras.Input(shape=(5, args.kernel_size, args.kernel_size, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d, padding='SAME')(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)

    if args.sparse_checkpoint:
        recon_model = keras.models.load_model(args.sparse_checkpoint)
        
    if args.checkpoint:
        classifier_model = keras.models.load_model(args.checkpoint)
    else:
        classifier_inputs = keras.Input(shape=(1, image_height // args.stride, image_width // args.stride, args.num_kernels))

        if args.dataset == 'pnb':
            classifier_outputs = PNBClassifier()(classifier_inputs)
        elif args.dataset == 'ptx':
            classifier_outputs = PTXClassifier()(classifier_inputs)
        else:
            raise Exception('No classifier exists for that dataset')

        classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)
        
    yolo_model = YoloModel()
        
    transform = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     MinMaxScaler(0, 255),
     torchvision.transforms.Normalize((0.2592,), (0.1251,)),
     torchvision.transforms.CenterCrop((100, 200))
    ])
    
    all_predictions = []
    
    all_files = list(os.listdir(args.input_dir))
    
    for f in all_files:
        print('Processing', f)
        #start_time = time.time()
        
        clipstride = 15
        
        vc = VideoClips([os.path.join(args.input_dir, f)],
                        clip_length_in_frames=5,
                        frame_rate=20,
                       frames_between_clips=clipstride)
    
        ### START time after loading video ###
        start_time = time.time()
        clip_predictions = []
        i = 0
        cliplist = []
        countclips = 0
        for i in range(vc.num_clips()):

            clip, _, _, _ = vc.get_clip(i)
            clip = clip.swapaxes(1, 3).swapaxes(0, 1).swapaxes(2, 3).numpy()
            
            bounding_boxes, classes = yolo_model.get_bounding_boxes(clip[:, 2, :, :].swapaxes(0, 2).swapaxes(0, 1))
            bounding_boxes = bounding_boxes.squeeze(0)
            if bounding_boxes.size == 0:
                continue
            #widths = []
            countclips = countclips + len(bounding_boxes)
            
            widths = [(bounding_boxes[i][3] - bounding_boxes[i][1]) for i in range(len(bounding_boxes))]
            
            #for i in range(len(bounding_boxes)):
            #    widths.append(bounding_boxes[i][3] - bounding_boxes[i][1])

            ind =  np.argmax(np.array(widths))
            #for bb in bounding_boxes:
            bb = bounding_boxes[ind]
            center_x = (bb[3] + bb[1]) / 2 * 1920
            center_y = (bb[2] + bb[0]) / 2 * 1080

            width=400
            height=400

            lower_y = round(center_y - height / 2)
            upper_y = round(center_y + height / 2)
            lower_x = round(center_x - width / 2)
            upper_x = round(center_x + width / 2)

            trimmed_clip = clip[:, :, lower_y:upper_y, lower_x:upper_x]

            trimmed_clip = torch.tensor(trimmed_clip).to(torch.float)

            trimmed_clip = transform(trimmed_clip)
            trimmed_clip.pin_memory()
            cliplist.append(trimmed_clip)

        if len(cliplist) > 0:
            with torch.no_grad():
                trimmed_clip = torch.stack(cliplist)
                images = trimmed_clip.permute(0, 2, 3, 4, 1).numpy()
                activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.weights[0], axis=0))]))

                pred = classifier_model(activations)
                #print(torch.nn.Sigmoid()(pred))
                clip_predictions = tf.math.round(tf.math.sigmoid(pred))

            final_pred = torch.mode(torch.tensor(clip_predictions.numpy()).view(-1))[0].item()
            if len(clip_predictions) % 2 == 0 and tf.math.reduce_sum(clip_predictions) == len(clip_predictions)//2:
                #print("I'm here")
                final_pred = torch.mode(torch.tensor(clip_predictions.numpy()).view(-1))[0].item()
                
            if final_pred == 1:
                str_pred = 'No Sliding'
            else:
                str_pred = 'Sliding'

        else:
            str_pred = "No Sliding"
            
        print(str_pred)
            
        end_time = time.time()
        
        all_predictions.append({'FileName': f, 'Prediction': str_pred, 'TotalTimeSec': end_time - start_time})
        
    with open('output_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', 'w+', newline='') as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=all_predictions[0].keys())
        
        writer.writeheader()
        writer.writerows(all_predictions)
