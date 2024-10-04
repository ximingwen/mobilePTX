import torch
import os
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
import scipy.stats
import cv2

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fast', action='store_true',
                    help='optimized for runtime')
    parser.add_argument('--accurate', action='store_true',
                    help='optimized for accuracy')
    parser.add_argument('--verbose', action='store_true',
                    help='output verbose')
    args = parser.parse_args()
    #print(args.accumulate(args.integers))
    device = 'cpu'
    batch_size = 1

    interpreter = tf.lite.Interpreter("keras/mobile_output/tf_lite_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    yolo_model = YoloModel()

    transform = torchvision.transforms.Compose(
    [VideoGrayScaler(),
     # MinMaxScaler(0, 255),
     # torchvision.transforms.Normalize((0.2592,), (0.1251,)),
     torchvision.transforms.CenterCrop((100, 200))
    ])

    all_predictions = []

    all_files = list(os.listdir('/shared_data/bamc_data/PTX_Sliding'))

    for f in all_files:
        print('Processing', f)
        #start_time = time.time()

        clipstride = 15
        if args.fast:
            clipstride = 20
        if args.accurate:
            clipstride = 10

        vc = VideoClips([os.path.join('/shared_data/bamc_data/PTX_Sliding', f)],
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

            bounding_boxes = yolo_model.get_bounding_boxes(clip[:, 2, :, :].swapaxes(0, 2).swapaxes(0, 1)).squeeze(0)
            # for bb in bounding_boxes:
            #     print(bb[1])
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

            # tensor_to_write = trimmed_clip.swapaxes(0, 1).swapaxes(1, 2).swapaxes(2, 3)
            # tensor_to_write[0][0][0][0] = 100
            # tensor_to_write[0][0][0][1] = 100
            # tensor_to_write[0][0][0][2] = 100
            # torchvision.io.write_video('clips_to_test_swift/' + str(countclips) + '.mp4', tensor_to_write, fps=20)
            # countclips += 1
            # trimmed_clip.pin_memory()
            cliplist.append(trimmed_clip)

        if len(cliplist) > 0:
            with torch.no_grad():
                for trimmed_clip in cliplist:
                    interpreter.set_tensor(input_details[0]['index'], trimmed_clip)

                    interpreter.invoke()

                    output_array = np.array(interpreter.get_tensor(output_details[0]['index']))

                    pred = output_array[0][0]
                    print(pred)

                    clip_predictions.append(pred.round())

            if args.verbose:
                print(clip_predictions)
                print("num of clips: ", countclips)

            final_pred = scipy.stats.mode(clip_predictions)[0][0]
            # if len(clip_predictions) % 2 == 0 and torch.sum(clip_predictions).item() == len(clip_predictions)//2:
            #     #print("I'm here")
            #     final_pred = (torch.nn.Sigmoid()(pred)).mean().round().detach().cpu().to(torch.long).item()

            if final_pred == 1:
                str_pred = 'No Sliding'
            else:
                str_pred = 'Sliding'

        else:
            str_pred = "No Sliding"

        end_time = time.time()

        print(str_pred)

        all_predictions.append({'FileName': f, 'Prediction': str_pred, 'TotalTimeSec': end_time - start_time})

    with open('output_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', 'w+', newline='') as csv_out:
        writer = csv.DictWriter(csv_out, fieldnames=all_predictions[0].keys())

        writer.writeheader()
        writer.writerows(all_predictions)
