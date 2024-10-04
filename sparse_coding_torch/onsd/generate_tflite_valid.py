from tensorflow import keras
import numpy as np
import torch
import tensorflow as tf
import cv2
import torchvision as tv
import torch
import torch.nn as nn
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
from sparse_coding_torch.onsd.classifier_model import MobileModelONSD
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='sparse_coding_torch/onsd/valid_frame_model_2/best_classifier.pt/', type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--image_width', type=int, default=512)
    
    args = parser.parse_args()
    #print(args.accumulate(args.integers))
    batch_size = args.batch_size

    image_height = args.image_height
    image_width = args.image_width
        
    classifier_model = keras.models.load_model(args.checkpoint)

    input_name = classifier_model.input_names[0]
    index = classifier_model.input_names.index(input_name)
    classifier_model.inputs[index].set_shape([batch_size, image_height, image_width, 3])

    converter = tf.lite.TFLiteConverter.from_keras_model(classifier_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    print('Converted')

    with open("./sparse_coding_torch/mobile_output/onsd_valid.tflite", "wb") as f:
        f.write(tflite_model)
