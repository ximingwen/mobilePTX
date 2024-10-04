from tensorflow import keras
import numpy as np
import torch
import tensorflow as tf
import cv2
import torchvision as tv
import torch
import torch.nn as nn
from sparse_coding_torch.utils import VideoGrayScaler, MinMaxScaler
from sparse_coding_torch.pnb.classifier_model import MobileModelPNB
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/shared_data/bamc_pnb_data/revised_training_data', type=str)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=32, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--max_activation_iter', default=150, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--sparse_checkpoint', default='sparse_coding_torch/output/sparse_pnb_32_long_train/sparse_conv3d_model-best.pt/', type=str)
    parser.add_argument('--checkpoint', default='sparse_coding_torch/classifier_outputs/32_filters_no_aug_3/best_classifier.pt/', type=str)
    parser.add_argument('--run_2d', action='store_true')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--image_height', type=int, default=285)
    parser.add_argument('--image_width', type=int, default=400)
    parser.add_argument('--clip_depth', type=int, default=5)
    
    args = parser.parse_args()
    #print(args.accumulate(args.integers))
    batch_size = args.batch_size

    image_height = args.image_height
    image_width = args.image_width
    clip_depth = args.clip_depth
    
    recon_model = keras.models.load_model(args.sparse_checkpoint)
        
    classifier_model = keras.models.load_model(args.checkpoint)

    inputs = keras.Input(shape=(5, image_height, image_width))

    outputs = MobileModelPNB(sparse_weights=recon_model.weights[0], classifier_model=classifier_model, batch_size=batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(inputs)

    model = keras.Model(inputs=inputs, outputs=outputs)

    input_name = model.input_names[0]
    index = model.input_names.index(input_name)
    model.inputs[index].set_shape([1, 5, image_height, image_width])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()

    print('Converted')

    with open("./sparse_coding_torch/mobile_output/pnb.tflite", "wb") as f:
        f.write(tflite_model)
