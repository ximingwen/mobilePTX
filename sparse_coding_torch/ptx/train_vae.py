import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.ptx.load_data import load_yolo_clips, load_covid_clips
import tensorflow.keras as keras
import tensorflow as tf
from sparse_coding_torch.ptx.classifier_model import VAEEncoderPTX, VAEDecoderPTX
import random
import copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--dataset', default='bamc', type=str)
    parser.add_argument('--crop_height', type=int, default=100)
    parser.add_argument('--crop_width', type=int, default=200)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--clip_depth', type=int, default=5)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    parser.add_argument('--latent_dim', type=int, default=1000)
    

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    crop_height = args.crop_height
    crop_width = args.crop_width

    image_height = int(crop_height / args.scale_factor)
    image_width = int(crop_width / args.scale_factor)
    clip_depth = args.clip_depth

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    data_augmentation = keras.Sequential([
        keras.layers.RandomFlip('horizontal'),
        keras.layers.RandomRotation(45)
    ])

    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))
        
    if args.dataset == 'bamc':
        splits, dataset = load_yolo_clips(args.batch_size, num_clips=1, num_positives=15, mode='all_train', device=device, n_splits=1, sparse_model=None, whole_video=False, positive_videos='positive_videos.json')
    else:
        splits, dataset = load_covid_clips(batch_size=args.batch_size, mode='all_train', clip_width=image_width, clip_height=image_height, clip_depth=clip_depth, n_splits=1)
    train_idx, test_idx = splits[0]
    
    train_loader = copy.deepcopy(dataset)
    train_loader.set_indicies(train_idx)

    train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels()))
    
#     print('Loaded', len(train_loader), 'train examples')

    encoder_inputs = keras.Input(shape=(5, image_height, image_width, 1))
        
    encoder_outputs = VAEEncoderPTX(args.latent_dim)(encoder_inputs)
    
    encoder_model = keras.Model(inputs=encoder_inputs, outputs=encoder_outputs)
    
    decoder_inputs = keras.Input(shape=(args.latent_dim))
    
    decoder_outputs = VAEDecoderPTX()(decoder_inputs)
    
    decoder_model = keras.Model(inputs=decoder_inputs, outputs=decoder_outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    loss_log = []
    best_so_far = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        running_loss = 0.0
        epoch_start = time.perf_counter()
        
        num_iters = 0

        for images, labels in tqdm(train_tf.batch(args.batch_size)):
            images = tf.transpose(images, [0, 2, 3, 4, 1])
            
            with tf.GradientTape() as tape:
                images = tf.reshape(images, (-1, images.shape[2], images.shape[3], images.shape[4]))
                images = data_augmentation(images)
                images = tf.reshape(images, (-1, 5, images.shape[1], images.shape[2], images.shape[3]))
                z, z_mean, z_var = encoder_model(images)
                recon = decoder_model(z)
                reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(images, recon), axis=(1, 2)
                )
                )
                kl_loss = -0.5 * (1 + z_var - tf.square(z_mean) - tf.exp(z_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                loss = reconstruction_loss + kl_loss

            epoch_loss += loss * images.shape[0]
            running_loss += loss * images.shape[0]

            gradients = tape.gradient(loss, encoder_model.trainable_weights + decoder_model.trainable_weights)

            optimizer.apply_gradients(zip(gradients, encoder_model.trainable_weights + decoder_model.trainable_weights))
                
            num_iters += 1

        epoch_end = time.perf_counter()
        epoch_loss /= num_iters

        if epoch_loss < best_so_far:
            print("found better model")
            # Save model parameters
            encoder_model.save(os.path.join(output_dir, "best_encoder.pt"))
            decoder_model.save(os.path.join(output_dir, "best_decoder.pt"))
            best_so_far = epoch_loss

        loss_log.append(epoch_loss)
        print('epoch={}, epoch_loss={:.2f}, time={:.2f}'.format(epoch, epoch_loss, epoch_end - epoch_start))

    plt.plot(loss_log)

    plt.savefig(os.path.join(output_dir, 'loss_graph.png'))
