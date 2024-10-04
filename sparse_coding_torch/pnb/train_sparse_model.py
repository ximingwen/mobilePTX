import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.pnb.load_data import load_pnb_videos, load_needle_clips
import tensorflow.keras as keras
import tensorflow as tf
from sparse_coding_torch.sparse_model import normalize_weights_3d, normalize_weights, SparseCode, load_pytorch_weights, ReconSparse
import random

def sparse_loss(images, recon, activations, batch_size, lam, stride):
    loss = 0.5 * (1/batch_size) * tf.math.reduce_sum(tf.math.pow(images - recon, 2))
    loss += lam * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(tf.reshape(activations, (batch_size, -1))), axis=1))
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=32, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=300, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run_2d', action='store_true')
    parser.add_argument('--save_filters', action='store_true')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--dataset', default='onsd', type=str)
    parser.add_argument('--crop_height', type=int, default=400)
    parser.add_argument('--crop_width', type=int, default=400)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--clip_depth', type=int, default=5)
    parser.add_argument('--frames_to_skip', type=int, default=1)

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

    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))

    if args.dataset == 'pnb':
        train_loader, test_loader, dataset = load_pnb_videos(args.batch_size, input_size=(image_height, image_width, clip_depth), crop_size=(crop_height, crop_width, clip_depth), classify_mode=False, balance_classes=False, mode='all_train', frames_to_skip=args.frames_to_skip)
    elif args.dataset == 'needle':
        train_loader, test_loader, dataset = load_needle_clips(args.batch_size, input_size=(image_height, image_width, clip_depth))
    else:
        raise Exception('Invalid dataset')
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           sampler=train_sampler)
    
    print('Loaded', len(train_loader), 'train examples')

    example_data = next(iter(train_loader))

    if args.run_2d:
        inputs = keras.Input(shape=(image_height, image_width, 5))
    else:
        inputs = keras.Input(shape=(5, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(5, args.kernel_size, args.kernel_size, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
    
    recon_inputs = keras.Input(shape=(1, (image_height - args.kernel_size) // args.stride + 1, (image_width - args.kernel_size) // args.stride + 1, args.num_kernels))
    
    recon_outputs = ReconSparse(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(recon_inputs)
    
    recon_model = keras.Model(inputs=recon_inputs, outputs=recon_outputs)
    
    if args.save_filters:
        if args.run_2d:
            filters = plot_filters(tf.stack(recon_model.get_weights(), axis=0))
        else:
            filters = plot_filters(recon_model.get_weights()[0])
        filters.save(os.path.join(args.output_dir, 'filters_start.mp4'))

    learning_rate = args.lr
    if args.optimizer == 'sgd':
        filter_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        filter_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_log = []
    best_so_far = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        running_loss = 0.0
        epoch_start = time.perf_counter()
        
        num_iters = 0

        for labels, local_batch, vid_f in tqdm(train_loader):
            if local_batch.size(0) != args.batch_size:
                continue
            if args.run_2d:
                images = local_batch.squeeze(1).permute(0, 2, 3, 1).numpy()
            else:
                images = local_batch.permute(0, 2, 3, 4, 1).numpy()
                
            activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
            
            with tf.GradientTape() as tape:
                recon = recon_model(activations)
                loss = sparse_loss(images, recon, activations, args.batch_size, args.lam, args.stride)

            epoch_loss += loss * local_batch.size(0)
            running_loss += loss * local_batch.size(0)

            gradients = tape.gradient(loss, recon_model.trainable_weights)

            filter_optimizer.apply_gradients(zip(gradients, recon_model.trainable_weights))
            
            if args.run_2d:
                weights = normalize_weights(recon_model.get_weights(), args.num_kernels)
            else:
                weights = normalize_weights_3d(recon_model.get_weights(), args.num_kernels)
            recon_model.set_weights(weights)
                
            num_iters += 1

        epoch_end = time.perf_counter()
        epoch_loss /= len(train_loader.sampler)
        
        if args.save_filters and epoch % 2 == 0:
            if args.run_2d:
                filters = plot_filters(tf.stack(recon_model.get_weights(), axis=0))
            else:
                filters = plot_filters(recon_model.get_weights()[0])
            filters.save(os.path.join(args.output_dir, 'filters_' + str(epoch) +'.mp4'))

        if epoch_loss < best_so_far:
            print("found better model")
            # Save model parameters
            recon_model.save(os.path.join(output_dir, "best_sparse.pt"))
            best_so_far = epoch_loss

        loss_log.append(epoch_loss)
        print('epoch={}, epoch_loss={:.2f}, time={:.2f}'.format(epoch, epoch_loss, epoch_end - epoch_start))

    plt.plot(loss_log)

    plt.savefig(os.path.join(output_dir, 'loss_graph.png'))
