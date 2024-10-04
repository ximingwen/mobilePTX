import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import argparse
import os
from sparse_coding_torch.onsd.load_data import load_onsd_videos
import tensorflow.keras as keras
import tensorflow as tf
from sparse_coding_torch.sparse_model import normalize_weights_3d, normalize_weights, SparseCode, load_pytorch_weights, ReconSparse
import random
from sparse_coding_torch.utils import plot_filters
from yolov4.get_bounding_boxes import YoloModel
import copy

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def sparse_loss(images, recon, activations, batch_size, lam, stride):
    loss = 0.5 * (1/batch_size) * tf.math.reduce_sum(tf.math.pow(images - recon, 2))
    loss += lam * tf.reduce_mean(tf.math.reduce_sum(tf.math.abs(tf.reshape(activations, (batch_size, -1))), axis=1))
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--kernel_width', default=60, type=int)
    parser.add_argument('--kernel_height', default=30, type=int)
    parser.add_argument('--kernel_depth', default=1, type=int)
    parser.add_argument('--num_kernels', default=16, type=int)
    parser.add_argument('--stride', default=1, type=int)
    parser.add_argument('--max_activation_iter', default=300, type=int)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--run_2d', action='store_true')
    parser.add_argument('--save_filters', action='store_true')
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--crop_height', type=int, default=30)
    parser.add_argument('--crop_width', type=int, default=300)
    parser.add_argument('--image_height', type=int, default=30)
    parser.add_argument('--image_width', type=int, default=250)
    parser.add_argument('--clip_depth', type=int, default=1)
    parser.add_argument('--frames_to_skip', type=int, default=1)
    

    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    crop_height = args.crop_height
    crop_width = args.crop_width

    image_height = args.image_height
    image_width = args.image_width
    clip_depth = args.clip_depth
    
    yolo_model = YoloModel('onsd')

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(output_dir, 'arguments.txt'), 'w+') as out_f:
        out_f.write(str(args))

#     splits, dataset = load_onsd_videos(args.batch_size, input_size=(image_height, image_width, clip_depth), mode='all_train')
    splits, dataset = load_onsd_videos(args.batch_size, crop_size=(crop_height, crop_width), yolo_model=yolo_model, mode='all_train', n_splits=1)
    train_idx, test_idx = list(splits)[0]
    
    train_loader = copy.deepcopy(dataset)
    train_loader.set_indicies(train_idx)

    train_tf = tf.data.Dataset.from_tensor_slices((train_loader.get_frames(), train_loader.get_labels(), train_loader.get_widths()))

    if args.run_2d:
        inputs = keras.Input(shape=(image_height, image_width, clip_depth))
    else:
        inputs = keras.Input(shape=(clip_depth, image_height, image_width, 1))
        
    filter_inputs = keras.Input(shape=(clip_depth, args.kernel_height, args.kernel_width, 1, args.num_kernels), dtype='float32')

    output = SparseCode(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_height=args.kernel_height, kernel_width=args.kernel_width, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(inputs, filter_inputs)

    sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
    
    recon_inputs = keras.Input(shape=(1, (image_height - args.kernel_height) // args.stride + 1, (image_width - args.kernel_width) // args.stride + 1, args.num_kernels))
    
    recon_outputs = ReconSparse(batch_size=args.batch_size, image_height=image_height, image_width=image_width, clip_depth=clip_depth, in_channels=1, out_channels=args.num_kernels, kernel_height=args.kernel_height, kernel_width=args.kernel_width, kernel_depth=args.kernel_depth, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d)(recon_inputs)
    
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
        
    crop_amount = (crop_width - image_width)
    assert crop_amount % 2 == 0
    crop_amount = crop_amount // 2
        
    data_augmentation = keras.Sequential([
#         keras.layers.RandomTranslation(0, 0.08),
#         keras.layers.Cropping2D((0, crop_amount))
        keras.layers.Resizing(image_height, image_width)
    ])

    loss_log = []
    best_so_far = float('inf')

    for epoch in range(args.epochs):
        epoch_loss = 0
        running_loss = 0.0
        epoch_start = time.perf_counter()
        
        num_iters = 0
        
        average_activations = []

        for images, labels, width in tqdm(train_tf.shuffle(len(train_tf)).batch(args.batch_size)):
            images = tf.expand_dims(data_augmentation(tf.transpose(images, [0, 2, 3, 1])), axis=1)
                
            activations = tf.stop_gradient(sparse_model([images, tf.stop_gradient(tf.expand_dims(recon_model.trainable_weights[0], axis=0))]))
            
            average_activations.append(float(tf.math.count_nonzero(activations)) / float(tf.math.reduce_prod(tf.shape(activations))))
            
            with tf.GradientTape() as tape:
                recon = recon_model(activations)
                loss = sparse_loss(images, recon, activations, images.shape[0], args.lam, args.stride)

            epoch_loss += loss * images.shape[0]
            running_loss += loss * images.shape[0]

            gradients = tape.gradient(loss, recon_model.trainable_weights)

            filter_optimizer.apply_gradients(zip(gradients, recon_model.trainable_weights))
            
            if args.run_2d:
                weights = normalize_weights(recon_model.get_weights(), args.num_kernels)
            else:
                weights = normalize_weights_3d(recon_model.get_weights(), args.num_kernels)
            recon_model.set_weights(weights)
                
            num_iters += 1

        epoch_end = time.perf_counter()
        
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
        
        sparsity = np.average(np.array(average_activations))
        print('epoch={}, epoch_loss={:.2f}, time={:.2f}, average sparsity={:.2f}'.format(epoch, epoch_loss, epoch_end - epoch_start, sparsity))

    plt.plot(loss_log)

    plt.savefig(os.path.join(output_dir, 'loss_graph.png'))
