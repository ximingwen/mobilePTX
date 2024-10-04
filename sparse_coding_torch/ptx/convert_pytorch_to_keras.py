import argparse
from tensorflow import keras
import tensorflow as tf
from sparse_coding_torch.sparse_model import SparseCode, ReconSparse, load_pytorch_weights
from sparse_coding_torch.ptx.classifier_model import PTXClassifier
import torch
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_checkpoint', default=None, type=str)
    parser.add_argument('--classifier_checkpoint', default=None, type=str)
    parser.add_argument('--kernel_size', default=15, type=int)
    parser.add_argument('--kernel_depth', default=5, type=int)
    parser.add_argument('--num_kernels', default=64, type=int)
    parser.add_argument('--stride', default=2, type=int)
    parser.add_argument('--input_image_height', default=100, type=int)
    parser.add_argument('--input_image_width', default=200, type=int)
    parser.add_argument('--output_dir', default='./converted_checkpoints', type=str)
    parser.add_argument('--lam', default=0.05, type=float)
    parser.add_argument('--activation_lr', default=1e-2, type=float)
    parser.add_argument('--max_activation_iter', default=100, type=int)
    parser.add_argument('--run_2d', action='store_true')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.classifier_checkpoint:
        classifier_inputs = keras.Input(shape=(1, args.input_image_height // args.stride, args.input_image_width // args.stride, args.num_kernels))

        classifier_outputs = PTXClassifier()(classifier_inputs)
        classifier_name = 'ptx_classifier'

        classifier_model = keras.Model(inputs=classifier_inputs, outputs=classifier_outputs)
        
        pytorch_checkpoint = torch.load(args.classifier_checkpoint, map_location='cpu')['model_state_dict']
        conv_weights = [pytorch_checkpoint['module.compress_activations_conv_1.weight'].squeeze(2).swapaxes(0, 2).swapaxes(1, 3).swapaxes(2, 3).numpy(), pytorch_checkpoint['module.compress_activations_conv_1.bias'].numpy()]
        classifier_model.get_layer(classifier_name).conv_1.set_weights(conv_weights)
        ff_3_weights = [pytorch_checkpoint['module.fc3.weight'].swapaxes(1,0).numpy(), pytorch_checkpoint['module.fc3.bias'].numpy()]
        classifier_model.get_layer(classifier_name).ff_3.set_weights(ff_3_weights)
        ff_4_weights = [pytorch_checkpoint['module.fc4.weight'].swapaxes(1,0).numpy(), pytorch_checkpoint['module.fc4.bias'].numpy()]
        classifier_model.get_layer(classifier_name).ff_4.set_weights(ff_4_weights)
        
        classifier_model.save(os.path.join(args.output_dir, "classifier.pt"))
        
    if args.sparse_checkpoint:
        input_shape = [1, args.input_image_height // args.stride , args.input_image_width // args.stride, args.num_kernels]
        recon_inputs = keras.Input(shape=input_shape)
    
        recon_outputs = ReconSparse(batch_size=1, image_height=args.input_image_height, image_width=args.input_image_width, in_channels=1, out_channels=args.num_kernels, kernel_size=args.kernel_size, stride=args.stride, lam=args.lam, activation_lr=args.activation_lr, max_activation_iter=args.max_activation_iter, run_2d=args.run_2d, padding='SAME')(recon_inputs)

        recon_model = keras.Model(inputs=recon_inputs, outputs=recon_outputs)
        
        pytorch_weights = load_pytorch_weights(args.sparse_checkpoint)
        recon_model.get_layer('recon_sparse').filters = tf.Variable(initial_value=pytorch_weights, dtype='float32', trainable=True)
        
        recon_model.save(os.path.join(args.output_dir, "sparse.pt"))