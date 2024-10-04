import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf

def get_model(img_size, num_classes, filter_size):
    inputs = keras.Input(shape=img_size + (1,))
    
    x = keras.layers.RandomTranslation(0.1, 0.1)(inputs)
    x = keras.layers.RandomRotation(0.1)(x)
    x = keras.layers.RandomFlip('horizontal')(x)
    x = keras.layers.RandomContrast(0.02)(x)
    x = keras.layers.RandomBrightness(0.02)(x)

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, filter_size, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, filter_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, filter_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(filter_size, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, filter_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, filter_size, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, filter_size, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    
    return model

class ONSDPositionalConv(keras.layers.Layer):
    def __init__(self):
        super(ONSDPositionalConv, self).__init__()
        
#         self.sparse_filters = tf.squeeze(keras.models.load_model(sparse_checkpoint).weights[0], axis=0)

        self.conv_1 = keras.layers.Conv2D(32, kernel_size=(8, 8), strides=(2), activation='relu', padding='valid')
        self.conv_2 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
        self.conv_3 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
        self.conv_4 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
        self.conv_5 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(2), activation='relu', padding='valid')
#         self.conv_6 = keras.layers.Conv2D(32, kernel_size=(4, 4), strides=(12), activation='relu', padding='valid')
#         self.conv_1 = keras.layers.Conv1D(10, kernel_size=3, strides=1, activation='relu', padding='valid')
#         self.conv_2 = keras.layers.Conv1D(10, kernel_size=3, strides=1, activation='relu', padding='valid')

        self.flatten = keras.layers.Flatten()

        self.dropout = keras.layers.Dropout(0.20)
        
        self.ff_dropout = keras.layers.Dropout(0.1)

#         self.ff_1 = keras.layers.Dense(1000, activation='relu', use_bias=True)
#         self.ff_2 = keras.layers.Dense(500, activation='relu', use_bias=True)
        self.ff_2 = keras.layers.Dense(100, activation='relu', use_bias=True)
        self.ff_3 = keras.layers.Dense(20, activation='relu', use_bias=True)
        self.ff_final_1 = keras.layers.Dense(1)
        self.ff_final_2 = keras.layers.Dense(1)
        self.do_dropout = True

#     @tf.function
    def call(self, activations):
#         activations = tf.expand_dims(activations, axis=1)
#         activations = tf.transpose(activations, [0, 2, 3, 1])
#         x = tf.nn.conv2d(activations, self.sparse_filters, strides=(1, 4), padding='VALID')
#         x = tf.nn.relu(x)
#         x = tf.stop_gradient(self.sparse_model([activations, tf.stop_gradient(tf.expand_dims(self.recon_model.trainable_weights[0], axis=0))]))

        x = self.conv_1(activations)
        x = self.conv_2(x)
#         x = self.dropout(x, self.do_dropout)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
#         x = self.conv_6(x)
        x = self.flatten(x)
#         x = self.ff_1(x)
#         x = self.dropout(x)
        x = self.ff_2(x)
        x = self.ff_dropout(x, self.do_dropout)
        x = self.ff_3(x)
#         x = self.dropout(x)
        pred_1 = self.ff_final_1(x)
        pred_2 = self.ff_final_2(x)

        return pred_1, pred_2
    
class ONSDPositionalModel(keras.Model):
    def __init__(self):
        super(ONSDPositionalModel, self).__init__()
        
        inputs = keras.Input(shape=(1, img_size[0], img_size[1], 1))
        
        filter_inputs = keras.Input(shape=(1, kernel_height, kernel_width, 1, num_kernels), dtype='float32')

        output = SparseCode(batch_size=batch_size, image_height=img_size[0], image_width=img_size[1], clip_depth=1, in_channels=1, out_channels=num_kernels, kernel_height=kernel_height, kernel_width=kernel_width, kernel_depth=1, stride=1, lam=0.05, activation_lr=1e-2, max_activation_iter=200, run_2d=False)(inputs, filter_inputs)

        self.sparse_model = keras.Model(inputs=(inputs, filter_inputs), outputs=output)
        self.recon_model = keras.models.load_model(sparse_checkpoint)
        
        self.conv_layer = ONSDPositionalConv()
        
    def train_step(self, data):
        x, y1, y2 = data
        
        activations = tf.stop_gradient(self.sparse_model([x, tf.stop_gradient(tf.expand_dims(self.recon_model.trainable_weights[0], axis=0))]))

        with tf.GradientTape() as tape:
            y1_pred, y2_pred = self.conv_layer(activations)
            
            loss = keras.losses.mean_squared_error(y1, y1_pred) + keras.losses.mean_squared_error(y2, y2_pred)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}