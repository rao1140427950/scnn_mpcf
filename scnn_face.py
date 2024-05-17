import tensorflow as tf
import os

import tqdm
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose
from keras.regularizers import l2
from keras import Model
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from utils.dataset import SimpleDataset


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)


def _conv_layer(inputs, filters, kernel_size, strides=1, padding='same', activation='relu', bn=True, regularizer=None):
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_regularizer=regularizer,
        )(inputs)
        if bn:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        return x

def build_model(
        conv_units=None,
        kernel_sizes=None,
        input_shape=(400, 532, 9),
        regularizer=l2(0.001),
        num_classes=2,
):
    if conv_units is None:
        conv_units = [16, 16, 16, 16, 16]
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 1, 3, 1]

    in_layer = Input(shape=input_shape, dtype=tf.float32)
    x = _conv_layer(in_layer, 8, 3, regularizer=regularizer)
    x = _conv_layer(x, 8, 3, strides=2, regularizer=regularizer, padding='same')

    x = _conv_layer(x, 16, 3, regularizer=regularizer)
    skip = x = _conv_layer(x, 16, 3, strides=2, regularizer=regularizer, padding='same')

    for unit, size  in zip(conv_units[1:], kernel_sizes[1:]):
        x = _conv_layer(x, unit, size, regularizer=regularizer)
    skip = x = Add()([skip, x])

    for unit, size  in zip(conv_units[1:], kernel_sizes[1:]):
        x = _conv_layer(x, unit, size, regularizer=regularizer)
    x = Add()([skip, x])

    x = Conv2DTranspose(16, 3, 2, kernel_regularizer=regularizer)(x)
    x = _conv_layer(x, 16, 3, regularizer=regularizer)

    x = Conv2DTranspose(8, 3, 2, padding='same', kernel_regularizer=regularizer)(x)
    x = _conv_layer(x, 8, 3, regularizer=regularizer)

    out_layer = Conv2D(num_classes, 3, padding='same', activation='softmax', kernel_regularizer=regularizer)(x)

    network = Model(inputs=in_layer, outputs=out_layer)
    network.compile(
        optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.),
        weighted_metrics=[CategoricalAccuracy()],
        loss=CategoricalCrossentropy(),
    )
    return network


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    start_epoch = 0
    batch_size = 16
    epoch = 500
    model_name = 'scnn-face'
    work_dir = './checkpoints'

    log_dir = work_dir + '/' + model_name
    output_model_file = work_dir + '/' + model_name + '.h5'
    weight_file = work_dir + '/' + model_name + '_weights.h5'
    checkpoint_path = work_dir + '/checkpoint-' + model_name + '.h5'

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_model(num_classes=2)
    # tf.keras.utils.plot_model(model, to_file='runtime_data/simple_dnn.png', show_shapes=True)

    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        print('Found weights file: {}, load weights.'.format(weight_file))
    else:
        print('No weights file found. Skip loading weights.')

    dataset = SimpleDataset(dataset_path='./runtime_data/face_led_dataset.mat', batch_size=batch_size, num_classes=2)
    train_samples, test_samples = dataset.read_dataset_from_matfile(validation_split=0.2)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=False,
        save_freq='epoch'
    )

    def scheduler(ep, lr):
        schedule = [100, 200, 300, 400]
        if ep in schedule:
            return lr * 0.1
        else:
            return lr
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False, write_images=False)

    model.fit(
        x=train_samples,
        validation_data=test_samples,
        epochs=epoch,
        callbacks=[checkpoint, lr_scheduler, tensorboard],
        initial_epoch=start_epoch,
        shuffle=False,
        verbose=1
    )

    model.save_weights(weight_file)
    model.save(output_model_file, include_optimizer=False)


