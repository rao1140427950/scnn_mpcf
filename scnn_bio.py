import tensorflow as tf
import os
from keras.layers import Input, Conv2D, BatchNormalization, Activation
from keras.regularizers import l2
from keras import Model
from keras.metrics import CategoricalAccuracy
from keras.losses import CategoricalCrossentropy
from utils.dataset import BioDataset

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

def _conv_block(inputs, filters, kernel_size=3):
    x = _conv_layer(inputs, filters, kernel_size, 1)
    # x = _conv_layer(x, filters, 3, 1)
    x = _conv_layer(x, filters * 2, kernel_size, 2)
    return x

def build_model(
        input_shape=(256, 256, 9),
        regularizer=l2(0.0005),
        num_classes=5,
):

    x = in_layer = Input(shape=input_shape, dtype=tf.float32)

    x = _conv_block(x, 16)  # 128
    x = _conv_block(x, 16)  # 64
    x = _conv_block(x, 16)  # 32
    x = _conv_block(x, 16)  # 16
    x = _conv_block(x, 16)  # 8
    x = _conv_block(x, 16)  # 4
    x = _conv_layer(x, 16, 2, 1, 'valid') # 3
    x = _conv_layer(x, 16, 2, 1, 'valid') # 2

    out_layer = Conv2D(num_classes, kernel_size=2, padding='valid', activation='softmax', kernel_regularizer=regularizer)(x)

    network = Model(inputs=in_layer, outputs=out_layer)
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, weight_decay=0.0005),
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
    batch_size = 32
    epoch = 90
    model_name = 'scnn-bio'
    work_dir = './checkpoints'

    log_dir = work_dir + '/' + model_name
    output_model_file = work_dir + '/' + model_name + '.h5'
    weight_file = work_dir + '/' + model_name + '_weights.h5'
    checkpoint_path = work_dir + '/checkpoint-' + model_name + '.h5'

    model = build_model()
    # tf.keras.utils.plot_model(model, to_file='runtime_data/simple_dnn.png', show_shapes=True)

    if os.path.exists(weight_file):
        model.load_weights(weight_file)
        print('Found weights file: {}, load weights.'.format(weight_file))
    else:
        print('No weights file found. Skip loading weights.')

    dataset = BioDataset(dataset_path='./runtime_data/thyroid_dataset_splitted.mat', batch_size=batch_size, num_classes=5)
    train_samples, test_samples = dataset.read_dataset_from_matfile()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        verbose=1,
        save_weights_only=False,
        save_best_only=False,
        save_freq='epoch',
    )

    def scheduler(ep, lr):
        schedule = [20, 40, 60, 80]
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


