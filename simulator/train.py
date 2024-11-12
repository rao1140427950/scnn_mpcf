import numpy as np
from utils.common import config, PI
from utils.dataset import generate_dataset_from_tfrecords
from models.EncoderOnly import EncoderOnly as Model
from utils.common import prepare_inputs, process_outputs, ExportWrapper
import tensorflow as tf
from keras.metrics import MeanSquaredError, CosineSimilarity


def build_datasets(batch_size, image_size=256):
    train_dataset = generate_dataset_from_tfrecords([
        './srcs/datasets/dataset_220nm_c4_type2.tfrecords',
        './srcs/datasets/dataset_220nm_c4_type1.tfrecords',
    ], batch_size=batch_size, img_size=image_size, shuffle=True, random_roll=True, out_seq_length=257,
        type_='encoder_only')
    val_dataset = generate_dataset_from_tfrecords([
        './srcs/datasets/dataset_220nm_c4_val.tfrecords',
    ], batch_size=batch_size, img_size=image_size, shuffle=False, random_roll=False, out_seq_length=257,
        type_='encoder_only')
    return train_dataset, val_dataset


def train():
    image_size = 256
    ffdim = 256
    num_layers = 4

    batch_size = 64 * 8
    epochs = 16
    model_name = 'encoderonly-4layer-soi'
    work_dir = './checkpoints/'
    log_dir = work_dir + model_name
    output_model_file = work_dir + model_name + '.h5'
    checkpoint_path = work_dir + 'checkpoint-' + model_name + '.h5'

    trainset, valset = build_datasets(batch_size, image_size)

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=False,
        write_images=False,
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='loss',
        save_weights_only=True,
        save_freq='epoch',
    )

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=0.001, decay_steps=2000 * epochs)
        optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9, weight_decay=2e-4)
        model = Model(num_layers=num_layers, d_model=ffdim, image_size=image_size, patch_size=16)
        img = tf.random.normal((4, 256, 256, 1))
        p = tf.random.normal((4, 1))
        model([img, p])
        model.compile(
            loss='mse',
            optimizer=optimizer,
            metrics=[MeanSquaredError(), CosineSimilarity(axis=-2)],
        )

    model.fit(
        trainset,
        epochs=epochs,
        validation_data=valset,
        callbacks=[tensorboard, checkpoint]
    )

    model.save_weights(output_model_file)


def export_model():
    image_size = 256
    ffdim = 256
    num_layers = 4

    model = Model(num_layers=num_layers, d_model=ffdim, image_size=image_size, patch_size=16)
    img = tf.random.normal((4, 256, 256, 1))
    p = tf.random.normal((4, 1))
    model([img, p])
    model.load_weights('./checkpoints/encoderonly-4layer-soi.h5')

    predictor = ExportWrapper(model)
    predictor(img, p)
    tf.saved_model.save(predictor, './exported_models/encoder_only_soi_4layer')


def predict():
    import scipy.io as sio
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list('grayblue', [(0.8588, 0.8588, 0.8588), (0.1804, 0.4588, 0.7137)])

    image_size = 256
    ffdim = 256
    num_layers = 4

    model = Model(num_layers=num_layers, d_model=ffdim, image_size=image_size, patch_size=16)
    img = tf.random.normal((4, 256, 256, 1))
    p = tf.random.normal((4, 1))
    model([img, p])
    model.load_weights('./checkpoints/encoderonly-4layer-soi.h5')

    mat = sio.loadmat('./srcs/data_00006535.mat')
    shape = mat['pattern']
    period = mat['period']
    tpower = np.squeeze(mat['T_power'])
    tphase = np.squeeze(mat['T_phase'])
    tphase %= (2 * PI)

    images, periods = prepare_inputs(shape, period)
    tpred = model([images, periods])
    tpred = tpred.numpy()

    tpower_pred, tphase_pred = process_outputs(tpred)

    sprange1 = np.linspace(400, 800, 201)
    sprange2 = np.linspace(400, 800, 257)

    plt.imshow(shape, cmap=cmap)
    plt.title(int(np.squeeze(period)))
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sprange2, tpower_pred, linewidth=2)
    plt.plot(sprange1, tpower, '--', linewidth=2)
    plt.title('T-Power')
    plt.subplot(1, 2, 2)
    plt.plot(sprange2, tphase_pred, linewidth=2)
    plt.plot(sprange1, tphase, '--', linewidth=2)
    plt.title('T-Phase')
    plt.show()


if __name__ == '__main__':
    config('0, 1, 2, 4, 5, 6, 8, 9')
    train()
    # predict()