import tensorflow as tf
import os
import scipy.io as sio
# from scipy.interpolate import interp1d
import numpy as np


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def generate_tfrecords_from_mats(mats_dir, save_path, label):
    files = os.listdir(mats_dir)
    writer = tf.io.TFRecordWriter(save_path)
    n = 0
    for file in files:
        mat = sio.loadmat(os.path.join(mats_dir, file))
        pattern = mat['pattern']
        period = mat['period'].reshape((1,))
        t_power = mat['T_power'].astype(np.float32).squeeze()
        t_phase = mat['T_phase'].astype(np.float32).squeeze()
        # xold = np.linspace(0, 1, 201)
        # xnew = np.linspace(0, 1, 257)
        # t_power = interp1d(xold, t_power)(xnew)
        # t_phase = interp1d(xold, t_phase)(xnew)
        feature = {'pattern': _bytes_feature(tf.image.encode_png(pattern[..., tf.newaxis])),
                   'period': _float_list_feature(period),
                   't_power': _float_list_feature(t_power),
                   't_phase': _float_list_feature(t_phase),
                   'label': _int64_feature(label),}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        n += 1
    writer.close()
    print('Found {:d} samples.'.format(n))


def get_base_dataset(data_path, batch_size=64, out_dim=201, img_size=256, shuffle=True, random_roll=True, cache=False):
    feature_description = {
        'pattern': tf.io.FixedLenFeature([], tf.string),
        'period': tf.io.FixedLenFeature((1,), tf.float32),
        't_power': tf.io.FixedLenFeature((out_dim,), tf.float32),
        't_phase': tf.io.FixedLenFeature((out_dim,), tf.float32),
    }

    def _parse_example_fcn(example_proto):
        example = tf.io.parse_single_example(example_proto, feature_description)
        pattern = example['pattern']
        pattern = tf.image.decode_png(pattern, channels=1)
        pattern = tf.image.resize(pattern, (img_size, img_size))

        if random_roll:
            shift = tf.random.uniform(shape=(2,), minval=0, maxval=img_size, dtype=tf.int32)
            pattern = tf.roll(pattern, shift, axis=(0, 1))

        example['pattern'] = pattern
        return example

    data = tf.data.TFRecordDataset(data_path, num_parallel_reads=len(data_path))
    data = data.map(_parse_example_fcn)
    if cache:
        data = data.cache()
    if shuffle:
        data = data.shuffle(2000, reshuffle_each_iteration=True)
    data = data.batch(batch_size)
    return data


def generate_dataset_from_tfrecords(data_path, batch_size=64, out_dim=201, img_size=256, shuffle=True, random_roll=True,
                                    type_=None, out_seq_length=201, do_fft=False):

    def _batch_map_fcn(example, min_p=300, max_p=800, rescale_tpower=True):

        pattern = example['pattern']
        pattern = tf.cast(pattern, tf.float32)
        pattern = pattern * 2. - 1.

        t_power = example['t_power']  # (bsize, 201)
        t_phase = example['t_phase']
        t_power = tf.cast(t_power, tf.float32)
        t_phase = tf.cast(t_phase, tf.float32)
        real_part = t_power * tf.math.cos(t_phase)
        imag_part = t_power * tf.math.sin(t_phase)

        if do_fft:
            sig = tf.cast(real_part, tf.complex64) + tf.cast(1.j, tf.complex64) * tf.cast(imag_part, tf.complex64)
            fft = tf.signal.fft(sig)
            real_part = tf.math.real(fft)
            imag_part = tf.math.imag(fft)

        t_power = tf.concat([real_part[..., tf.newaxis], imag_part[..., tf.newaxis]], axis=-1)  # (bsize, 201, 2)
        if out_seq_length != 201:
            t_power = tf.image.resize(t_power[..., tf.newaxis], (out_seq_length, 2), method='bicubic')
            t_power = tf.squeeze(t_power)
        if rescale_tpower:
            t_power *= 5.

        period = example['period']
        period = tf.cast((period - min_p) / (max_p - min_p), tf.float32) * 2. - 1.  # (bsize, 1)
        if type_ == 'encoder_only':
            return (pattern, period), t_power
        else:
            # period = tf.tile(period, (1, 2))  # (bsize, 2)
            tpower_input = t_power[:, :-1, :]  # (bsize, 201, 2)
            return (pattern, period, tpower_input), t_power

    data = get_base_dataset(data_path, batch_size, out_dim, img_size, shuffle, random_roll)
    data = data.map(_batch_map_fcn, num_parallel_calls=4)
    return data.prefetch(16)


if __name__ == '__main__':
    generate_tfrecords_from_mats(r'E:\raosj20\PycharmProjects\aigm\runtime_data\dataset_220nm_sparam_c4_type2',
                                 '../srcs/datasets/dataset_220nm_sparam_c4_type2_d201.tfrecords', label=0)


    # dataset = generate_dataset_from_tfrecords(['../srcs/datasets/dataset_220nm_c4_unified.tfrecords'],
    #                                           batch_size=32, shuffle=False, random_roll=False, type_='encoder_only', do_fft=True, out_seq_length=257)
    # dataset = generate_dataset_from_tfrecords([
    #     '../runtime_data/dataset_220nm_c4_type2.tfrecords',
    #     '../runtime_data/dataset_220nm_c4_type1.tfrecords',
    #     '../runtime_data/dataset_220nm_sparam_random_r2.tfrecords',
    #     '../runtime_data/dataset_220nm_sparam_random_r1.tfrecords'
    # ], batch_size=4).take(1)

    # nomalizer = tf.keras.layers.Normalization(axis=-1)
    # nomalizer.adapt(dataset)
    #
    # print(nomalizer.mean, nomalizer.variance)
    #
    # nomalizer = tf.keras.layers.Normalization(axis=None)
    # nomalizer.adapt(dataset)
    #
    # print(nomalizer.mean, nomalizer.variance)
    #
    # dataset = generate_dataset_from_tfrecords([
    #     '../srcs/datasets/dataset_220nm_c4_type2.tfrecords',
    #     '../srcs/datasets/dataset_220nm_c4_type1.tfrecords',
    #     '../srcs/datasets/dataset_220nm_sparam_random_r2.tfrecords',
    #     '../srcs/datasets/dataset_220nm_sparam_random_r1.tfrecords',
    # ], batch_size=32, shuffle=False, random_roll=False, type_='encoder_only', do_fft=True, out_seq_length=257)
    #
    # nomalizer = tf.keras.layers.Normalization(axis=-1)
    # nomalizer.adapt(dataset)
    #
    # print(nomalizer.mean, nomalizer.variance)
    #
    # nomalizer = tf.keras.layers.Normalization(axis=None)
    # nomalizer.adapt(dataset)
    #
    # print(nomalizer.mean, nomalizer.variance)

    # for (img, x), y in dataset:
    #     break
    #
    # img = img.numpy()
    # x = x.numpy()
    # y = y.numpy()
    #
    # print(img.shape)
    # print(x.shape)
    # print(y.shape)
    #
    # pass