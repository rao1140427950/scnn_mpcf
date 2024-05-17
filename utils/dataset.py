import os.path
import numpy as np
import tensorflow as tf
from scipy.io import loadmat, savemat


class SimpleDataset:

    def __init__(self, data_dir=None, dataset_path=None, num_classes=2, batch_size=10):
        self._data_dir = data_dir
        self._nclasses = num_classes
        self._batch_size = batch_size
        self._dataset_path = dataset_path

        self._shuffle_buffer_size = 200
        self._prefetch_buffer_size = 16

    def _read_data(self):
        xdata = []
        ydata = []
        wdata = []
        files = os.listdir(self._data_dir)
        for file in files:
            if file.split('.')[-1] == 'mat':

                label_file = loadmat(os.path.join(self._data_dir, file))
                labels = label_file['labels']
                image_cube = label_file['image']

                w, h, c = np.shape(image_cube)
                image_cube = np.reshape(image_cube, (1, w, h, c))
                labels = np.reshape(labels, (1, w, h, 1))
                mask = np.zeros_like(labels)
                mask[labels >= 0] = 1
                labels[labels < 0] = 0

                xdata.append(image_cube)
                ydata.append(labels)
                wdata.append(mask)

        xdata = np.concatenate(xdata, axis=0)
        ydata = np.concatenate(ydata, axis=0)
        wdata = np.concatenate(wdata, axis=0)

        data_all = np.concatenate([ydata, wdata, xdata], axis=-1)
        np.random.shuffle(data_all)

        xdata = data_all[..., 2:]
        ydata = data_all[..., :1]
        wdata = data_all[..., 1:2]

        return xdata, ydata, wdata

    def write_matfile_from_labels(self):
        xdata, ydata, wdata = self._read_data()
        savemat(self._dataset_path, {'xdata': xdata, 'ydata': ydata, 'wdata': wdata}, do_compression=True)

    def _process_raw_dataset(self, raw_dataset, shuffle=True):

        def _process_single_sample(x, y, w, method='norm'):

            y = tf.cast(y, tf.int32)
            x = tf.cast(x, tf.float32)
            w = tf.cast(w, tf.float32)
            if method == 'norm':
                x = (x / 512.) - 1.
            elif method == 'standardize':
                std = tf.math.reduce_std(x, axis=-1, keepdims=True)
                mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
                x = x - mean
                x = tf.math.divide_no_nan(x, std)
            else:
                raise ValueError('`method` should be \'norm\' or \'standardize\' but get \'{}\''.format(method))

            y = tf.one_hot(tf.squeeze(y), depth=self._nclasses, dtype=tf.float32)

            return x, y, w

        if shuffle:
            shuffle_dataset = raw_dataset.shuffle(self._shuffle_buffer_size)
        else:
            shuffle_dataset = raw_dataset
        parsed_dataset = shuffle_dataset.map(_process_single_sample, num_parallel_calls=4)
        return parsed_dataset.batch(self._batch_size).prefetch(self._prefetch_buffer_size)

    def read_dataset_from_matfile(self, validation_split=None):
        matfile = loadmat(self._dataset_path)
        xdata = matfile['xdata']
        ydata = matfile['ydata']
        wdata = matfile['wdata']

        if validation_split is not None:
            n = xdata.shape[0]
            split = round(n * validation_split)
            val_x = xdata[:split, ...]
            train_x = xdata[split:, ...]
            val_y = ydata[:split, ...]
            train_y = ydata[split:, ...]
            val_w = wdata[:split, ...]
            train_w = wdata[split:, ...]
            train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y, train_w))
            train_dataset = self._process_raw_dataset(train_dataset, shuffle=True)
            val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y, val_w))
            val_dataset = self._process_raw_dataset(val_dataset, shuffle=False)
            return train_dataset, val_dataset
        else:
            raw_dataset = tf.data.Dataset.from_tensor_slices((xdata, ydata, wdata))
            return self._process_raw_dataset(raw_dataset), None


class BioDataset:

    def __init__(self, data_dir=None, dataset_path=None, num_classes=5, batch_size=10, image_size=256, repetition=10):
        self._data_dir = data_dir
        self._nclasses = num_classes
        self._batch_size = batch_size
        self._dataset_path = dataset_path
        self.image_size = image_size
        self.repetition = repetition

        self._shuffle_buffer_size = 200
        self._prefetch_buffer_size = 16

    def _read_data(self, validation_split=0.2):
        txdata = []
        tydata = []
        vxdata = []
        vydata = []
        dirs = os.listdir(self._data_dir)

        h, w, c = 0, 0, 0
        for n, dir_ in enumerate(dirs):
            dir_path = os.path.join(self._data_dir, dir_)
            files = os.listdir(dir_path)
            files.sort()
            split = round(len(files) * (1 - validation_split))

            for k, file in enumerate(files):
                if file.split('.')[-1] == 'mat':

                    label_file = loadmat(os.path.join(dir_path, file))
                    image_cube = label_file['image']
                    h, w, c = image_cube.shape
                    image_cube = np.reshape(image_cube, (1, -1))

                    if k < split:
                        txdata.append(image_cube)
                        tydata.append(n)
                    else:
                        vxdata.append(image_cube)
                        vydata.append(n)

        txdata = np.concatenate(txdata, axis=0)
        tydata = np.array(tydata)
        tydata = np.reshape(tydata, (-1, 1))

        data_all = np.concatenate([tydata, txdata], axis=-1)
        np.random.shuffle(data_all)
        txdata = data_all[..., 1:]
        tydata = data_all[..., :1]
        txdata = np.reshape(txdata, (-1, h, w, c))

        vxdata = np.concatenate(vxdata, axis=0)
        vydata = np.array(vydata)
        vydata = np.reshape(vydata, (-1, 1))

        data_all = np.concatenate([vydata, vxdata], axis=-1)
        np.random.shuffle(data_all)

        vxdata = data_all[..., 1:]
        vydata = data_all[..., :1]
        vxdata = np.reshape(vxdata, (-1, h, w, c))

        return (txdata, tydata), (vxdata, vydata)

    def write_matfile_from_labels(self, validation_split=0.2):
        (txdata, tydata), (vxdata, vydata) = self._read_data(validation_split)
        savemat(self._dataset_path, {'train_xdata': txdata, 'train_ydata': tydata,
                                     'val_xdata': vxdata, 'val_ydata': vydata,}, do_compression=True)

    def _process_raw_dataset(self, raw_dataset, shuffle=True):

        def _process_single_sample(x, y, method='norm'):

            y = tf.cast(y, tf.int32)
            x = tf.cast(x, tf.float32)
            if method == 'norm':
                x = (x / 512.) - 1.
            elif method == 'standardize':
                std = tf.math.reduce_std(x, axis=-1, keepdims=True)
                mean = tf.math.reduce_mean(x, axis=-1, keepdims=True)
                x = x - mean
                x = tf.math.divide_no_nan(x, std)
            else:
                raise ValueError('`method` should be \'norm\' or \'standardize\' but get \'{}\''.format(method))


            x = tf.image.random_flip_left_right(x)
            x = tf.image.random_flip_up_down(x)
            x = tf.image.random_crop(x, (self.image_size, self.image_size, 9))
            y = tf.one_hot(tf.squeeze(y), depth=self._nclasses, dtype=tf.float32)
            y = tf.reshape(y, (1, 1, -1))

            return x, y

        if shuffle:
            shuffle_dataset = raw_dataset.shuffle(self._shuffle_buffer_size)
        else:
            shuffle_dataset = raw_dataset
        parsed_dataset = shuffle_dataset.map(_process_single_sample, num_parallel_calls=4)
        return parsed_dataset.batch(self._batch_size).prefetch(self._prefetch_buffer_size)

    def read_dataset_from_matfile(self):
        matfile = loadmat(self._dataset_path)
        train_x = matfile['train_xdata']
        train_y = matfile['train_ydata']
        val_x = matfile['val_xdata']
        val_y = matfile['val_ydata']

        train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).repeat(self.repetition)
        train_dataset = self._process_raw_dataset(train_dataset, shuffle=True)
        val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y)).repeat(self.repetition)
        val_dataset = self._process_raw_dataset(val_dataset, shuffle=False)
        return train_dataset, val_dataset




if __name__ == '__main__':
    dataset = SimpleDataset(
        data_dir='../runtime_data/cache/face_led',
        dataset_path='../runtime_data/face_led_dataset.mat',
    )
    dataset.write_matfile_from_labels()

