import os
import numpy as np
import tensorflow as tf


PI = 3.14159

def prepare_inputs(images, periods, img_size=256, min_p=300., max_p=700.):
    if images.ndim == 2:
        images = images[np.newaxis, :, :, np.newaxis]
    elif images.ndim == 3:
        images = images[..., np.newaxis]
    images = tf.image.resize(images, (img_size, img_size))
    periods = np.reshape(periods, (-1, 1))
    images = images * 2. - 1.
    periods = (periods - min_p) / (max_p - min_p) * 2. - 1.
    # periods = np.tile(periods, (1, 2))
    return tf.cast(images, tf.float32), tf.cast(periods, tf.float32)


def process_outputs(tpower, rescale=5., do_ifft=False):
    tpower /= rescale
    real_part = tpower[..., 0]
    imag_part = tpower[..., 1]
    tpower = real_part + 1.j * imag_part
    if do_ifft:
        tpower = np.fft.ifft(tpower)
    return np.squeeze(np.abs(tpower)), np.squeeze(np.angle(tpower) % (2 * np.pi))


def config(gpu_ids):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.minimum(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2), 1e-3)

  def get_config(self):
      return {'d_model': self.d_model, 'warmup_steps': self.warmup_steps}


class ExportWrapper(tf.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
    def __call__(self, shapes, periods):
        return self.model([shapes, periods])