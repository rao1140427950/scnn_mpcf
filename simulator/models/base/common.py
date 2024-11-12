import numpy as np
import tensorflow as tf
from keras import layers


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth   # (1, depth)

    angle_rates = 1 / (10000 ** depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(layers.Layer):
    def __init__(self, length, depth, trainable_embedding=False, rescale=False):
        super(PositionalEmbedding, self).__init__()
        self.length = length
        self.depth = depth
        self.rescale = rescale
        if trainable_embedding:
            self.position_emb = self.add_weight(shape=(1, length, depth), dtype=tf.float32, name='position_embd',
                                                initializer=tf.keras.initializers.random_normal(), trainable=True)
        else:
            self.position_emb = positional_encoding(length, depth)[tf.newaxis, ...]

    def call(self, inputs, *args, **kwargs):
        if self.rescale:
            inputs *= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        length = tf.shape(inputs)[1]
        return inputs + self.position_emb[:, :length, :]


class PatchEmbedding(layers.Layer):
    def __init__(self, image_size=256, patch_size=16, emb_size=256, cls_token=False, **kwargs):
        super().__init__(**kwargs)
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.emb_size = emb_size

        self.proj = layers.Conv2D(emb_size, kernel_size=patch_size, strides=patch_size)
        self.reshape = layers.Reshape((-1, emb_size))
        if cls_token:
            self.cls_token = self.add_weight(shape=(1, 1, emb_size), dtype=tf.float32, name='class_token',
                                             initializer=tf.keras.initializers.random_normal(), trainable=True)
        else:
            self.cls_token = None

    def build(self, input_shape):
        b, h, w, c = input_shape
        assert h == self.image_size
        assert w == self.image_size
        super(PatchEmbedding, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        x = self.proj(inputs)  # (b, h // p, w // p, emb)
        x = self.reshape(x)  # (b, nh * nw, emb)

        if self.cls_token is not None:
            bsize = tf.shape(inputs)[0]
            cls_token = tf.tile(self.cls_token, (bsize, 1, 1))
            x = tf.concat([cls_token, x], axis=1)  # (b, nh * nw + 1, emb)

        return x


class FeatureEmbedding(layers.Layer):
    """
    input: (bsize, seq_length, 2). Each element is normalized to [-1, 1]
    """
    def __init__(self, emb_size=256, **kwargs):
        super(FeatureEmbedding, self).__init__(**kwargs)
        self.proj = layers.Dense(emb_size)
        self.emb_size = emb_size

    def call(self, inputs, *args, **kwargs):
        x = self.proj(inputs)
        return x


class EncoderPreprocessing(layers.Layer):

    def __init__(
            self,
            image_size=256,
            patch_size=16,
            emb_size=256,
            period_token=True,
            trainable_positional_embedding=False,
            rescale=False,
    ):
        super(EncoderPreprocessing, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, emb_size)
        npos = (image_size // patch_size) ** 2
        self.period_token = period_token
        if period_token:
            npos += 1
            self.period_proj = layers.Dense(emb_size)
        self.pos_embedding = PositionalEmbedding(npos, emb_size, trainable_positional_embedding, rescale)
        self.npos = npos

    def call(self, inputs, *args, **kwargs):
        if self.period_token:
            images, periods = inputs  # (bsize, imgsize, imgsize, 1), (bsize, 1)
            periods = self.period_proj(periods)  # (bsize, emb)
            images = self.patch_embedding(images)  # (b, nh * nw, emb)
            context = tf.concat([images, periods[:, tf.newaxis, :]], axis=1)
        else:
            context = self.patch_embedding(inputs)

        return self.pos_embedding(context)


class DecoderPreprocessing(layers.Layer):

    def __init__(
            self,
            input_depth=2,
            emb_size=256,
            max_length=201,
            trainable_positional_embedding=False,
            rescale=False,
            start_token=True,
            trainable_start_token=True,
            apply_start_token=True,
    ):
        super(DecoderPreprocessing, self).__init__()
        self.vector_embedding = FeatureEmbedding(emb_size)
        self.pos_embedding = PositionalEmbedding(max_length, emb_size, trainable_positional_embedding, rescale)
        if start_token:
            if trainable_start_token:
                self.start_token = self.add_weight(shape=(1, 1, input_depth), dtype=tf.float32, name='start_token',
                                                initializer=tf.keras.initializers.zeros(), trainable=True)
            else:
                self.start_token = tf.zeros(shape=(1, 1, input_depth))
        else:
            self.start_token =None
        self.apply_start_token = apply_start_token

    def call(self, inputs, *args, **kwargs):
        if self.apply_start_token:
            bsize = tf.shape(inputs)[0]
            start_token = tf.tile(self.start_token, (bsize, 1, 1))
            x = tf.concat([start_token, inputs], axis=1)
        else:
            x = inputs

        x = self.vector_embedding(x)
        return self.pos_embedding(x)




