import tensorflow as tf
from simulator.models.base.common import EncoderPreprocessing
from simulator.models.base.transformer import Encoder
from keras import layers


class EncoderOnly(tf.keras.Model):

    def __init__(
            self,
            num_layers=8,
            d_model=256,
            num_heads=8,
            image_size=256,
            patch_size=16,
            dropout=0.1,
            out_ffdim=2,
            trainable_position_emb=False,
            rescale=False,
            period_token=True,
    ):
        super(EncoderOnly, self).__init__()
        self.encoder_preprocesssing = EncoderPreprocessing(
            image_size=image_size,
            patch_size=patch_size,
            emb_size=d_model,
            period_token=period_token,
            trainable_positional_embedding=trainable_position_emb,
            rescale=rescale,
        )
        self.encoder = Encoder(num_layers, num_heads, d_model, dropout)
        self.final_layer = layers.Dense(out_ffdim)
        self.seq_length = self.encoder_preprocesssing.npos

    def call(self, inputs, training=None, mask=None):
        x = self.encoder_preprocesssing(inputs)
        x = self.encoder(x)
        x = self.final_layer(x)
        return x


if __name__ == '__main__':
    img = tf.random.normal((4, 256, 256, 1))
    p = tf.random.normal((4, 1))
    encoder = EncoderOnly()
    y = encoder([img, p])
    pass