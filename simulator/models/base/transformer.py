import tensorflow as tf
from keras import layers, activations


class FeedForward(layers.Layer):
    def __init__(self, in_dim, expansion=4, dropout=0.1, activation=activations.relu, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(in_dim * expansion)
        self.act = activation
        self.dense2 = layers.Dense(in_dim)
        self.dropout = layers.Dropout(dropout)
        self.add = layers.Add()
        self.layer_norm = layers.LayerNormalization()

    def call(self, inputs, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.act(x)
        x = self.dense2(x)
        x = self.dropout(x)
        x = self.add([x, inputs])
        x = self.layer_norm(x)
        return x


class BaseAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = layers.MultiHeadAttention(**kwargs)
        self.layernorm = layers.LayerNormalization()
        self.add = layers.Add()
        self.last_attn_scores = None


class CrossAttention(BaseAttention):
    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class GlobalSelfAttention(BaseAttention):
    def call(self, x, *args, **kwargs):
        attn_output, attn_scores = self.mha(
            query=x,
            value=x,
            key=x,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class CausalSelfAttention(BaseAttention):
    def call(self, x, *args, **kwargs):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask=True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class EncoderLayer(layers.Layer):
    """
    inputs: (bsize, seq_length, ffdim)
    outputs: (bsize, seq_length, ffdim)
    """

    def __init__(self, num_heads, emb_size, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = GlobalSelfAttention(num_heads=num_heads, key_dim=emb_size, dropout=dropout)
        self.ffn = FeedForward(emb_size, dropout=dropout)
        self.last_attn_scores = None

    def call(self, inputs, *args, **kwargs):
        x = self.self_attention(inputs)
        x = self.ffn(x)
        self.last_attn_scores = self.self_attention.last_attn_scores
        return x


class DecoderLayer(layers.Layer):
    """
    inputs: x: (bsize, seq1, ffdim)  context: (bsize, seq2, ffdim)
    outputs: (bsize, seq1, ffdim)
    """

    def __init__(self, num_heads, emb_size, dropout=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.causal_self_attention = CausalSelfAttention(num_heads=num_heads, key_dim=emb_size, dropout=dropout)
        self.cross_attention = CrossAttention(num_heads=num_heads, key_dim=emb_size, dropout=dropout)
        self.ffn = FeedForward(emb_size, dropout=dropout)
        self.last_attn_scores = None

    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        x = self.causal_self_attention(x)
        x = self.cross_attention([x, context])
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
        return x


class Encoder(layers.Layer):

    def __init__(self, num_layers, num_heads, emb_size, dropout=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.enc_layers = [EncoderLayer(num_heads, emb_size, dropout) for _ in range(num_layers)]

    def call(self, inputs, *args, **kwargs):
        x = inputs
        for layer in self.enc_layers:
            x = layer(x)
        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, num_heads, emb_size, dropout=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dec_layers = [DecoderLayer(num_heads, emb_size, dropout) for _ in range(num_layers)]
        self.last_attn_scores = None

    def call(self, inputs, *args, **kwargs):
        x, context = inputs
        for layer in self.dec_layers:
            x = layer([x, context])
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x


if __name__ == '__main__':
    encoder_inputs = tf.random.normal((4, 256, 256))
    decoder_inputs = tf.random.normal((4, 202, 256))

    encoder = Encoder(4, 8, 256)
    decoder = Decoder(4, 8, 256)

    ctt = encoder(encoder_inputs)
    outputs = decoder([decoder_inputs, ctt])
