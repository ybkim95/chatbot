# Date: 2021.02.20 SAT
# Author: Y.B.KIM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        return position * 1 / tf.pow(100000, (2*i//2) / tf.cast(d_model, tf.float32))

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = d_model)

        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        angle_rads = np.zeros(angle_rads.shape)
        angle_rads[:, 0::2] = sines
        angle_rads[:, 1::2] = cosines
        pos_encoding = tf.constant(angle_rads)
        pos_encoding = pos_encoding[tf.newaxis, ...]

        print(pos_encoding.shape)

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def scaled_dot_product_attention(query, key, value, mask): # mask: In order to exclude the similarity of the <PAD> token

    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # If mask is used
    if mask is not None:
        logits += (mask * -1e9)
    
    attention_weigths = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weigths, value)

    return output, attention_weigths


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name) # 그냥 super와 차이는 없으나 파생클래스와 self를 넣어 현재 클래스가 어떤 클래스인지 명확하게 하려는 뜻임 
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0 # Checks if it is not the case

        self.depth = d_model // self.num_heads

        # Dense Layer for the W_Q, W_K, W_V
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        # Dense Layer for the W_0
        self.dense = tf.keras.layers.Dense(units=d_model)

    # Split Query, Key, Value into # of num_heads
    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # 1. Pass through Dense Layer (W_Q, W_K, W_V)
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. Split Head
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. Scaled Dot Product Attention
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])

        # 4. Concat Heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # 5. Pass through Dense Layer (W_0)
        outputs = self.dense(concat_attention)

        return outputs

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]

def encoder_layer(dff, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")

    padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({'query': inputs, 'key': inputs, 'value': inputs, 'mask': padding_mask})

    # Dropout + Residual Connection + Layer Normalization (1)
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)  
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs+attention) 

    # Position-Wise FFNN (2nd Sub-Layer)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Dropout + Residual Connection + Layer Normalization (2)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention+outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="encoder"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # padding mask
    padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")

    # Positional Encoding + Dropout
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32).numpy())
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # Stack Encoder for # of num_layer 
    for i in range(num_layers):
        outputs = encoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name="encoder_layer_{}".format(i))([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)

def decoder_layer(dff, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    # Look ahead mask (1st Sub-Layer)
    look_ahead_mask = tf.keras.Input(shape=(1,None,None), name="look_ahead_mask")

    # Padding Mask (2nd Sub-Layer)
    padding_mask = tf.keras.Input(shape=(1,1,None), name="padding_mask")

    # Multi-head Attention (1st Sub-Layer / Masked Self-Attention)
    attention1 = MultiHeadAttention(d_model, num_heads, name="attention1")(inputs={'query': inputs, 'key': inputs, 'value': inputs, 'mask': look_ahead_mask})

    # Residual Connection & Layer Normalization
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1+inputs)

    # Multi-head Attention (2nd Sub-Layer / Decoder-Encoder Attention)
    attention2 = MultiHeadAttention(d_model, num_heads, name="attention2")(inputs={'query': attention1, 'key': enc_outputs, 'value': enc_outputs, 'mask': padding_mask})

    # Dropout + Residual Coonection + Layer Normalization
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2+attention1)

    # Position-wise FFNN (3rd Sub-Layer)
    outputs = tf.keras.layers.Dense(units=dff, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)

    # Dropout + Residual Coonection + Layer Normalization
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs+attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

def decoder(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="decoder"):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")

    # Look ahead Mask + Padding Mask
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    # Positional Encoding + Dropout
    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32).numpy())
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    # Stack decoder for # of num_layers
    for i in range(num_layers):
        outputs = decoder_layer(dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout, name="decoder_layer_{}".format(i))(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    
    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def transformer(vocab_size, num_layers, dff, d_model, num_heads, dropout, name="transformer"):

    # Encoder Input
    inputs = tf.keras.Input(shape=(None,), name="inputs")

    # Decoder Input
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    # Encoder's Padding Mask
    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None), name="enc_padding_mask")(inputs)

    # Decoder's Look ahead Mask (1st Sub-Layer)
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1,1,None), name='look_ahead_mask')(dec_inputs)

    # Decoder's padding Mask (2nd Sub-Layer)
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1,1,None), name='dec_padding_mask')(inputs)

    # Encoder's output heads to Decoder
    enc_outputs = encoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[inputs, enc_padding_mask])

    # Decoder's output heads to Output Layer
    dec_outputs = decoder(vocab_size=vocab_size, num_layers=num_layers, dff=dff, d_model=d_model, num_heads=num_heads, dropout=dropout)(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    # Output Layer
    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

def loss_function(y_true, y_pred):
    MAX_LENGTH = 40
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
    
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32).numpy() # 매우 중요
        self.warmup_steps = warmup_steps
    
    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        }
        return config
    
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step) # reciprocal(x): 1/x
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)





if __name__ == '__main__':

    # sample_pos_encoding = PositionalEncoding(50, 128)

    # plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    # plt.xlabel('Depth')
    # plt.xlim((0,128))
    # plt.ylabel('Position')
    # plt.colorbar()
    # plt.show()

    # np.set_printoptions(suppress=True)
    # temp_k = tf.constant([[10,0,0],
    #                       [0,10,0],
    #                       [0,0,10],
    #                       [0,0,10]], dtype=tf.float32) # (4,3)
    # temp_v = tf.constant([[   1,0],
    #                       [  10,0],
    #                       [ 100,5],
    #                       [1000,6]], dtype=tf.float32) # (4,2)
    # temp_q = tf.constant([[0,10,0]], dtype=tf.float32) # (1,3)

    # temp_out, temp_attn = PositionalEncoding.scaled_dot_product_attention(temp_q, temp_k, temp_v, None)
    # print(temp_attn) # attention distribution
    # print(temp_out)  # attention value


    small_transformer = transformer(vocab_size = 9000, 
                                    num_layers = 4,
                                    dff = 512,
                                    d_model = 128,
                                    num_heads = 4,
                                    dropout = 0.3,
                                    name = "small_transformer")

    tf.keras.utils.plot_model(small_transformer, to_file='small_transformer.png', show_shapes=True)

    sample_learning_rate = CustomSchedule(d_model=128)

    plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    plt.ylabel("lr")
    plt.xlabel("train step")

    # Text(0.5, 0, 'train step')
    
