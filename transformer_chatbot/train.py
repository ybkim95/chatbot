import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from urllib.request import urlopen
import time
import tensorflow_datasets as tfds
import tensorflow as tf
from transformer import *
import os

train_data = pd.read_csv("ChatBotData.csv")

questions = []
for sentence in train_data['Q']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence) # 정규표현식: "?" "." "!" ","과 같은 구두점을 기준으로 공백
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence) # 정규표현식: "?" "." "!" ","과 같은 구두점을 기준으로 공백
    sentence = sentence.strip()
    answers.append(sentence)

tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(questions+answers, target_vocab_size=2**13)

START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size+1]

VOCAB_SIZE = tokenizer.vocab_size + 2

sample_string = questions[20]

tokenized_string = tokenizer.encode(sample_string)
original_string = tokenizer.decode(tokenized_string)

MAX_LENGTH = 40

# tokenize / integer encoding / add START, END token / padding
def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [],[]

    # add START, END token
    for (s1,s2) in zip(inputs,outputs):
        s1 = START_TOKEN + tokenizer.encode(s1) + END_TOKEN
        s2 = START_TOKEN + tokenizer.encode(s2) + END_TOKEN

        tokenized_inputs.append(s1)
        tokenized_outputs.append(s2)
    
    # padding
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_inputs, maxlen=MAX_LENGTH, padding="post")
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(tokenized_outputs, maxlen=MAX_LENGTH, padding="post")

    return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)

# 4. Encoder, Decoder Input and make Label
BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1] # remove END token
    },
    {
        'outputs': answers[:, 1:] # remove START token
    },
))

dataset = dataset.cache() # store dataset in cache, file
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE) # get data in prior for the operation

# 5. Transformer
tf.keras.backend.clear_session()

def get_model():
    # hyperparameter
    D_MODEL = 256
    NUM_LAYERS = 2
    NUM_HEADS = 8
    DFF = 512
    DROPOUT = 0.1

    model = transformer(
        vocab_size = VOCAB_SIZE,
        num_layers = NUM_LAYERS,
        dff = DFF,
        d_model = D_MODEL,
        num_heads = NUM_HEADS,
        dropout = DROPOUT
    )

    learning_rate = CustomSchedule(D_MODEL)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def accuracy(y_true, y_pred):
        y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH-1))
        return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

    model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

    return model

if __name__ == "__main__":

    EPOCHS = 50
    checkpoint_path = "./model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1)

    new_model = get_model()

    history = new_model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])