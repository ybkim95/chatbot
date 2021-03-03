from transformer import *
from train import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from urllib.request import urlopen
import time
import tensorflow_datasets as tfds
import tensorflow as tf
import os

def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(sentence, model):
    checkpoint_path = "./model/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_path)

    sentence = preprocess_sentence(sentence)
    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis = 0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break
    
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)

def predict(sentence, model):

    prediction = evaluate(sentence, model)

    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    # print("입력: {}".format(sentence))
    print("챗봇 >>> {}".format(predicted_sentence))

    return predicted_sentence

if __name__ == "__main__":

    model = get_model()

    print()
    print("본 프로그램을 종료하고 싶을 경우 [quit, q, exit, esc] 중 하나를 입력해주세요.")

    while True:
        sentence = input("입력 >>> ")
        
        if sentence in ["quit", "q", "exit", "esc"]:
            break
        
        predict(sentence, model)