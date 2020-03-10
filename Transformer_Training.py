import json
import os
import tensorflow as tf
import numpy as np
import unicodedata
import re

import wget

# Hyper-Parameters
EPOCHS = 30
MODEL_SIZE = 128

# File Locations
TRAIN_FILE = "data/squad/train-v2.0.json"
DEV_FILE = "data/squad/dev-v2.0.json"
CHECKPOINT_FILE = "data/models/qna_transformer.hdf5"


def download_dataset():
    print("Downloading SQuAD 2.0 Training and Dev Dataset")
    train = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    dev = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    wget.download(train, "data/squad/train-v2.0.json")
    wget.download(dev, "data/squad/dev-v2.0.json")


def load_data(file_dir):
    print("Opening " + file_dir)
    with open(file_dir, 'r') as file:
        data = json.load(file)

    data = [item for topic in data['data'] for item in topic['paragraphs']]

    return data


def unicode_to_ascii(sentence):
    return ''.join(
        c for c in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(sentence):
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r'([!.?])', r' \1', sentence)
    sentence = re.sub(r'[^a-zA-Z.!?]+', r' ', sentence)
    sentence = re.sub(r'\s+', r' ', sentence)
    return sentence


def preprocess_raw_data(raw_data):
    print("Pre-processing Data...")
    raw_data_ques = []
    raw_data_ans = []

    for feature in raw_data:
        context = feature['context']
        for qas in feature['qas']:
            raw_data_ques.append([context, qas['question']])
            if qas['answers']:
                for answers in qas['answers']:
                    raw_data_ans.append(answers['text'])
            else:
                raw_data_ans.append('No Answer')
    print("Finished")
    return raw_data_ques, raw_data_ans


def prepare_data(raw_data):
    print("Preparing Data...")
    raw_data_ques, raw_data_ans = preprocess_raw_data(raw_data)

    print("Normalizing raw data..")
    raw_data_ques = [[normalize_string(data[0]), normalize_string(data[1])] for data in raw_data_ques]
    print("Adding special tokens to answers")
    raw_data_ans_in = ['<start> ' + normalize_string(data) for data in raw_data_ans]
    raw_data_ans_out = [normalize_string(data) + ' <end>' for data in raw_data_ans]

    print("Creating Tokenizers..")
    ques_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    ans_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    print("Building Vocabulary..")
    ques_tokenizer.fit_on_texts(raw_data_ques)
    ans_tokenizer.fit_on_texts(raw_data_ans_in)
    ans_tokenizer.fit_on_texts(raw_data_ans_out)

    print("Tokenizing raw data")
    data_encoder = ques_tokenizer.texts_to_sequences(raw_data_ques)
    data_encoder = tf.keras.preprocessing.sequence.pad_sequences(data_encoder, padding='post')

    data_decoder_in = ans_tokenizer.texts_to_sequences(raw_data_ans_in)
    data_decoder_in = tf.keras.preprocessing.sequence.pad_sequences(data_decoder_in, padding='post')

    data_decoder_out = ans_tokenizer.texts_to_sequences(raw_data_ans_out)
    data_decoder_out = tf.keras.preprocessing.sequence.pad_sequences(data_decoder_out, padding='post')

    print("Done")
    return data_encoder, data_decoder_in, data_decoder_out


def positional_embedding(pos, model_size):
    POS_EMB = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            POS_EMB[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            POS_EMB[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return POS_EMB


if __name__ == '__main__':
    # Slightly error prone but should be ok
    print("Checking if dataset exists...")
    if os.path.isfile(TRAIN_FILE):
        print("Dataset exists")
    else:
        print("Dataset does not exist")
        download_dataset()

    raw_train_data = load_data(TRAIN_FILE)
    raw_eval_data = load_data(DEV_FILE)

    data_ques, data_ans_in, data_ans_out = prepare_data(raw_train_data)

    max_length = max(len(data_ques[0]), len(data_ans_in[0]))
    pos_encoding = []
    for i in range(max_length):
        pos_encoding.append(positional_embedding(i, MODEL_SIZE))

    pos_encoding = np.concatenate(pos_encoding, axis=0)
    pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((data_ques, data_ans_in, data_ans_out))
    dataset = dataset.shuffle(20).batch(5)
