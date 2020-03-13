import re
import tensorflow as tf
import os
import wget
import json
import unicodedata

TRAIN_FILE = "data/squad/train-v2.0.json"
DEV_FILE = "data/squad/dev-v2.0.json"


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


def build_raw_dataset(raw_data):
    print("Pre-processing Data...")
    raw_dataset_ques = []
    raw_dataset_ans = []

    for feature in raw_data:
        for qas in feature['qas']:
            raw_dataset_ques.append(qas['question'])
            if qas['answers']:
                for answers in qas['answers']:
                    raw_dataset_ans.append(answers['text'])
            else:
                raw_dataset_ans.append('No Answer')
    print("Finished")
    return raw_dataset_ques, raw_dataset_ans


def create_dataset(batch_size):
    # Slightly error prone but should be ok
    print("Checking if dataset exists...")
    if os.path.isfile(TRAIN_FILE):
        print("Dataset exists")
    else:
        print("Dataset does not exist")
        download_dataset()

    raw_train_data = load_data(TRAIN_FILE)

    print("Preparing Data...")
    raw_data_ques, raw_data_ans = build_raw_dataset(raw_train_data)

    print("Normalizing raw data..")
    raw_data_ques = [normalize_string(data) for data in raw_data_ques]
    print("Adding special tokens to answers")
    raw_data_ans_in = ['<start> ' + normalize_string(data) for data in raw_data_ans]
    raw_data_ans_out = [normalize_string(data) + ' <end>' for data in raw_data_ans]

    """## Tokenization"""

    print("Creating Tokenizers..")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

    print("Building Vocabulary..")
    tokenizer.fit_on_texts(raw_data_ques)
    tokenizer.fit_on_texts(raw_data_ans_in)
    tokenizer.fit_on_texts(raw_data_ans_out)

    print("Tokenizing raw data")
    data_ques = tokenizer.texts_to_sequences(raw_data_ques)
    data_ques = tf.keras.preprocessing.sequence.pad_sequences(data_ques, padding='post')

    data_ans_in = tokenizer.texts_to_sequences(raw_data_ans_in)
    data_ans_in = tf.keras.preprocessing.sequence.pad_sequences(data_ans_in, padding='post')

    data_ans_out = tokenizer.texts_to_sequences(raw_data_ans_out)
    data_ans_out = tf.keras.preprocessing.sequence.pad_sequences(data_ans_out, padding='post')

    """## TF Dataset Creation"""

    dataset = tf.data.Dataset.from_tensor_slices((data_ques, data_ans_in, data_ans_out))
    dataset = dataset.shuffle(len(data_ques)).batch(batch_size)

    """## Create the Positional Embedding"""

    max_length = max(len(data_ques[0]), len(data_ans_in[0]))
    vocab_size = len(tokenizer.word_index) + 1

    info = {
        'max_length': max_length,
        'vocab_size': vocab_size,
        'data_size': len(raw_data_ques),
        'tokenizer': tokenizer
    }

    return dataset, info
