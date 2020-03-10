from __future__ import absolute_import, division, print_function, unicode_literals

import json
import os

import wget

# Setup Parameters
BATCH_SIZE = 32
BUFFER_SIZE = 20000
MAX_SAMPLES = 50000
MAX_LENGTH = 40

# Hyper-Parameters
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT = 0.1
EPOCHS = 30

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


# Slightly error prone but should be ok
print("Checking if dataset exists...")
if os.path.isfile(TRAIN_FILE):
    print("Dataset exists")
else:
    print("Dataset does not exist")
    download_dataset()

train_data = load_data(TRAIN_FILE)
eval_data = load_data(DEV_FILE)


