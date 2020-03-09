from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from simpletransformers.question_answering import QuestionAnsweringModel
import json

CHCKPT_DIR = "data/checkpoints"
BEST_CHCKPT_DIR = "data/checkpoints/best"
TRAIN_FILE = "data/squad/train-v2.0.json"
EVAL_FILE = "data/squad/dev-v2.0.json"
EPOCHS = 30


def load_data(train_file, eval_file):
    with open(train_file, 'r') as file:
        train_data = json.load(file)

    with open(eval_file, 'r') as file:
        eval_data = json.load(file)

    train_data = [item for topic in train_data['data'] for item in topic['paragraphs']]
    eval_data = [item for topic in eval_data['data'] for item in topic['paragraphs']]

    return train_data, eval_data


train_d, eval_d = load_data(TRAIN_FILE, EVAL_FILE)

model = QuestionAnsweringModel('distilbert', 'distilbert-base-uncased-distilled-squad', use_cuda=False,
                               args={'reprocess_input_data': True, 'overwrite_output_dir': True,
                                     'evaluate_during_training': True, 'num_train_epochs': EPOCHS,
                                     'output_dir': CHCKPT_DIR, 'best_model_dir': BEST_CHCKPT_DIR})
model.train_model(train_d, eval_data=eval_d)

