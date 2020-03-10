import json
import os
import tensorflow as tf
import numpy as np
import unicodedata
import re
import time

import wget

# Hyper-Parameters
EPOCHS = 30
MODEL_SIZE = 128
ATTENTION_HEADS = 2
NUM_LAYERS = 2
BATCH_SIZE = 5

# File Locations
TRAIN_FILE = "data/squad/train-v2.0.json"
DEV_FILE = "data/squad/dev-v2.0.json"
CHECKPOINT_FILE = "data/models/qna_transformer.hdf5"


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h

        # One Emedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)

        # n MHAs and Normalization Layers
        self.attention = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

        # N FFN
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]

        # N Normalization layers
        self.ffn_norm = [tf.keras.layers.LayerNormalization() for _ in range(num_layers)]

    def call(self, sequence):
        sub_in = []
        for i in range(sequence.shape[1]):
            # Compute the embedded vector
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))

            # Add positional encoding to the embedded vector
            sub_in.append(embed + pos_encoding[i, :])

        # Concatenate the result so that the shape is (batch_size, length, model_size)
        sub_in = tf.concat(sub_in, axis=1)

        # We will have num_layers of (Attention + FFN)
        for i in range(self.num_layers):
            sub_out = []

            # Iterate along the sequence length
            for j in range(sub_in.shape[1]):
                # Compute the context vector towards the whole sequence
                attention = self.attention[i](
                    tf.expand_dims(sub_in[:, j, :], axis=1), sub_in)

                sub_out.append(attention)

            # Concatenate the result to have shape (batch_size, length, model_size)
            sub_out = tf.concat(sub_out, axis=1)

            # Residual connection
            sub_out = sub_in + sub_out
            # Normalize the output
            sub_out = self.attention_norm[i](sub_out)

            # The FFN input is the output of the Multi-Head Attention
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            # Add the residual connection
            ffn_out = ffn_in + ffn_out
            # Normalize the output
            ffn_out = self.ffn_norm[i](ffn_out)

            # Assign the FFN output to the next layer's Multi-Head Attention input
            sub_in = ffn_out

        # Return the result when done
        return ffn_out


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.attention_bot = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(model_size, h) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        # FFN
        self.dense_1 = [tf.keras.layers.Dense(model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(model_size) for _ in range(num_layers)]

        # Normalization
        self.ffn_norm = [tf.keras.layers.BatchNormalization() for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output):
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = []
        for i in range(sequence.shape[1]):
            embed = self.embedding(tf.expand_dims(sequence[:, i], axis=1))
            embed_out.append(embed + pos_encoding[i, :])

        embed_out = tf.concat(embed_out, axis=1)
        bot_sub_in = embed_out

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            bot_sub_out = []

            for j in range(bot_sub_in.shape[1]):
                # the value vector must not contain tokens that lies on the right of the current token
                values = bot_sub_in[:, :j, :]
                attention = self.attention_bot[i](
                    tf.expand_dims(bot_sub_in[:, j, :], axis=1), values)

                bot_sub_out.append(attention)
            bot_sub_out = tf.concat(bot_sub_out, axis=1)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out = []
            for j in range(mid_sub_in.shape[1]):
                attention = self.attention_mid[i](
                    tf.expand_dims(mid_sub_in[:, j, :], axis=1), encoder_output)

                mid_sub_out.append(attention)

            mid_sub_out = tf.concat(mid_sub_out, axis=1)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits


class MultiHeadAttention(tf.keras.Model):
    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.query_size = model_size // h
        self.key_size = model_size // h
        self.value_size = model_size // h
        self.h = h
        self.wq = [tf.keras.layers.Dense(self.query_size) for _ in range(h)]
        self.wk = [tf.keras.layers.Dense(self.key_size) for _ in range(h)]
        self.wv = [tf.keras.layers.Dense(self.value_size) for _ in range(h)]
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value):
        heads = []
        for i in range(self.h):
            score = tf.matmul(self.wq[i](query), self.wk[i](value), transpose_b=True)

            score /= tf.math.sqrt(tf.dtypes.cast(self.key_size, tf.float32))

            alignment = tf.nn.softmax(score, axis=2)

            head = tf.matmul(alignment, self.wv[i](value))

            heads.append(head)

        heads = tf.concat(heads, axis=2)
        heads = self.wo(heads)

        return heads


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
            raw_data_ques.append(qas['question'])
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
    raw_data_ques = [normalize_string(data) for data in raw_data_ques]
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
    encoder_vocab_size = len(ques_tokenizer.word_index) + 1
    decoder_vocab_size = len(ans_tokenizer.word_index) + 1
    return data_encoder, data_decoder_in, data_decoder_out, encoder_vocab_size, decoder_vocab_size


def positional_embedding(pos, model_size):
    POS_EMB = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            POS_EMB[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            POS_EMB[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return POS_EMB


def loss_func(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


@tf.function
def train_step(source_seq, target_seq_in, target_seq_out):
    with tf.GradientTape() as tape:
        encoder_output = encoder(source_seq)

        decoder_output = decoder(target_seq_in, encoder_output)

        loss = loss_func(target_seq_out, decoder_output)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


def predict(test_source_text=None):
    # If test sentence is not provided
    # randomly pick up one from the training data
    if test_source_text is None:
        test_source_text = raw_data_ques[np.random.choice(len(raw_data_ques))]

    print(test_source_text)

    # Tokenize the test sentence to obtain source sequence
    test_source_seq = ques_tokenizer.texts_to_sequences([test_source_text])
    print(test_source_seq)

    en_output = encoder(tf.constant(test_source_seq))

    de_input = tf.constant([[ans_tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output = decoder(de_input, en_output)

        # Take the last token as the predicted token
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(ans_tokenizer.index_word[new_word.numpy()[0][0]])

        # The next input is a new sequence
        # contains both the input sequence and the predicted token
        de_input = tf.concat((de_input, new_word), axis=-1)

        # End if hitting <end> or length exceeds 14
        if out_words[-1] == '<end>' or len(out_words) >= 14:
            break

    print(' '.join(out_words))


if __name__ == '__main__':
    # Do not amend these attributes
    ques_tokenizer = []
    ans_tokenizer = []
    raw_data_ques = []

    # Slightly error prone but should be ok
    print("Checking if dataset exists...")
    if os.path.isfile(TRAIN_FILE):
        print("Dataset exists")
    else:
        print("Dataset does not exist")
        download_dataset()

    raw_train_data = load_data(TRAIN_FILE)
    raw_eval_data = load_data(DEV_FILE)

    data_ques, data_ans_in, data_ans_out, ques_vocab_size, ans_vocab_size = prepare_data(raw_train_data)

    max_length = max(len(data_ques[0]), len(data_ans_in[0]))

    print("Determining Positional Encoding")
    pos_encoding = []
    for i in range(max_length):
        pos_encoding.append(positional_embedding(i, MODEL_SIZE))

    pos_encoding = np.concatenate(pos_encoding, axis=0)
    pos_encoding = tf.constant(pos_encoding, dtype=tf.float32)

    print("Building TF Dataset")
    dataset = tf.data.Dataset.from_tensor_slices((data_ques, data_ans_in, data_ans_out))
    dataset = dataset.shuffle(20).batch(BATCH_SIZE)

    print("Building Transformer Model")
    encoder = Encoder(ques_vocab_size, MODEL_SIZE, NUM_LAYERS, ATTENTION_HEADS)

    ques_sequence_in = tf.constant([[1, 2, 3, 4, 6, 7, 8, 0, 0, 0],
                                    [1, 2, 3, 4, 6, 7, 8, 0, 0, 0]])
    encoder_output = encoder(ques_sequence_in)

    max_len_ans = data_ans_in.shape[1]

    decoder = Decoder(ans_vocab_size, MODEL_SIZE, NUM_LAYERS, ATTENTION_HEADS)
    ans_sequence_in = tf.constant([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0],
                                   [1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0]])
    decoder_output = decoder(ans_sequence_in, encoder_output)
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    start_time = time.time()
    for epoch in range(EPOCHS):
        for batch, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in, target_seq_out)

        print('Epoch {} loss {:.4f}'.format(epoch + 1, loss.numpy()))

        if(epoch + 1) % 10 == 0:
            end_time = time.time()
            print('Average elapsed time: {:.2f}s'.format((end_time - start_time) / (epoch + 1)))
            try:
                predict()
            except Exception as e:
                print(e)
                continue
