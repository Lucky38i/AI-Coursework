"""
This is a Transformer Model developed using multiple sources listed below:
https://machinetalk.org/2019/04/29/create-the-transformer-with-tensorflow-2-0/
https://www.tensorflow.org/tutorials/text/transformer

Credit goes towards Trung Tran at MachineTalk for explaining Multi-Head Attention and the semantics behind
a Transformer Model. Several altercations have been made to his code to preprocess the dataset being used
Minor changes have been made to the Hyper-Parameters as-well as configurations in the Model's layers including
Dropout to prevent over-fitting on this dataset.

Alex McBean N0696066
"""
import glob
import os
import time

from data_handling import create_dataset
from transformer_model import create_transformer, create_optimizer, train_step, predict

# Hyper-Parameters
EPOCHS = 100
MODEL_SIZE = 512
ATTENTION_HEADS = 8
NUM_LAYERS = 6
BATCH_SIZE = 128
INIT_LEARN_RATE = 5e-3

# File Locations
CHECKPOINT_DIR = "data/checkpoints"

if __name__ == '__main__':
    dataset, info = create_dataset(BATCH_SIZE)
    encoder, decoder = create_transformer(info['vocab_size'], MODEL_SIZE,
                                          info['max_length'], NUM_LAYERS, ATTENTION_HEADS)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    encoder_ckpt_path = os.path.join(
        CHECKPOINT_DIR, 'encoder_epoch_{}.h5')
    decoder_ckpt_path = os.path.join(
        CHECKPOINT_DIR, 'decoder_epoch_{}.h5')

    encoder_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, 'encoder*.h5'))
    decoder_ckpts = glob.glob(os.path.join(CHECKPOINT_DIR, 'decoder*.h5'))
    epoch_start = 0

    if len(encoder_ckpts) > 0 and len(decoder_ckpts) > 0:
        latest_encoder_ckpt = max(encoder_ckpts, key=os.path.getctime)
        encoder.load_weights(latest_encoder_ckpt)
        latest_decoder_ckpt = max(decoder_ckpts, key=os.path.getctime)
        decoder.load_weights(latest_decoder_ckpt)
        epoch_start = int(latest_encoder_ckpt[latest_encoder_ckpt.rfind('_')+1:-3])

    num_steps = info['data_size'] // BATCH_SIZE
    print('num steps', num_steps)
    optimizer = create_optimizer(
        MODEL_SIZE, initial_lr=INIT_LEARN_RATE, trained_steps=epoch_start*num_steps)

    start_time = time.time()
    for epoch in range(epoch_start, EPOCHS):
        avg_loss = 0.0
        for i, (source_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
            loss = train_step(source_seq, target_seq_in, target_seq_out,
                              encoder, decoder, optimizer)
            avg_loss = (avg_loss * i + loss.numpy()) / (i + 1)

            if (i + 1) % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f} Elapsed time {:.2f}s'.format(
                    epoch + 1, i + 1, avg_loss, time.time() - start_time))
                start_time = time.time()

        encoder.save_weights(encoder_ckpt_path.format(epoch + 1))
        decoder.save_weights(decoder_ckpt_path.format(epoch + 1))

