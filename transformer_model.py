import tensorflow as tf
import numpy as np

crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)


def loss_func(targets, logits):
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    mask = tf.cast(mask, dtype=tf.int64)
    loss = crossentropy(targets, logits, sample_weight=mask)

    return loss


class WarmupThenDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Learning schedule for training the Transformer
    Attributes:
        model_size: d_model in the paper (depth size of the model)
        warmup_steps: number of warmup steps at the beginning
    """

    def __init__(self, model_size, initial_lr, warmup_steps=4000, trained_steps=0):
        super(WarmupThenDecaySchedule, self).__init__()

        self.trained_steps = trained_steps
        self.model_size = model_size
        self.initial_lr = initial_lr
        self.model_size = tf.cast(self.model_size, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step_term = tf.math.rsqrt(step + self.trained_steps)
        warmup_term = step * (self.warmup_steps ** -1.5)

        if self.initial_lr:
            return self.initial_lr * tf.math.minimum(step_term, warmup_term)

        return tf.math.rsqrt(self.model_size) * tf.math.minimum(step_term, warmup_term)


def create_optimizer(model_size, initial_lr, trained_steps):
    lr = WarmupThenDecaySchedule(model_size, initial_lr=initial_lr, trained_steps=trained_steps)
    optimizer = tf.keras.optimizers.Adam(lr,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    return optimizer


@tf.function
def train_step(source_seq, target_seq_in,
               target_seq_out, encoder, decoder, optimizer):
    """ Execute one training step (forward pass + backward pass)
    Args:
        source_seq: source sequences
        target_seq_in: input target sequences (<start> + ...)
        target_seq_out: output target sequences (... + <end>)
    Returns:
        The loss value of the current pass
    """
    with tf.GradientTape() as tape:
        encoder_mask = 1 - tf.cast(tf.equal(source_seq, 0), dtype=tf.float32)
        # encoder_mask has shape (batch_size, source_len)
        # we need to add two more dimensions in between
        # to make it broadcastable when computing attention heads
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_mask = tf.expand_dims(encoder_mask, axis=1)
        encoder_output, _ = encoder(source_seq, encoder_mask=encoder_mask)

        decoder_output, _, _ = decoder(
            target_seq_in, encoder_output, encoder_mask=encoder_mask)

        loss = loss_func(target_seq_out, decoder_output)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss


"""## Positional Encoding"""


def positional_encoding(pos, model_size):
    """ Compute positional encoding for a particular position
    Args:
        pos: position of a token in the sequence
        model_size: depth size of the model

    Returns:
        The positional encoding for the given token
    """
    pos_encoding = np.zeros((1, model_size))
    for i in range(model_size):
        if i % 2 == 0:
            pos_encoding[:, i] = np.sin(pos / 10000 ** (i / model_size))
        else:
            pos_encoding[:, i] = np.cos(pos / 10000 ** ((i - 1) / model_size))
    return pos_encoding


"""## Common Embedding"""


class CommonEmbedding(tf.keras.Model):
    def __init__(self, vocab_size, model_size, max_length):
        super(CommonEmbedding, self).__init__()
        pes = []
        for i in range(max_length):
            pes.append(positional_encoding(i, model_size))

        pes = np.concatenate(pes, axis=0)
        self.pes = tf.constant(pes, dtype=tf.float32)
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        self.embedding_dropout = tf.keras.layers.Dropout(0.3)
        self.model_size = model_size

    def call(self, sequence):
        embed_out = self.embedding(sequence)

        embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        embed_out += self.pes[:sequence.shape[1], :]
        embed_out = self.embedding_dropout(embed_out)

        return embed_out


"""## Create the Multi-head Attention layer"""


class MultiHeadAttention(tf.keras.Model):
    """ Class for Multi-Head Attention layer
    Attributes:
        key_size: d_key in the paper
        h: number of attention heads
        wq: the Linear layer for Q
        wk: the Linear layer for K
        wv: the Linear layer for V
        wo: the Linear layer for the output
    """

    def __init__(self, model_size, h):
        super(MultiHeadAttention, self).__init__()
        self.key_size = model_size // h
        self.h = h
        self.wq = tf.keras.layers.Dense(model_size)
        self.wk = tf.keras.layers.Dense(model_size)
        self.wv = tf.keras.layers.Dense(model_size)
        self.wo = tf.keras.layers.Dense(model_size)

    def call(self, query, value, mask=None):
        """ The forward pass for Multi-Head Attention layer
        Args:
            query: the Q matrix
            value: the V matrix, acts as V and K
            mask: mask to filter out unwanted tokens
                  - zero mask: mask for padded tokens
                  - right-side mask: mask to prevent attention towards tokens on the right-hand side

        Returns:
            The concatenated context vector
            The alignment (attention) vectors of all heads
        """
        # query has shape (batch, query_len, model_size)
        # value has shape (batch, value_len, model_size)
        query = self.wq(query)
        key = self.wk(value)
        value = self.wv(value)

        # Split matrices for multi-heads attention
        batch_size = query.shape[0]

        # Originally, query has shape (batch, query_len, model_size)
        # We need to reshape to (batch, query_len, h, key_size)
        query = tf.reshape(query, [batch_size, -1, self.h, self.key_size])
        # In order to compute matmul, the dimensions must be transposed to (batch, h, query_len, key_size)
        query = tf.transpose(query, [0, 2, 1, 3])

        # Do the same for key and value
        key = tf.reshape(key, [batch_size, -1, self.h, self.key_size])
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.reshape(value, [batch_size, -1, self.h, self.key_size])
        value = tf.transpose(value, [0, 2, 1, 3])

        # Compute the dot score
        # and divide the score by square root of key_size (as stated in paper)
        # (must convert key_size to float32 otherwise an error would occur)
        score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(tf.dtypes.cast(self.key_size, dtype=tf.float32))
        # score will have shape of (batch, h, query_len, value_len)

        # Mask out the score if a mask is provided
        # There are two types of mask:
        # - Padding mask (batch, 1, 1, value_len): to prevent attention being drawn to padded token (i.e. 0)
        # - Look-left mask (batch, 1, query_len, value_len): to prevent decoder to draw attention to tokens to the right
        if mask is not None:
            score *= mask

            # We want the masked out values to be zeros when applying softmax
            # One way to accomplish that is assign them to a very large negative value
            score = tf.where(tf.equal(score, 0), tf.ones_like(score) * -1e9, score)

        # Alignment vector: (batch, h, query_len, value_len)
        alignment = tf.nn.softmax(score, axis=-1)

        # Context vector: (batch, h, query_len, key_size)
        context = tf.matmul(alignment, value)

        # Finally, do the opposite to have a tensor of shape (batch, query_len, model_size)
        context = tf.transpose(context, [0, 2, 1, 3])
        context = tf.reshape(context, [batch_size, -1, self.key_size * self.h])

        # Apply one last full connected layer (WO)
        heads = self.wo(context)

        return heads, alignment


"""## Create the Encoder"""


class Encoder(tf.keras.Model):
    """ Class for the Encoder
    Args:
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention: array of Multi-Head Attention layers
        attention_dropout: array of Dropout layers for Multi-Head Attention
        attention_norm: array of LayerNorm layers for Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
    """

    def __init__(self, embedding, vocab_size, model_size, num_layers, h):
        super(Encoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        # self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        # self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.embedding = embedding
        self.attention = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_dropout = [
            tf.keras.layers.Dropout(0.3) for _ in range(num_layers)]

        self.attention_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(
            0.3) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

    def call(self, sequence, training=True, encoder_mask=None):
        """ Forward pass for the Encoder
        Args:
            sequence: source input sequences
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The alignment (attention) vectors for all layers
        """
        embed_out = self.embedding(sequence)

        # embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        # embed_out += pes[:sequence.shape[1], :]
        # embed_out = self.embedding_dropout(embed_out)

        sub_in = embed_out
        alignments = []

        for i in range(self.num_layers):
            sub_out, alignment = self.attention[i](
                sub_in, sub_in, encoder_mask)
            sub_out = self.attention_dropout[i](sub_out, training=training)
            sub_out = sub_in + sub_out
            sub_out = self.attention_norm[i](sub_out)

            alignments.append(alignment)
            ffn_in = sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_in + ffn_out
            ffn_out = self.ffn_norm[i](ffn_out)

            sub_in = ffn_out

        return ffn_out, alignments


"""## Create the Decoder"""


class Decoder(tf.keras.Model):
    """ Class for the Decoder
    Args:
        model_size: d_model in the paper (depth size of the model)
        num_layers: number of layers (Multi-Head Attention + FNN)
        h: number of attention heads
        embedding: Embedding layer
        embedding_dropout: Dropout layer for Embedding
        attention_bot: array of bottom Multi-Head Attention layers (self attention)
        attention_bot_dropout: array of Dropout layers for bottom Multi-Head Attention
        attention_bot_norm: array of LayerNorm layers for bottom Multi-Head Attention
        attention_mid: array of middle Multi-Head Attention layers
        attention_mid_dropout: array of Dropout layers for middle Multi-Head Attention
        attention_mid_norm: array of LayerNorm layers for middle Multi-Head Attention
        dense_1: array of first Dense layers for FFN
        dense_2: array of second Dense layers for FFN
        ffn_dropout: array of Dropout layers for FFN
        ffn_norm: array of LayerNorm layers for FFN
        dense: Dense layer to compute final output
    """

    def __init__(self, embedding, vocab_size, model_size, num_layers, h):
        super(Decoder, self).__init__()
        self.model_size = model_size
        self.num_layers = num_layers
        self.h = h
        # self.embedding = tf.keras.layers.Embedding(vocab_size, model_size)
        # self.embedding_dropout = tf.keras.layers.Dropout(0.1)
        self.embedding = embedding
        self.attention_bot = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_bot_dropout = [
            tf.keras.layers.Dropout(0.3) for _ in range(num_layers)]
        self.attention_bot_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]
        self.attention_mid = [MultiHeadAttention(
            model_size, h) for _ in range(num_layers)]
        self.attention_mid_dropout = [
            tf.keras.layers.Dropout(0.3) for _ in range(num_layers)]
        self.attention_mid_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense_1 = [tf.keras.layers.Dense(
            model_size * 4, activation='relu') for _ in range(num_layers)]
        self.dense_2 = [tf.keras.layers.Dense(
            model_size) for _ in range(num_layers)]
        self.ffn_dropout = [tf.keras.layers.Dropout(
            0.1) for _ in range(num_layers)]
        self.ffn_norm = [tf.keras.layers.LayerNormalization(
            epsilon=1e-6) for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, sequence, encoder_output, training=True, encoder_mask=None):
        """ Forward pass for the Decoder
        Args:
            sequence: source input sequences
            encoder_output: output of the Encoder (for computing middle attention)
            training: whether training or not (for Dropout)
            encoder_mask: padding mask for the Encoder's Multi-Head Attention
        Returns:
            The output of the Encoder (batch_size, length, model_size)
            The bottom alignment (attention) vectors for all layers
            The middle alignment (attention) vectors for all layers
        """
        # EMBEDDING AND POSITIONAL EMBEDDING
        embed_out = self.embedding(sequence)

        # embed_out *= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        # embed_out += pes[:sequence.shape[1], :]
        # embed_out = self.embedding_dropout(embed_out)

        bot_sub_in = embed_out
        bot_alignments = []
        mid_alignments = []

        for i in range(self.num_layers):
            # BOTTOM MULTIHEAD SUB LAYER
            seq_len = bot_sub_in.shape[1]

            if training:
                mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
            else:
                mask = None
            bot_sub_out, bot_alignment = self.attention_bot[i](
                bot_sub_in, bot_sub_in, mask)
            bot_sub_out = self.attention_bot_dropout[i](
                bot_sub_out, training=training)
            bot_sub_out = bot_sub_in + bot_sub_out
            bot_sub_out = self.attention_bot_norm[i](bot_sub_out)

            bot_alignments.append(bot_alignment)

            # MIDDLE MULTIHEAD SUB LAYER
            mid_sub_in = bot_sub_out

            mid_sub_out, mid_alignment = self.attention_mid[i](
                mid_sub_in, encoder_output, encoder_mask)
            mid_sub_out = self.attention_mid_dropout[i](
                mid_sub_out, training=training)
            mid_sub_out = mid_sub_out + mid_sub_in
            mid_sub_out = self.attention_mid_norm[i](mid_sub_out)

            mid_alignments.append(mid_alignment)

            # FFN
            ffn_in = mid_sub_out

            ffn_out = self.dense_2[i](self.dense_1[i](ffn_in))
            ffn_out = self.ffn_dropout[i](ffn_out, training=training)
            ffn_out = ffn_out + ffn_in
            ffn_out = self.ffn_norm[i](ffn_out)

            bot_sub_in = ffn_out

        logits = self.dense(ffn_out)

        return logits, bot_alignments, mid_alignments


'''## Transformer Model Creation'''


def create_transformer(vocab_size, model_size, max_length, num_layers, h):
    embedding = CommonEmbedding(vocab_size, model_size, max_length * 2)
    encoder = Encoder(embedding, vocab_size, model_size, num_layers, h)
    decoder = Decoder(embedding, vocab_size, model_size, num_layers, h)

    # Build the network by putting in some random tensors
    encoder_out, _ = encoder(tf.constant([[1, 2, 3, 4, 5]]))
    decoder_out, _, _ = decoder(tf.constant([[1, 2, 3, 4, 5]]), encoder_out)

    return encoder, decoder


def predict(encoder, decoder, tokenizer, raw_input_lines, max_length):
    """ Predict the output sentence for a given input sentence
    Args:
        test_source_text: input sentence (raw string)
    Returns:
        The encoder's attention vectors
        The decoder's bottom attention vectors
        The decoder's middle attention vectors
        The input string array (input sentence split by ' ')
        The output string array
    """
    test_source_text = np.random.choice(raw_input_lines)
    # print(test_source_text)
    test_source_seq = tokenizer.texts_to_sequences([test_source_text])
    # print(test_source_seq)

    en_output, en_alignments = encoder(
        tf.constant(test_source_seq), training=False)

    de_input = tf.constant(
        [[tokenizer.word_index['<start>']]], dtype=tf.int64)

    out_words = []

    while True:
        de_output, de_bot_alignments, de_mid_alignments = decoder(
            de_input, en_output, training=False)
        new_word = tf.expand_dims(tf.argmax(de_output, -1)[:, -1], axis=1)
        out_words.append(tokenizer.index_word[new_word.numpy()[0][0]])

        # Transformer doesn't have sequential mechanism (i.e. states)
        # so we have to add the last predicted word to create a new input sequence
        de_input = tf.concat((de_input, new_word), axis=-1)

        # TODO: get a nicer constraint for the sequence length!
        if out_words[-1] == '<end>' or len(out_words) >= max_length:
            break

    print(' '.join(out_words[:-1]))
    return en_alignments, de_bot_alignments, de_mid_alignments, test_source_text.split(' '), out_words
