import tensorflow as tf

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self._batch_size = batch_size

        # Define the placeholders
        self._states = None
        self._action = None

        # The output operations
        self._logits = None
        self._optimizer = None
        self._var_init = None

        # Setup Model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=)