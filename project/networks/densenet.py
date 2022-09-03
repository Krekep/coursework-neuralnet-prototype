import tensorflow as tf
from tensorflow import keras
from keras import layers


class MyDense(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32, activation_func=tf.keras.activations.linear, is_debug=False, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )
        self._is_debug = is_debug
        self.activation_func = activation_func

    def call(self, inputs):
        return self.activation_func(tf.matmul(inputs, self.w) + self.b)

    def __str__(self):
        res = f"Layer {self.name}\n"
        res += f"weights shape = {self.w.shape}\n"
        if self._is_debug:
            res += f"weights = {self.w.numpy()}\n"
            res += f"biases = {self.b.numpy()}\n"
        return res


class DenseNet(tf.keras.Model):

    def __init__(self, input_size=2, block_size=None, output_size=10, is_debug=False, **kwargs):
        super(DenseNet, self).__init__(**kwargs)
        self.blocks = []
        if type(block_size) is list:
            self.blocks.append(
                MyDense(block_size[0], input_size, activation_func=tf.keras.activations.sigmoid, is_debug=is_debug,
                        name=f"Linear0"))
            for i in range(1, len(block_size)):
                self.blocks.append(
                    MyDense(block_size[i], block_size[i - 1], activation_func=tf.keras.activations.sigmoid,
                            is_debug=is_debug, name=f"MyDense{i}"))
        self.classifier = layers.Dense(output_size, activation="linear")

    def call(self, inputs):
        x = inputs
        for layer in self.blocks:
            x = layer(x)
        return self.classifier(x)

    def train_step(self, data):
        """
        Custom train step from tensorflow tutorial

        Parameters
        ----------
        data: tuple
            Pair of x and y (or dataset)
        Returns
        -------

        """
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def set_name(self, new_name):
        self._name = new_name

    def __str__(self):
        res = f"INetwork {self.name}\n"
        for layer in self.blocks:
            res += str(layer)
        res += "Classifier to string not implemented\n"
        return res