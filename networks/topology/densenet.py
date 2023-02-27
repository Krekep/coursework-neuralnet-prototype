from typing import List

import tensorflow as tf
from tensorflow import keras

from networks.ilayer import ILayer


class DenseNet(tf.keras.Model):
    def __init__(
        self,
        input_size: int = 2,
        block_size: list = None,
        output_size: int = 10,
        activation_func=tf.keras.activations.linear,
        weight=keras.initializers.get("ones"),
        biases=keras.initializers.get("zeros"),
        is_debug: bool = False,
        **kwargs,
    ):
        activation_names = None
        decorator_params = None
        if "activation_names" in kwargs.keys():
            activation_names = kwargs.get("activation_names")
            kwargs.pop("activation_names")
        else:
            activation_names = [activation_func.__name__] * (len(block_size) + 1)

        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")
        else:
            decorator_params = None

        if decorator_params is None:
            decorator_params = [None] * (len(block_size) + 1)

        super(DenseNet, self).__init__(**kwargs)
        self.blocks = []

        if not isinstance(activation_func, list):
            activation_func = [activation_func] * (len(block_size) + 1)
        if len(block_size) != 0:
            self.blocks.append(
                ILayer.create_dense(
                    input_size,
                    block_size[0],
                    activation=activation_func[0],
                    weight=weight,
                    bias=biases,
                    is_debug=is_debug,
                    name=f"MyDense0",
                    activation_names=activation_names[0],
                    decorator_params=decorator_params[0],
                )
            )
            for i in range(1, len(block_size)):
                self.blocks.append(
                    ILayer.create_dense(
                        block_size[i - 1],
                        block_size[i],
                        activation=activation_func[i],
                        weight=weight,
                        bias=biases,
                        is_debug=is_debug,
                        name=f"MyDense{i}",
                        activation_names=activation_names[i],
                        decorator_params=decorator_params[i],
                    )
                )
        else:
            block_size = [input_size]

        self.classifier = ILayer.create_dense(
            block_size[-1],
            output_size,
            activation=activation_func[-1],
            weight=weight,
            bias=biases,
            is_debug=is_debug,
            name=f"OutputLayerMyDense",
            activation_names=activation_names[-1],
            decorator_params=decorator_params[-1],
        )

        self.activation_func = activation_func
        self.weight_initializer = weight
        self.bias_initializer = biases
        self.input_size = input_size
        self.block_size = block_size
        self.output_size = output_size

    def call(self, inputs, **kwargs):
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
        res = f"IModel {self.name}\n"
        for layer in self.blocks:
            res += str(layer)
        res += "Classifier to string not implemented\n"
        return res

    def to_dict(self, **kwargs):
        res = {
            "net_type": "MyDense",
            "name": self._name,
            "input_size": self.input_size,
            "block_size": self.block_size,
            "output_size": self.output_size,
            # "activation_func": activations.serialize(self.activation_func),       # Layers have own activation param
            # "out_activation": activations.serialize(self.out_activation),         # Layers have own activation param
            "layer": [],
            "classifier": self.classifier.to_dict(),
        }

        for i, layer in enumerate(self.blocks):
            res["layer"].append(layer.to_dict())

        return res

    @classmethod
    def from_layers(
        cls,
        input_size: int,
        block_size: List[int],
        output_size: int,
        layers: List[ILayer],
        **kwargs,
    ):
        res = cls(
            input_size=input_size,
            block_size=block_size,
            output_size=output_size,
            **kwargs,
        )

        for layer_num in range(len(res.blocks)):
            res.blocks[layer_num] = layers[layer_num]

        return res

    def from_dict(self, config, **kwargs):
        input_size = config["input_size"]
        block_size = config["block_size"]
        output_size = config["output_size"]

        layers: List[ILayer] = []
        for layer_config in config["layer"]:
            layers.append(ILayer.from_dict(layer_config))

        for layer_num in range(len(self.blocks)):
            self.blocks[layer_num] = layers[layer_num]

        self.classifier = ILayer.from_dict(config["classifier"])

    def get_config(self):
        config = super(
            DenseNet, self
        ).get_config()  # get config of the base Layer class

        # config.update({'units': self.units,
        #                'activation_func': tf.keras.activations.serialize(self.activation_func),
        #                'weight': tf.keras.initializers.serialize(self.weight_initializer),
        #                'biases': tf.keras.initializers.serialize(self.bias_initializer),
        #                })
        # you need to serialize the callable activation function

        return config
