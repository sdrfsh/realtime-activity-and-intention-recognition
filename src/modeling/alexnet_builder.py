"""Docx 3.3: shallow AlexNet (5 conv + 3 FC layers) chosen for its known
strong performance on embedded processors. Tanh activations on layers 1-7
(ReLU dies on negative inputs, Sigmoid is asymmetric around the origin;
Tanh's symmetric (-1, 1) range gave better class separation), Softmax output,
and normal-Glorot kernel initialization (paired best with Tanh)."""
from __future__ import annotations

from keras import Input, Sequential
from keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)

from config import NetworkSettings


class AlexNetBuilder:
    """Single job: construct the (uncompiled) AlexNet architecture."""

    def __init__(self, settings: NetworkSettings) -> None:
        self._settings = settings

    def build(self) -> Sequential:
        s = self._settings
        model = Sequential(name="alexnet_activity_recognizer")

        model.add(Input(shape=s.input_shape))
        model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), padding="valid",
                          kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

        model.add(Conv2D(256, kernel_size=(11, 11), strides=(1, 1), padding="valid",
                          kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                          kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))

        model.add(Conv2D(384, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                          kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))

        model.add(Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding="valid",
                          kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

        model.add(Flatten())

        model.add(Dense(4096, kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))
        model.add(Dropout(0.5))

        model.add(Dense(1000, kernel_initializer=s.kernel_initializer))
        model.add(Activation(s.hidden_activation))
        model.add(Dropout(0.5))

        model.add(Dense(s.num_classes))
        model.add(Activation(s.output_activation))

        return model
