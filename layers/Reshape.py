from .generic_layer import Layer
import numpy as np

class Reshape:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input, training):
        self.input = input
        batch_size = input.shape[0]
        self.output = np.reshape(input, (self.output_shape[0], self.output_shape[1]))

    def backward(self, output_gradient):
        batch_size = output_gradient.shape[0]
        self.dinputs = np.reshape(output_gradient, self.input_shape)
