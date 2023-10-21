from .generic_layer import Layer
from scipy import signal
import numpy as np

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, learning_rate=0.01):
        batch_size, input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_depth = input_depth
        self.output_shape = (batch_size, depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.dkernels = np.zeros(self.kernels_shape)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, inputs, training):
        self.input = inputs
        self.output = np.copy(self.biases)
        for b in range(self.batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.output[b][i] += signal.correlate2d(self.input[b][j], self.kernels[i, j], "valid")

    def backward(self, dvalues):
        input_gradient = np.zeros(self.input_shape)
        self.dbiases = dvalues
        for b in range(self.batch_size):
            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.dkernels[i, j] = signal.correlate2d(self.input[b][j], self.dbiases[b][i], "valid")
                    input_gradient[b][j] += signal.convolve2d(self.dbiases[b][i], self.kernels[i, j], "full")

        self.dinputs = input_gradient
