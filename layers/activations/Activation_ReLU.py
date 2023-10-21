from .generic_activation import Activation
import numpy as np

class Activation_ReLU(Activation):

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs
