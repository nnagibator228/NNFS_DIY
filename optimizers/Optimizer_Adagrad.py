from .generic_optimizer import Optimizer
import numpy as np

class Optimizer_Adagrad(Optimizer):

    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if hasattr(layer, 'weights'):
            if not hasattr(layer, 'weight_cache'):
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.weight_cache += layer.dweights ** 2
            layer.bias_cache += layer.dbiases ** 2

            layer.weights += -self.current_learning_rate * \
                             layer.dweights / \
                             (np.sqrt(layer.weight_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * \
                            layer.dbiases / \
                            (np.sqrt(layer.bias_cache) + self.epsilon)

        # Для convolutional
        elif hasattr(layer, 'kernels'):
            if not hasattr(layer, 'kernel_cache'):
                layer.kernel_cache = np.zeros_like(layer.kernels)
                layer.bias_cache = np.zeros_like(layer.biases)

            layer.kernel_cache += layer.dkernels ** 2
            layer.bias_cache += layer.dbiases ** 2

            layer.kernels += -self.current_learning_rate * \
                             layer.dkernels / \
                             (np.sqrt(layer.kernel_cache) + self.epsilon)
            layer.biases += -self.current_learning_rate * \
                            layer.dbiases / \
                            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
