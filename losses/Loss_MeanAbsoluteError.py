from generic_loss import Loss
import numpy as np

class Loss_MeanAbsoluteError(Loss):  # L1 loss

    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
