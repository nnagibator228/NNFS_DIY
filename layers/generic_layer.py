class Layer:

    def forward(self, inputs, training):
        pass

    def backward(self, dvalues):
        pass


class Layer_Input:

    def forward(self, inputs, training):
        self.output = inputs