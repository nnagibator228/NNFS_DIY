from models.Model import Model
from layers.Convolutional import Convolutional
from layers.activations.Activation_Sigmoid import Activation_Sigmoid
from layers.Reshape import Reshape
from layers.Layer_Dense import Layer_Dense
from layers.activations.Activation_ReLU import Activation_ReLU
from layers.activations.Activation_Softmax import Activation_Softmax
from losses.Loss_CategoricalCrossentropy import Loss_CategoricalCrossentropy
from optimizers.Optimizer_Adam import Optimizer_Adam
from accuracy.Accuracy_Categorical import Accuracy_Categorical
from utils.preprocess_data import X, y, X_test, y_test, items, classes

model = Model()

model.add(Convolutional((items * classes, 1, 28, 28), 3, 5))
model.add(Activation_Sigmoid())
model.add(Reshape((items * classes, 5, 26, 26), (items * classes, 5 * 26 * 26, 1)))
model.add(Layer_Dense(5 * 26 * 26, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, classes))
model.add(Activation_Softmax())

model.set(
    loss=Loss_CategoricalCrossentropy(),
    optimizer=Optimizer_Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=75, batch_size=items * classes, print_every=100)
print("-" * 21)
print("Testing with test vals: ")
model.evaluate(X_test, y_test)