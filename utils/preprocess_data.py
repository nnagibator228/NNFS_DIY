import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical

classes = 10

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    # ---
    two_index = np.where(y == 2)[0][:limit]
    three_index = np.where(y == 3)[0][:limit]
    four_index = np.where(y == 4)[0][:limit]
    five_index = np.where(y == 5)[0][:limit]
    six_index = np.where(y == 6)[0][:limit]
    seven_index = np.where(y == 7)[0][:limit]
    eight_index = np.where(y == 8)[0][:limit]
    nine_index = np.where(y == 9)[0][:limit]

    all_indices = np.hstack((zero_index, one_index, two_index, three_index, four_index, five_index, six_index,
                            seven_index, eight_index, nine_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), classes)
    return x, y


items = 800

(X, y), (X_test, y_test) = mnist.load_data()

X, y = preprocess_data(X, y, items)
X_test, y_test = preprocess_data(X_test, y_test, items)
