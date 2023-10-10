import os
import cv2
import numpy as np

def load_mnist_dataset(dataset, path):
    labels = os.listdir(os.path.join(path, dataset))
    X = []
    y = []
    for label in labels:
        for file in os.listdir(os.path.join(path, dataset, label)):
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            image_container = [image]
            label_container = [1 if i == int(label)-1 else 0 for i in range(10)]
            X.append(image_container)
            y.append(label_container)
        
    return np.array(X), np.array(y)

def create_data_mnist(path):
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')
print(y.shape, y[0])
