import numpy as np

def accuracy(X,y):

    X = np.argmax(X, axis=-1)
    y = np.argmax(y, axis=-1)
    acc = np.sum(X == y) / np.size(X)

    return acc