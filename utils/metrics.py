import numpy as np

def accuracy(X,y):
    # print(X)
    # print(y)
    X = np.argmax(X, axis=-1)
    y = np.argmax(y, axis=-1)
    acc = np.sum(X == y) / np.size(X)

    return acc


def accuracy_per_doc(classifieds):

    aciertos = 0

    for c in classifieds:
        if c[0] == c[1]:
            aciertos += 1

    acc = aciertos / len(classifieds)

    return acc