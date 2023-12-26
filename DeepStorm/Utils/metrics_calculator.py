import numpy as np

def calc_accuracy(preds, labels):
    preds = np.argmax(preds, axis=1)
    labels = np.argmax(labels, axis=1)
    return np.mean(preds == labels)
