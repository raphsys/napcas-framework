import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps)

def cross_entropy_loss(y_pred, y_true):
    # Implémentation de la perte croisée
    pass
