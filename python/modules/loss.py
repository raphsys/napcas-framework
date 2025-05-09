from napcas import MSELoss, CrossEntropyLoss

class MSELoss:
    def __init__(self):
        self.loss = MSELoss()

    def forward(self, y_pred, y_true):
        return self.loss.forward(y_pred, y_true)

    def backward(self, y_pred, y_true):
        return self.loss.backward(y_pred, y_true)

class CrossEntropyLoss:
    def __init__(self):
        self.loss = CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        return self.loss.forward(y_pred, y_true)

    def backward(self, y_pred, y_true):
        return self.loss.backward(y_pred, y_true)
