from napcas import ReLU, Sigmoid, Tanh

class ReLU:
    def __init__(self):
        self.relu = ReLU()

    def forward(self, x):
        return self.relu.forward(x)

    def backward(self, grad_output):
        return self.relu.backward(grad_output)

class Sigmoid:
    def __init__(self):
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.sigmoid.forward(x)

    def backward(self, grad_output):
        return self.sigmoid.backward(grad_output)

class Tanh:
    def __init__(self):
        self.tanh = Tanh()

    def forward(self, x):
        return self.tanh.forward(x)

    def backward(self, grad_output):
        return self.tanh.backward(grad_output)
