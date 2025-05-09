from napcas import Linear

class Linear:
    def __init__(self, in_features, out_features):
        self.linear = Linear(in_features, out_features)

    def forward(self, x):
        return self.linear.forward(x)

    def backward(self, grad_output):
        return self.linear.backward(grad_output)

    def update(self, lr):
        self.linear.update(lr)
