from napcas import Conv2d

class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.conv2d = Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return self.conv2d.forward(x)

    def backward(self, grad_output):
        return self.conv2d.backward(grad_output)

    def update(self, lr):
        self.conv2d.update(lr)
