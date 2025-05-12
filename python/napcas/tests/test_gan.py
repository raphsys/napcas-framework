import pytest
from napcas import GAN, Tensor

def test_gan():
    model = GAN([100, 256, 784], [784, 256, 1])
    input = Tensor([2, 100], [float(i) for i in range(200)])
    output = Tensor([2, 784])
    model.forward(input, output)
    assert output.shape() == [2, 784]
