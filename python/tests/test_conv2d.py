import pytest
from napcas import Conv2d, MaxPool2d, Tensor

def test_conv2d():
    model = Conv2d(3, 16, 3, stride=2, padding=1)
    input = Tensor([2, 3, 32, 32], [float(i % 10) for i in range(2 * 3 * 32 * 32)])
    output = Tensor([2, 16, 16, 16])
    model.forward(input, output)
    assert output.shape() == [2, 16, 16, 16]

def test_maxpool2d():
    model = MaxPool2d(2, 2)
    input = Tensor([2, 3, 32, 32], [float(i % 10) for i in range(2 * 3 * 32 * 32)])
    output = Tensor([2, 3, 16, 16])
    model.forward(input, output)
    assert output.shape() == [2, 3, 16, 16]
