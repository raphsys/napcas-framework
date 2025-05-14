import pytest
from napcas import RNN, Tensor

def test_rnn():
    model = RNN(10, 20, 2)
    input = Tensor([5, 2, 10], [float(i) for i in range(100)])
    output = Tensor([5, 2, 20])
    model.forward(input, output)
    assert output.shape() == [5, 2, 20]
