import pytest
from napcas import LSTM, GRU, Tensor

def test_lstm():
    model = LSTM(10, 20, 2)
    input = Tensor([5, 2, 10], [float(i % 10) for i in range(100)])
    output = Tensor([5, 2, 20])
    model.forward(input, output)
    assert output.shape() == [5, 2, 20]

def test_gru():
    model = GRU(10, 20, 2)
    input = Tensor([5, 2, 10], [float(i % 10) for i in range(100)])
    output = Tensor([5, 2, 20])
    model.forward(input, output)
    assert output.shape() == [5, 2, 20]
