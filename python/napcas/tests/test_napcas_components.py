import pytest
from napcas import NAPCA_Sim, NNCel, Tensor

def test_napca_sim():
    model = NAPCA_Sim(10, 5, alpha=0.6, threshold=0.5)
    input = Tensor([2, 10], [float(i) for i in range(20)])
    output = Tensor([2, 5])
    model.forward(input, output)
    assert output.shape() == [2, 5]
    model.prune_connections(0.01)
    assert model.get_weights().shape() == [5, 10]

def test_nncel():
    model = NNCel(10, 5)
    input = Tensor([2, 10], [float(i) for i in range(20)])
    output = Tensor([2, 5])
    model.forward(input, output)
    assert output.shape() == [2, 5]
