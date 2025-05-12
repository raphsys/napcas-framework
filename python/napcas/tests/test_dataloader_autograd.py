import pytest
from napcas import DataLoader, Autograd, Tensor

def test_dataloader():
    loader = DataLoader("data/test.csv", batch_size=2, augment=True)
    input, target = loader.next()
    assert input.shape() == [2, 10]  # Example shape
    assert target.shape() == [2, 1]

def test_autograd():
    tensor = Tensor([2, 2], [1.0, 2.0, 3.0, 4.0])
    Autograd.zero_grad([tensor])
    assert all(x == 0.0 for x in tensor.data())
