import pytest
from napcas import DataLoader, NAPCAS, SGD, CrossEntropyLoss

def test_mnist():
    loader = DataLoader("data/mnist", batch_size=64)
    model = NAPCAS(784, 10)
    optimizer = SGD([model], lr=0.01)
    loss_fn = CrossEntropyLoss()
    input, target = loader.next()
    output = Tensor([64, 10])
    model.forward(input, output)
    loss = loss_fn.forward(output, target)
    assert loss >= 0
    optimizer.step()
