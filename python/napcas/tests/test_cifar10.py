import pytest
from napcas import DataLoader, Conv2d, CrossEntropyLoss, Adam

def test_cifar10():
    loader = DataLoader("data/cifar10", batch_size=64)
    model = Conv2d(3, 16, 3)
    optimizer = Adam([model], lr=0.001)
    loss_fn = CrossEntropyLoss()
    input, target = loader.next()
    output = Tensor([64, 16, 30, 30])  # Example output shape
    model.forward(input, output)
    loss = loss_fn.forward(output, target)
    assert loss >= 0
    optimizer.step()
