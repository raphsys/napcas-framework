from napcas import Conv2d, DataLoader, CrossEntropyLoss, Adam

def main():
    loader = DataLoader("data/cifar10", batch_size=64)
    model = Conv2d(3, 16, 3)
    optimizer = Adam([model], lr=0.001)
    loss_fn = CrossEntropyLoss()

    for epoch in range(10):
        input, target = loader.next()
        output = Tensor([64, 16, 30, 30])
        model.forward(input, output)
        loss = loss_fn.forward(output, target)
        grad_output = loss_fn.backward(output, target)
        grad_input = Tensor(input.shape())
        model.backward(grad_output, grad_input)
        optimizer.step()

if __name__ == "__main__":
    main()
