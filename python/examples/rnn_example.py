from napcas import RNN, DataLoader, MSELoss, SGD

def main():
    loader = DataLoader("data/sequence.csv", batch_size=16)
    model = RNN(10, 20, 2)
    optimizer = SGD([model], lr=0.01)
    loss_fn = MSELoss()

    for epoch in range(10):
        input, target = loader.next()
        output = Tensor([5, 16, 20])
        model.forward(input, output)
        loss = loss_fn.forward(output, target)
        grad_output = loss_fn.backward(output, target)
        grad_input = Tensor(input.shape())
        model.backward(grad_output, grad_input)
        optimizer.step()

if __name__ == "__main__":
    main()
