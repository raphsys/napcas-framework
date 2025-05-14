from napcas import LSTM, DataLoader, SGD, MSELoss, Tensor

def main():
    loader = DataLoader("data/synthetic.csv", batch_size=2, augment=True)
    model = LSTM(10, 20, 2)
    optimizer = SGD([model], lr=0.01)
    loss_fn = MSELoss()

    for epoch in range(5):
        input, target = loader.next()
        input = input.reshape([1, 2, 10])  # seq_len=1 for simplicity
        target = target.reshape([1, 2, 1])
        output = Tensor([1, 2, 20])
        model.forward(input, output)
        loss = loss_fn.forward(output, target)
        grad_output = loss_fn.backward(output, target)
        grad_input = Tensor(input.shape())
        model.backward(grad_output, grad_input)
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss}")

if __name__ == "__main__":
    main()
