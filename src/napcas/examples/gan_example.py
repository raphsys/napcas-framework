# python/examples/gan_example.py

from napcas import GAN, DataLoader, MSELoss, Adam, Tensor

def main():
    loader = DataLoader("data/synthetic.csv", batch_size=64, augment=True)
    model = GAN([100, 256, 784], [784, 256, 1])
    optimizer = Adam([model], lr=0.0002, beta1=0.5, beta2=0.999)
    loss_fn = MSELoss()

    for epoch in range(10):
        input, target = loader.next()
        output = Tensor([64, 784])
        model.forward(input, output)
        loss = loss_fn.forward(output, target)
        grad_output = loss_fn.backward(output, target)
        grad_input = Tensor(input.shape())
        model.backward(grad_output, grad_input)
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss}")

if __name__ == "__main__":
    main()

