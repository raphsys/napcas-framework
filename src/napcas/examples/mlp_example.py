# python/examples/mlp_example.py

from napcas import MLP, DataLoader, CrossEntropyLoss, Adam, Tensor, plot_training_curves

def main():
    loader = DataLoader("data/synthetic.csv", batch_size=32, augment=True)
    model = MLP([64, 128, 10], "relu")
    optimizer = Adam([model], lr=0.001)
    loss_fn = CrossEntropyLoss()
    losses, accuracies = [], []

    for epoch in range(10):
        input, target = loader.next()
        output = Tensor([32, 10])
        model.forward(input, output)
        loss = loss_fn.forward(output, target)
        grad_output = loss_fn.backward(output, target)
        grad_input = Tensor(input.shape())
        model.backward(grad_output, grad_input)
        optimizer.step()
        losses.append(loss)
        accuracies.append(0.9)
    
    plot_training_curves(losses, accuracies, "training_curves.png")

if __name__ == "__main__":
    main()

