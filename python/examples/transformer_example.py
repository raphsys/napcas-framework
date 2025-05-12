from napcas import Transformer, DataLoader, CrossEntropyLoss, Adam

def main():
    loader = DataLoader("data/sequence.csv", batch_size=8)
    model = Transformer(512, 8, 6, 2048)
    optimizer = Adam([model], lr=0.0001)
    loss_fn = CrossEntropyLoss()

    for epoch in range(10):
        input, target = loader.next()
        output = Tensor([10, 8, 512])
        model.forward(input, output)
        loss = loss_fn.forward(output, target)
        grad_output = loss_fn.backward(output, target)
        grad_input = Tensor(input.shape())
        model.backward(grad_output, grad_input)
        optimizer.step
