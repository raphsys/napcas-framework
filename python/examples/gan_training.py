from napcas import GAN, DataLoader, Tensor, Adam

def main():
    loader = DataLoader("data/synthetic.csv", batch_size=64, augment=True)
    model = GAN([100, 256, 784], [784, 256, 1])
    optimizer = Adam([model], lr=0.0002, beta1=0.5, beta2=0.999)

    for epoch in range(10):
        real_data, _ = loader.next()
        noise = Tensor([64, 100], [float(i % 100) for i in range(64 * 100)])
        gen_loss = Tensor([1])
        disc_loss = Tensor([1])
        model.train_step(real_data, noise, 0.0002, gen_loss, disc_loss)
        optimizer.step()
        print(f"Epoch {epoch+1}, Gen Loss: {gen_loss[0]}, Disc Loss: {disc_loss[0]}")

if __name__ == "__main__":
    main()
