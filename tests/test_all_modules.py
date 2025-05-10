import napcas

# Test Linear
x = napcas.Tensor([1, 10], [0.5]*10)
linear = napcas.Linear(10, 5)
output = linear.forward(x)
print(output)

# Test Conv2d
conv = napcas.Conv2d(1, 16, 3)
input_tensor = napcas.Tensor([1, 1, 28, 28], [0.5] * 784)
output_tensor = napcas.Tensor([1, 16, 26, 26])
conv.forward(input_tensor, output_tensor)
print(output_tensor)

# Test Activation
relu = napcas.ReLU()
output = relu.forward(x)
print(output)

# Test Loss
loss = napcas.MSELoss()
loss_value = loss.forward(x, x)
print(loss_value)

# Test Optimizer
opt = napcas.SGD([linear])
opt.step()
