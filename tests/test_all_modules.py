import napcas

# Test Linear
x = napcas.Tensor([1, 10], [0.5]*10)
linear = napcas.Linear(10, 5)
output = linear.forward(x)
print(output)

# Test Conv2d
# ...

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
