from napcas import Linear, ReLU, MSELoss, SGD, DataLoader
import napcas

# Test de la classe Linear
linear = Linear(784, 10)
input_tensor = napcas.Tensor([1, 784], [0.0] * 784)  # 2D tensor with batch size 1
output_tensor = napcas.Tensor([1, 10])  # 2D tensor for output
linear.forward(input_tensor, output_tensor)
print("Output of Linear:", [output_tensor[i] for i in range(output_tensor.size())])

# Test de ReLU
relu = ReLU()
output_relu = napcas.Tensor(input_tensor.shape())
relu.forward(input_tensor, output_relu)
print("Output of ReLU:", [output_relu[i] for i in range(output_relu.size())])

# Test de MSELoss
mse_loss = MSELoss()
y_pred = napcas.Tensor([1, 10], [0.5] * 10)  # 2D tensor with batch size 1
y_true = napcas.Tensor([1, 10], [1.0] * 10)  # 2D tensor with batch size 1
loss = mse_loss.forward(y_pred, y_true)
grad = mse_loss.backward(y_pred, y_true)
print("MSE Loss:", loss)
print("Gradient of MSE Loss:", [grad[i] for i in range(grad.size())])

# Test de DataLoader
data_loader = DataLoader("dataset.csv", 64)
inputs, targets = data_loader.next()
print("Inputs shape:", inputs.shape())
print("Targets shape:", targets.shape())
