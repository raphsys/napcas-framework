# python/test_napcas.py
from napcas import Linear, ReLU, MSELoss, SGD, DataLoader

# Test de la classe Linear
linear = Linear(784, 10)
input_tensor = [0.0] * 784
output_tensor = linear.forward(input_tensor)

print("Output of Linear:", output_tensor)

# Test de ReLU
relu = ReLU()
output_relu = relu.forward(input_tensor)

print("Output of ReLU:", output_relu)

# Test de MSELoss
mse_loss = MSELoss()
y_pred = [0.5] * 10
y_true = [1.0] * 10
loss = mse_loss.forward(y_pred, y_true)
grad = mse_loss.backward(y_pred, y_true)

print("MSE Loss:", loss)
print("Gradient of MSE Loss:", grad)

# Test de DataLoader
data_loader = DataLoader("dataset", 64)
for batch in data_loader:
    inputs, targets = batch
    print("Inputs shape:", inputs.shape)
    print("Targets shape:", targets.shape)
    break
