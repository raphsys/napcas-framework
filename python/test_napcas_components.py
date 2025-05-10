import pytest
import napcas

# Fonction pour vérifier les gradients par différences finies
def check_gradients(module, input_tensor, output_tensor, grad_output, epsilon=1e-5):
    module.forward(input_tensor, output_tensor)
    module.backward(grad_output, input_tensor)
    grad_weights = module.get_grad_weights().data()
    
    for i in range(min(10, grad_weights.size())):  # Limiter pour accélérer
        original_weight = module.get_weights()[i]
        module.get_weights()[i] = original_weight + epsilon
        module.forward(input_tensor, output_tensor)
        loss_plus = output_tensor.data().sum()
        module.get_weights()[i] = original_weight - epsilon
        module.forward(input_tensor, output_tensor)
        loss_minus = output_tensor.data().sum()
        module.get_weights()[i] = original_weight
        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
        assert pytest.approx(grad_weights[i], rel=1e-3) == numerical_grad

def test_linear():
    linear = napcas.Linear(10, 5)
    input_tensor = napcas.Tensor([1, 10], [0.5] * 10)
    output_tensor = napcas.Tensor([1, 5])
    grad_output = napcas.Tensor([1, 5], [1.0] * 5)
    
    linear.forward(input_tensor, output_tensor)
    assert output_tensor.shape() == [1, 5]
    check_gradients(linear, input_tensor, output_tensor, grad_output)

def test_conv2d():
    conv = napcas.Conv2d(1, 16, 3)
    input_tensor = napcas.Tensor([1, 1, 28, 28], [0.5] * 784)
    output_tensor = napcas.Tensor([1, 16, 26, 26])
    grad_output = napcas.Tensor([1, 16, 26, 26], [1.0] * (16 * 26 * 26))
    
    conv.forward(input_tensor, output_tensor)
    assert output_tensor.shape() == [1, 16, 26, 26]
    check_gradients(conv, input_tensor, output_tensor, grad_output)

def test_conv2d_performance():
    conv = napcas.Conv2d(1, 16, 3)
    input_tensor = napcas.Tensor([1, 1, 28, 28], [0.5] * 784)
    output_tensor = napcas.Tensor([1, 16, 26, 26])
    grad_output = napcas.Tensor([1, 16, 26, 26], [1.0] * (16 * 26 * 26))
    grad_input = napcas.Tensor([1, 1, 28, 28])
    
    # Mesurer le temps de la rétropropagation
    start_time = time.time()
    for _ in range(100):  # Répéter pour une mesure fiable
        conv.forward(input_tensor, output_tensor)
        conv.backward(grad_output, grad_input)
    end_time = time.time()
    
    print(f"Conv2d backward time (100 iterations): {(end_time - start_time):.4f} seconds")
    assert end_time - start_time < 1.0, "Backward propagation too slow"

def test_relu():
    relu = napcas.ReLU()
    input_tensor = napcas.Tensor([1, 5], [-1.0, -0.5, 0.0, 0.5, 1.0])
    output_tensor = napcas.Tensor([1, 5])
    
    relu.forward(input_tensor, output_tensor)
    expected = [0.0, 0.0, 0.0, 0.5, 1.0]
    for i in range(5):
        assert pytest.approx(output_tensor[i], rel=1e-5) == expected[i]

def test_mse_loss():
    mse = napcas.MSELoss()
    y_pred = napcas.Tensor([1, 5], [0.5, 0.4, 0.3, 0.2, 0.1])
    y_true = napcas.Tensor([1, 5], [1.0, 0.8, 0.6, 0.4, 0.2])
    
    loss = mse.forward(y_pred, y_true)
    expected_loss = sum((p - t) ** 2 for p, t in zip(y_pred.data(), y_true.data())) / 5
    assert pytest.approx(loss, rel=1e-5) == expected_loss

def test_cross_entropy_loss():
    ce = napcas.CrossEntropyLoss()
    y_pred = napcas.Tensor([1, 3], [0.1, 0.2, 0.7])
    y_true = napcas.Tensor([1, 3], [0.0, 0.0, 1.0])
    
    loss = ce.forward(y_pred, y_true)
    softmax = [math.exp(y_pred[i]) / sum(math.exp(y_pred[j]) for j in range(3)) for i in range(3)]
    expected_loss = -math.log(softmax[2])
    assert pytest.approx(loss, rel=1e-5) == expected_loss

def test_sgd():
    linear = napcas.Linear(10, 5)
    sgd = napcas.SGD([linear], lr=0.01)
    weights_before = linear.get_weights().data().copy()
    
    input_tensor = napcas.Tensor([1, 10], [0.5] * 10)
    output_tensor = napcas.Tensor([1, 5])
    grad_output = napcas.Tensor([1, 5], [1.0] * 5)
    
    linear.forward(input_tensor, output_tensor)
    linear.backward(grad_output, input_tensor)
    sgd.step()
    
    weights_after = linear.get_weights().data()
    for i in range(len(weights_before)):
        assert weights_before[i] != weights_after[i], "Weights not updated"

def test_adam():
    linear = napcas.Linear(10, 5)
    adam = napcas.Adam([linear], lr=0.001)
    weights_before = linear.get_weights().data().copy()
    
    input_tensor = napcas.Tensor([1, 10], [0.5] * 10)
    output_tensor = napcas.Tensor([1, 5])
    grad_output = napcas.Tensor([1, 5], [1.0] * 5)
    
    linear.forward(input_tensor, output_tensor)
    linear.backward(grad_output, input_tensor)
    adam.step()
    
    weights_after = linear.get_weights().data()
    for i in range(len(weights_before)):
        assert weights_before[i] != weights_after[i], "Weights not updated"
