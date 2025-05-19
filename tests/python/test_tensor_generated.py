import napcas

def test_tensor_addition():
    a = napcas.Tensor.ones([2, 2])
    b = napcas.Tensor.ones([2, 2])
    c = a + b
    assert c.shape() == [2, 2]
