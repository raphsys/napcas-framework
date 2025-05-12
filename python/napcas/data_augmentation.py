import numpy as np

def augment_data(tensor):
    """Applies Gaussian noise to the tensor."""
    noise = np.random.normal(0, 0.01, tensor.shape())
    tensor.data()[:] = tensor.data() + noise.flatten()
    return tensor
