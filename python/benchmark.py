import time
import numpy as np
from napcas import Conv2d, Transformer, Tensor

def benchmark_conv2d():
    model = Conv2d(3, 16, 3, 1, 1)
    input_tensor = Tensor([2, 3, 64, 64], [1.0] * (2 * 3 * 64 * 64))
    output_tensor = Tensor([2, 16, 64, 64])

    start = time.time()
    for _ in range(10):
        model.forward(input_tensor, output_tensor)
    end = time.time()
    print(f"Conv2d forward (im2col): {(end - start) / 10 * 1000:.2f} ms per iteration")

def benchmark_transformer():
    model = Transformer(64, 8, 2, 128)
    input_tensor = Tensor([512, 2, 64], [1.0] * (512 * 2 * 64))
    output_tensor = Tensor([512, 2, 64])

    start = time.time()
    for _ in range(10):
        model.forward(input_tensor, output_tensor)
    end = time.time()
    print(f"Transformer forward (memory-efficient): {(end - start) / 10 * 1000:.2f} ms per iteration")

if __name__ == "__main__":
    print("Running benchmarks...")
    benchmark_conv2d()
    benchmark_transformer()
