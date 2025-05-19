# napcas/__init__.py

# Import the C++ extension submodule...
from ._napcas import Tensor, Device, DeviceType, DType

__all__ = ["Tensor", "Device", "DeviceType", "DType"]

