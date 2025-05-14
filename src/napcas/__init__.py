"""
napcas – high-level Python API wrapping the C++ core (_napcas).
Provides Pythonic overloads and utilities on top of the raw bindings.
"""

from importlib import import_module as _imp
import numpy as _np

# Load the raw pybind11 extension
_bind = _imp("napcas._napcas")

# Expose all public names from _bind
for _name in dir(_bind):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_bind, _name)

# Save original Tensor class
_RawTensor = _bind.Tensor

# Override Tensor to accept nested lists or numpy arrays

def Tensor(arg=None, data=None):
    """
    Create a Tensor. Usage:
      Tensor(shape: list[int], data: list[float]=[])
      Tensor(nested_list: list or numpy array)
    """
    # Only arg provided and it's array-like → infer shape/data
    if data is None and arg is not None \
       and (isinstance(arg, (list, tuple, _np.ndarray))):
        arr = _np.asarray(arg, dtype=float)
        shape = list(arr.shape)
        flat = arr.ravel().tolist()
        return _RawTensor(shape, flat)
    # Fall back to original signature
    if arg is None:
        return _RawTensor()
    return _RawTensor(arg, data)  

# Public API
__all__ = [n for n in globals().keys() if not n.startswith('_')]

# Monkey-patch Autograd.zero_grad pour zéroter aussi data()
_orig_zero_grad = Autograd.zero_grad
def _patched_zero_grad(tensors):
    _orig_zero_grad(tensors)
    for t in tensors:
        t.fill(0.0)
# On remplace la méthode statique
Autograd.zero_grad = staticmethod(_patched_zero_grad)



