# core.py — importe le sous‑module C++ et expose l’API
import importlib

_napcas = importlib.import_module("napcas._napcas")

Tensor     = _napcas.Tensor
Device     = _napcas.Device
DeviceType = _napcas.DeviceType
DType      = _napcas.DType

__all__ = ["Tensor", "Device", "DeviceType", "DType"]
