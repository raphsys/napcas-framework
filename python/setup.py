from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import os

cpp_src = "../cpp/src/"
source_files = [os.path.join(cpp_src, f) for f in [
    "python_bindings.cpp",
    "linear.cpp",
    "conv2d.cpp",
    "activation.cpp",
    "loss.cpp",
    "optimizer.cpp",
    "data_loader.cpp",
    "autograd.cpp",
    "nncell.cpp",
    "tensor.cpp",
    "napcas.cpp",
]]

ext_modules = [
    Extension(
        "_napcas",
        source_files,
        include_dirs=[
            "../cpp/include",
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

setup(
    name="napcas",
    version="0.1.0",
    packages=["napcas"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)

