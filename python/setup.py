from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'napcas',
        sources=['../cpp/src/python_bindings.cpp', '../cpp/src/activation.cpp', '../cpp/src/autograd.cpp', '../cpp/src/conv2d.cpp', '../cpp/src/data_loader.cpp', '../cpp/src/linear.cpp', '../cpp/src/loss.cpp', '../cpp/src/napcas.cpp', '../cpp/src/nncell.cpp', '../cpp/src/optimizer.cpp', '../cpp/src/tensor.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '../cpp/include'
        ],
        library_dirs=['../cpp/build'],
        libraries=['napcas'],
        language='c++',
        extra_compile_args=['-std=c++17'],
    )
]

setup(
    name='napcas',
    version='0.1',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6'],
)
