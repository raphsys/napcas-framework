from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

__version__ = "0.1.0"

ext_modules = [
    Extension(
        '_napcas',
        ['../cpp/src/activation.cpp', 
        '../cpp/src/autograd.cpp', 
        '../cpp/src/conv2d.cpp', 
        '../cpp/src/data_loader.cpp', 
        '../cpp/src/linear.cpp', 
        '../cpp/src/loss.cpp', 
        '../cpp/src/nncell.cpp', 
        '../cpp/src/napcas.cpp', 
        '../cpp/src/napca_sim.cpp', 
        '../cpp/src/tensor.cpp', 
        '../cpp/src/python_bindings.cpp', 
        '../cpp/src/optimizer.cpp'],
        include_dirs=[
            '../cpp/include',
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '/usr/include/eigen3',  # Chemin d'inclusion pour Eigen
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-fvisibility=hidden'],
        extra_link_args=['-std=c++17'],
    ),
]

setup(
    name='napcas',
    version=__version__,
    author='Komla M. DOLEAGBENOU',
    author_email='mkomla.doleagbenou@gmail.com',
    description='A sample Python project',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
