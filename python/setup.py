from setuptools import setup, Extension
import pybind11

# Spécifiez explicitement les fichiers sources
ext_modules = [
    Extension(
        'napcas',
        sources=['../cpp/src/python_bindings.cpp'],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            '../cpp/include'
        ],
        language='c++',
        extra_compile_args=['-std=c++17'],
    )
]

setup(
    name='napcas',
    version='0.1',
    ext_modules=ext_modules,
)
