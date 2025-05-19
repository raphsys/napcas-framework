# setup.py
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

here = os.path.abspath(os.path.dirname(__file__))
cpp_src = os.path.join(here, "cpp", "src")
cpp_inc = os.path.join(here, "cpp", "include")
eigen_inc = "/usr/include/eigen3"

ext_modules = [
    Pybind11Extension(
        "napcas._napcas",
        sources=[
            os.path.join(cpp_src, "python_bindings.cpp"),
            os.path.join(cpp_src, "tensor.cpp"),
        ],
        include_dirs=[cpp_inc, eigen_inc],
        language="c++",
        extra_compile_args=["-std=c++17","-O3","-fPIC"],
    ),
]

setup(
    name="napcas",
    version="0.1.0",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
