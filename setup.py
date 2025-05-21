# setup.py (at napcas-framework/setup.py)

from setuptools import setup, find_packages

setup(
    name="napcas",
    version="0.1.0",
    # Tell setuptools that our Python code lives under the "python/" dir
    package_dir={"": "python"},
    # Find all packages (i.e. python/napcas)
    packages=find_packages(where="python"),
    # Include the .so built by CMake as package data
    package_data={
        "napcas": ["_napcas*.so"],
    },
    include_package_data=True,
    zip_safe=False,
)

