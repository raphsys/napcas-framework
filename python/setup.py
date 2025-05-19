# python/setup.py
from setuptools import setup, find_packages

setup(
    name="napcas",
    version="0.1.0",
    description="NAPCAS core library (Tensor, Module) with C++ backend",
    packages=find_packages(),      # trouvera napcas/
    include_package_data=True,
    zip_safe=False,
)

