Installation
============

Prerequisites
-------------

- CMake >= 3.10
- Eigen3
- CUDA (optional for GPU support)
- Python >= 3.8
- Pybind11

Build Instructions
------------------

.. code-block:: bash

   mkdir build && cd build
   cmake ..
   make
   make install

Python Package
--------------

.. code-block:: bash

   cd python
   pip install .
