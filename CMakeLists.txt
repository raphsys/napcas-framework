cmake_minimum_required(VERSION 3.10)
project(napcas VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit QUIET)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Doxygen QUIET)

# Source files
set(SOURCES
    cpp/src/tensor.cpp
    cpp/src/napcas.cpp
    cpp/src/napca_sim.cpp
    cpp/src/nncell.cpp
    cpp/src/linear.cpp
    cpp/src/conv2d.cpp
    cpp/src/pooling.cpp
    cpp/src/attention.cpp
    cpp/src/activation.cpp
    cpp/src/loss.cpp
    cpp/src/optimizer.cpp
    cpp/src/data_loader.cpp
    cpp/src/autograd.cpp
    cpp/src/mlp.cpp
    cpp/src/rnn.cpp
    cpp/src/transformer.cpp
    cpp/src/gan.cpp
    cpp/src/lstm.cpp
    cpp/src/gru.cpp
    cpp/src/visualization.cpp
    cpp/src/gpu_utils.cpp
    cpp/src/python_bindings.cpp
)

# Build static library
add_library(napcas STATIC ${SOURCES})

# Set include directories for napcas
target_include_directories(napcas PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Link CUDA if available
if(CUDAToolkit_FOUND)
    add_definitions(-DUSE_CUDA)
    target_link_libraries(napcas PRIVATE CUDA::cudart CUDA::cublas)
endif()

# Link other dependencies
target_link_libraries(napcas PRIVATE Eigen3::Eigen pybind11::pybind11)

# Build test executable
add_executable(napcas_test cpp/tests/test_main.cpp)

# Set include directories for napcas_test
target_include_directories(napcas_test PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/cpp/include
)

# Link napcas_test with napcas
target_link_libraries(napcas_test PRIVATE napcas)

# Optional Doxygen documentation
if(DOXYGEN_FOUND)
    doxygen_add_docs(docs cpp/include cpp/src ALL)
else()
    message(WARNING "Doxygen not found. Documentation will not be generated.")
endif()

# Install targets
install(TARGETS napcas DESTINATION lib)
install(DIRECTORY cpp/include/ DESTINATION include/napcas)
