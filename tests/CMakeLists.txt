cmake_minimum_required(VERSION 3.14)
project(napcas-tests LANGUAGES CXX)

find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# Répertoire racine du projet
get_filename_component(NAPCAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# 1) OBJECT-library compilant tout le cœur C++ (sans python_bindings)
add_library(napcas_core_objects OBJECT
    ${NAPCAS_ROOT}/cpp/src/tensor.cpp
    ${NAPCAS_ROOT}/cpp/src/module.cpp
    ${NAPCAS_ROOT}/cpp/src/autograd.cpp
    ${NAPCAS_ROOT}/cpp/src/grad_fn.cpp
    ${NAPCAS_ROOT}/cpp/src/architecture/linear.cpp
)
target_include_directories(napcas_core_objects PUBLIC
    ${NAPCAS_ROOT}/cpp/include
    ${EIGEN3_INCLUDE_DIR}
)

# 2) test_tensor
add_executable(test_tensor
    cpp/test_tensor.cpp
)
target_link_libraries(test_tensor PRIVATE
    napcas_core_objects
    GTest::gtest_main
    Threads::Threads
)
target_include_directories(test_tensor PRIVATE
    ${NAPCAS_ROOT}/cpp/include
    ${EIGEN3_INCLUDE_DIR}
)
add_test(NAME TensorTest COMMAND test_tensor)

# 3) test_linear
add_executable(test_linear
    architecture/test_linear.cpp
)
target_link_libraries(test_linear PRIVATE
    napcas_core_objects
    GTest::gtest_main
    Threads::Threads
)
target_include_directories(test_linear PRIVATE
    ${NAPCAS_ROOT}/cpp/include
)
add_test(NAME LinearTest COMMAND test_linear)

