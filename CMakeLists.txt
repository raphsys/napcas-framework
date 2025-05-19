cmake_minimum_required(VERSION 3.14)
project(_napcas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(cpp)
add_subdirectory(tests)
