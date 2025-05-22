cmake_minimum_required(VERSION 3.14)
project(_napcas LANGUAGES CXX)

add_subdirectory(cpp)
enable_testing()
add_subdirectory(tests)
