cmake_minimum_required(VERSION 3.14)
project(napcas VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-fvisibility=hidden)

find_package(Eigen3 REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)

set(NAPCAS_SOURCES
    src/tensor.cpp
    src/napcas.cpp
    src/napca_sim.cpp
    src/nncell.cpp
    src/linear.cpp
    src/conv2d.cpp
    src/pooling.cpp
    src/attention.cpp
    src/activation.cpp
    src/loss.cpp
    src/optimizer.cpp
    src/data_loader.cpp
    src/autograd.cpp
    src/mlp.cpp
    src/rnn.cpp
    src/transformer.cpp
    src/gan.cpp
    src/lstm.cpp
    src/gru.cpp
    src/visualization.cpp
    src/gpu_utils.cpp
)

# Bibliothèque stb compilée séparément avec -fPIC
add_library(stb_image_impl STATIC src/stb_image_impl.cpp)
set_target_properties(stb_image_impl PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Bibliothèque principale
add_library(napcas_lib STATIC ${NAPCAS_SOURCES})
target_compile_options(napcas_lib PRIVATE -fPIC)

target_include_directories(napcas_lib PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(napcas_lib PUBLIC
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
    stb_image_impl
)

pybind11_add_module(_napcas MODULE src/python_bindings.cpp)

set_target_properties(_napcas PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/napcas"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/napcas")   # Windows

# Où sera créée la lib lors de l'étape install
install(TARGETS _napcas
        LIBRARY DESTINATION napcas             # Unix (.so)
        RUNTIME DESTINATION napcas             # Windows (.pyd/.dll)
        ARCHIVE DESTINATION napcas)            # Static lib (rare)
        
set_target_properties(_napcas PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src/napcas"
        PREFIX ""
)
target_link_libraries(_napcas PRIVATE napcas_lib ${HDF5_LIBRARIES} Eigen3::Eigen)
target_include_directories(_napcas PRIVATE ${PROJECT_SOURCE_DIR}/include ${HDF5_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
set_target_properties(_napcas PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)

option(BUILD_TESTS "Build C++ unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(napcas_test tests/test_main.cpp)
    target_include_directories(napcas_test PRIVATE ${PROJECT_SOURCE_DIR}/include ${HDF5_INCLUDE_DIRS})
    target_link_libraries(napcas_test PRIVATE napcas_lib)
    add_test(NAME napcas_cpp_tests COMMAND napcas_test)
endif()

install(TARGETS napcas_lib _napcas LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/napcas)

