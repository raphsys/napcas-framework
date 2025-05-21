# cpp/CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(napcas CXX)

set(CMAKE_CXX_VISIBILITY_PRESET    default)
set(CMAKE_VISIBILITY_INLINES_HIDDEN OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

# Source files for the Python extension
set(SOURCES
    src/tensor.cpp
    src/module.cpp
    src/autograd.cpp
    src/grad_fn.cpp
    src/python_bindings.cpp
)

# Build the _napcas extension module
pybind11_add_module(_napcas MODULE ${SOURCES})


# Include headers from cpp/include
target_include_directories(_napcas PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against Eigen
target_link_libraries(_napcas PRIVATE
    Eigen3::Eigen
)

# Export every non‐static symbol so the dynamic linker can resolve Autograd::backward
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.13")
    target_link_options(_napcas PRIVATE "-Wl,--export-dynamic")
else()
    set_target_properties(_napcas PROPERTIES
        LINK_FLAGS "-Wl,--export-dynamic"
    )
endif()

# Ensure the .so lands in python/napcas so that `import napcas._napcas` works
set_target_properties(_napcas PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/napcas"
    OUTPUT_NAME "_napcas"
)
