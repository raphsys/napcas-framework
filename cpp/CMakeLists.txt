# cpp/CMakeLists.txt — compile tout dans _napcas
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(SOURCES
    src/python_bindings.cpp
    src/tensor.cpp
)

pybind11_add_module(_napcas MODULE ${SOURCES})

target_include_directories(_napcas PRIVATE
    ${PROJECT_SOURCE_DIR}/cpp/include
)

target_link_libraries(_napcas PRIVATE
    Eigen3::Eigen
)

set_target_properties(_napcas PROPERTIES
    OUTPUT_NAME "napcas._napcas"
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
