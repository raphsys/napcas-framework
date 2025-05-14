cmake_minimum_required(VERSION 3.14)
project(napcas VERSION 1.0.0 LANGUAGES CXX)

# --------------------------------------------------
# 1) Paramètres généraux
# --------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Flags pour PIC et visibilité
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -fvisibility-inlines-hidden"
)

# --------------------------------------------------
# 2) Dépendances externes
# --------------------------------------------------
# Pybind11 / Python
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# Eigen3
find_package(Eigen3 REQUIRED)

# HDF5 (C++ API)
find_package(HDF5 REQUIRED COMPONENTS CXX)
# HDF5_LIBRARIES et HDF5_INCLUDE_DIRS sont définis par find_package(HDF5)
include_directories(${HDF5_INCLUDE_DIRS})

# --------------------------------------------------
# 3) Sources C++ du framework
# --------------------------------------------------
set(NAPCAS_SRC
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
)

# Vérification (facultative)
foreach(src ${NAPCAS_SRC})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        message(WARNING "Source file not found: ${src}")
    endif()
endforeach()

# --------------------------------------------------
# 4) Bibliothèque statique napcas_lib
# --------------------------------------------------
add_library(napcas_lib STATIC ${NAPCAS_SRC})

# Include pour Eigen, HDF5, STB_IMAGE (placé dans cpp/include) et nos headers
target_include_directories(napcas_lib PUBLIC
    ${EIGEN3_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

# Link des dépendances
target_link_libraries(napcas_lib PUBLIC
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
)

# --------------------------------------------------
# 5) Module Python (_napcas) via Pybind11
# --------------------------------------------------
pybind11_add_module(_napcas MODULE cpp/src/python_bindings.cpp)

target_link_libraries(_napcas PRIVATE
    napcas_lib
    Python3::Python
    Eigen3::Eigen
    ${HDF5_LIBRARIES}
)

# Masquer les symboles inutiles
set_target_properties(_napcas PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# --------------------------------------------------
# 6) Tests C++ (optionnel)
# --------------------------------------------------
option(BUILD_TESTS "Build C++ unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(napcas_test cpp/tests/test_main.cpp)
    target_include_directories(napcas_test PRIVATE
        ${EIGEN3_INCLUDE_DIR}
        ${HDF5_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
    )
    target_link_libraries(napcas_test PRIVATE
        napcas_lib
    )
    add_test(NAME run_napcas_test COMMAND napcas_test)
endif()

# --------------------------------------------------
# 7) Installation (facultative)
# --------------------------------------------------
install(TARGETS napcas_lib DESTINATION lib)
install(TARGETS _napcas DESTINATION lib)
install(DIRECTORY cpp/include/ DESTINATION include/napcas)

