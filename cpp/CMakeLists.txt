cmake_minimum_required(VERSION 3.14)
project(napcas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilation sécurisée
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -fvisibility-inlines-hidden")

# Trouver pybind11 et Python
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# Trouver Eigen
find_package(Eigen3 REQUIRED)

# Définir les sources de la bibliothèque C++
set(NAPCAS_SRC
    src/activation.cpp
    src/autograd.cpp
    src/conv2d.cpp
    src/data_loader.cpp
    src/linear.cpp
    src/loss.cpp
    src/napcas.cpp
    src/nncell.cpp
    src/optimizer.cpp
    src/tensor.cpp
)

# Créer la bibliothèque statique
add_library(libnapcas STATIC ${NAPCAS_SRC})
target_include_directories(libnapcas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(libnapcas PUBLIC Eigen3::Eigen)

# Créer le module Python
pybind11_add_module(napcas MODULE src/python_bindings.cpp)
target_link_libraries(napcas PRIVATE libnapcas Python3::Python Eigen3::Eigen)

# Propriétés de visibilité
set_target_properties(napcas PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)
