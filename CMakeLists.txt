# napcas-framework/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(napcas VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
find_package(CUDAToolkit QUIET)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
find_package(Doxygen QUIET)
find_package(HDF5 REQUIRED COMPONENTS C CXX)

include_directories(${HDF5_INCLUDE_DIRS})

# Ajoute le sous-dossier cpp
add_subdirectory(cpp)

# Génération de documentation
if(DOXYGEN_FOUND)
    doxygen_add_docs(docs cpp/include cpp/src ALL)
else()
    message(WARNING "Doxygen not found. Documentation will not be generated.")
endif()

# Installation (optionnelle)
install(DIRECTORY cpp/include/ DESTINATION include/napcas)

