cmake_minimum_required(VERSION 3.14)
project(napcas)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compilation sécurisée
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fvisibility=hidden -fvisibility-inlines-hidden")

# Activer PYBIND11_FINDPYTHON pour éviter l'avertissement CMP0148
set(PYBIND11_FINDPYTHON ON)

# Trouver pybind11 et Python
find_package(pybind11 REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Development)

# Trouver Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Définir les sources de la bibliothèque C++
set(NAPCAS_SRC
    cpp/src/activation.cpp
    cpp/src/autograd.cpp
    cpp/src/conv2d.cpp
    cpp/src/data_loader.cpp
    cpp/src/linear.cpp
    cpp/src/loss.cpp
    cpp/src/napcas.cpp
    cpp/src/nncell.cpp
    cpp/src/optimizer.cpp
    cpp/src/tensor.cpp
)

# Vérifier que les fichiers source existent
foreach(src_file ${NAPCAS_SRC})
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${src_file}")
        message(WARNING "Source file ${src_file} not found")
    endif()
endforeach()

# Vérifier que python_bindings.cpp existe
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/python_bindings.cpp")
    message(FATAL_ERROR "Source file cpp/src/python_bindings.cpp not found")
endif()

# Ajouter le chemin du répertoire contenant la bibliothèque
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../build)

# Créer la bibliothèque statique
add_library(libnapcas STATIC ${NAPCAS_SRC})
target_include_directories(libnapcas PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include ${EIGEN3_INCLUDE_DIR})
target_link_libraries(libnapcas PUBLIC Eigen3::Eigen)

# Créer le module Python
pybind11_add_module(napcas MODULE cpp/src/python_bindings.cpp)
target_link_libraries(napcas PRIVATE libnapcas Python3::Python Eigen3::Eigen)

# Propriétés de visibilité
set_target_properties(napcas PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# Ajouter les tests si la variable BUILD_TESTS est définie
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
