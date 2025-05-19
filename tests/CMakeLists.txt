# tests/CMakeLists.txt
cmake_minimum_required(VERSION 3.14)
project(napcas-tests LANGUAGES CXX)

# Récupère le répertoire racine du projet
get_filename_component(NAPCAS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

# Sources du test
set(TEST_SOURCES
    cpp/test_tensor.cpp
    # On compile directement tensor.cpp avec le test
    ${NAPCAS_ROOT}/cpp/src/tensor.cpp
)

add_executable(test_tensor ${TEST_SOURCES})

# Inclure les headers du framework
target_include_directories(test_tensor PRIVATE
    ${NAPCAS_ROOT}/cpp/include
    /usr/include/eigen3            # n'oubliez pas d'installer libeigen3-dev
)

# Spécifie les standards et options
target_compile_features(test_tensor PRIVATE cxx_std_17)
target_compile_options(test_tensor PRIVATE -O2 -fPIC)

# Pas de linkage externe requis (tout est dans le binaire final)

