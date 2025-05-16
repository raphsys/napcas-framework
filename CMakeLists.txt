cmake_minimum_required(VERSION 3.19)
project(napcas LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(NAPCAS_BUILD_TESTS      "Build C++ unit tests"          ON)
option(NAPCAS_FETCH_DEPS       "Automatically download deps"   ON)
option(NAPCAS_ENABLE_CUDA      "Build CUDA backend (opt)"      OFF)

# ---------------------------------------------------------------------------
# Désactiver les tests d'Eigen (importés via FetchContent)
# ---------------------------------------------------------------------------
set(EIGEN_BUILD_TESTING   OFF CACHE BOOL "Disable Eigen test suite" FORCE)
set(EIGEN_BUILD_DOC       OFF CACHE BOOL "Disable Eigen documentation build" FORCE)
set(EIGEN_BUILD_EXAMPLES  OFF CACHE BOOL "Disable Eigen examples build" FORCE)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Dependencies : Eigen • pybind11 • Catch2
# ---------------------------------------------------------------------------
include(FetchContent)

if(NAPCAS_FETCH_DEPS)
    ## Eigen
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
    )
    ## pybind11
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.13.1
    )
    ## Catch2 (tests only)
    FetchContent_Declare(
        catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.2
    )
    FetchContent_MakeAvailable(eigen pybind11)
    
# ---------------------------------------------------------------------------
# Eigen alias (nécessaire uniquement si on a utilisé FetchContent)
# ---------------------------------------------------------------------------
# Eigen crée une interface library nommée 'eigen' ; on crée l'alias standard
# 'Eigen3::Eigen' pour que target_link_libraries fonctionne partout.

if(NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
endif()

    if(NAPCAS_BUILD_TESTS)
        FetchContent_MakeAvailable(catch2)
    endif()
else()
    find_package(Eigen3      REQUIRED CONFIG)
    find_package(pybind11   REQUIRED CONFIG)
    if(NAPCAS_BUILD_TESTS)
        find_package(Catch2 REQUIRED CONFIG)
    endif()
endif()

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
file(GLOB_RECURSE NAPCAS_HEADERS CONFIGURE_DEPENDS
     cpp/include/napcas/*.h)
file(GLOB_RECURSE NAPCAS_SOURCES CONFIGURE_DEPENDS
     cpp/src/*.cpp)
list(REMOVE_ITEM NAPCAS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/python_bindings.cpp)

# ---------------------------------------------------------------------------
# Core static / shared library
# ---------------------------------------------------------------------------
add_library(napcas_core ${NAPCAS_HEADERS} ${NAPCAS_SOURCES})
target_include_directories(napcas_core
    PUBLIC
        cpp/include
        ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(napcas_core PUBLIC Eigen3::Eigen)
target_compile_definitions(napcas_core PRIVATE EIGEN_MPL2_ONLY)
if (NAPCAS_ENABLE_CUDA)
    target_compile_definitions(napcas_core PUBLIC NAPCAS_WITH_CUDA)
    # Ajoutez ici vos liens CUDA si besoin (cublas, cuda_runtime, etc.)
endif()

# ---------------------------------------------------------------------------
# Python binding (_napcas)
# ---------------------------------------------------------------------------
pybind11_add_module(_napcas
    cpp/src/python_bindings.cpp
)
target_link_libraries(_napcas PRIVATE napcas_core)
target_include_directories(_napcas PRIVATE cpp/include)
set_target_properties(_napcas PROPERTIES
    OUTPUT_NAME "_napcas"
    BUILD_WITH_INSTALL_RPATH YES
)

# On force la sortie de la bibliothèque dans python/napcas
set_target_properties(_napcas PROPERTIES
	LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/napcas
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if (NAPCAS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install rules
# ---------------------------------------------------------------------------
install(TARGETS napcas_core _napcas
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY cpp/include/napcas DESTINATION include)


