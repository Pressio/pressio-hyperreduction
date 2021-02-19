
cmake_minimum_required(VERSION 3.13.0 FATAL_ERROR)
project(pressiotools CXX)

# add cmake utilities
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake;${PROJECT_SOURCE_DIR}/cmake/tpls)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt.")
endif()

# default to release if build type is empty ###
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "${CMAKE_BUILD_TYPE} is not specified, defaulting to Release.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# we need c++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORT_CPP11)
if(NOT COMPILER_SUPPORT_CPP11)
  message(FATAL_ERROR "Compiler does not support -std=c++11. This is required.")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#=====================================================================
set(modulename _linalg)
add_subdirectory            (${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
pybind11_add_module         (${modulename} src/main_binder.cc)
target_compile_definitions  (${modulename} PRIVATE MODNAME=${modulename})

if (NOT ${TRILINOS_ROOT} STREQUAL "")
  include(tplTrilinos)
  target_include_directories(${modulename} PRIVATE ${TRILINOS_ROOT}/include)
  target_link_directories   (${modulename} PRIVATE ${TRILINOS_ROOT}/lib)
  target_link_libraries     (${modulename} PRIVATE ${TRILINOS_LIB_NAMES})
  target_compile_definitions(${modulename} PRIVATE PRESSIOTOOLS_ENABLE_TPL_TRILINOS)
  target_compile_definitions(${modulename} PRIVATE PRESSIOTOOLS_ENABLE_TPL_MPI)
endif()
