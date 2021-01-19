
cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)
project(pressiotools C CXX)

set(CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/cmake;${PROJECT_SOURCE_DIR}/cmake/tpls)

# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.
    Please make a new directory (called a build directory) and run CMake from there.
    You may need to remove CMakeCache.txt.")
endif()

# default to relase if build type is empty
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
# if(NOT cmake_build_type_tolower STREQUAL "debug"
#     AND NOT cmake_build_type_tolower STREQUAL "release")
#   message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\".
#     Allowed values are Debug, Release (case-insensitive).")
# endif()
### default to release if build type is empty ###
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "${CMAKE_BUILD_TYPE} is not specified, defaulting to Release.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

#=====================================================================
# we need c++11
#=====================================================================
# commands to test if compiler supports C++11
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORT_CPP11)
if(NOT COMPILER_SUPPORT_CPP11)
  message(FATAL_ERROR "Compiler does not support -std=c++11. This is required.")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#=====================================================================
include(tplTrilinos)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pybind11)
pybind11_add_module(pressiotools src/main_binder.cc)
target_compile_definitions(pressiotools PRIVATE VERSION_IN=${VERSION_INFO})
target_include_directories(pressiotools PRIVATE ${TRILINOS_ROOT}/include)
target_link_directories(pressiotools PRIVATE ${TRILINOS_ROOT}/lib)
target_link_libraries(pressiotools PRIVATE ${TRILINOS_LIB_NAMES})

#add_subdirectory(tests)
# link_libraries(${CMAKE_DL_LIBS})
# include(GNUInstallDirs)
# include(tplEigen)
# include(tplMPI)
# include(tplBlas)
# include(tplLapack)
# link_libraries(${CMAKE_DL_LIBS})
