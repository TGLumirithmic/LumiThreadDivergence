cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NeuralOptiXRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GTX 980 = SM 52
set(CMAKE_CUDA_ARCHITECTURES 52)

find_package(CUDAToolkit REQUIRED)

# ============================================================
# tiny-cuda-nn fixed location at /opt/tiny-cuda-nn
# ============================================================
set(TCNN_PREFIX /opt/tiny-cuda-nn/)

set(TCNN_INCLUDE_DIR     ${TCNN_PREFIX}/include)
set(TCNN_DEPS_DIR        ${TCNN_PREFIX}/dependencies)
set(TCNN_BUILD_DIR       ${TCNN_PREFIX}/build)

# Library files
set(TCNN_LIB             ${TCNN_BUILD_DIR}/libtiny-cuda-nn.a)
set(TCNN_RESOURCES_LIB   ${TCNN_BUILD_DIR}/libtiny-cuda-nn-resources.a)
set(FMT_LIB              ${TCNN_BUILD_DIR}/dependencies/fmt/libfmt.a)

# Basic sanity checking
if(NOT EXISTS "${TCNN_LIB}")
    message(FATAL_ERROR "tiny-cuda-nn library not found at: ${TCNN_LIB}")
endif()

if(NOT EXISTS "${TCNN_INCLUDE_DIR}/tiny-cuda-nn/common.h")
    message(FATAL_ERROR "tiny-cuda-nn headers not found at: ${TCNN_INCLUDE_DIR}")
endif()

# ------------------------------------------------------------
# Create imported target TCNN::tcnn
# ------------------------------------------------------------
add_library(TCNN::tcnn INTERFACE IMPORTED)

target_include_directories(TCNN::tcnn
    INTERFACE
        ${TCNN_INCLUDE_DIR}
        ${TCNN_DEPS_DIR}
        ${TCNN_DEPS_DIR}/fmt/include
        ${TCNN_DEPS_DIR}/cutlass/include
)

# Add required compile definitions for tiny-cuda-nn
# Note: TCNN_HALF_PRECISION must match how tiny-cuda-nn was built
# For GTX 980 (sm_52), use float precision (TCNN_HALF_PRECISION=0)
target_compile_definitions(TCNN::tcnn INTERFACE
    TCNN_HALF_PRECISION=0               # matches how you built it
    TCNN_MIN_GPU_ARCH=52                # GTX 980
)
target_compile_features(TCNN::tcnn INTERFACE cxx_std_17)

# Link actual libraries
target_link_libraries(TCNN::tcnn INTERFACE
    ${TCNN_LIB}
    ${TCNN_RESOURCES_LIB}
    ${FMT_LIB}
)



message(STATUS "tiny-cuda-nn loaded from ${TCNN_PREFIX}")

# ============================================================
# Build output dirs
# ============================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================
# Global include dirs (not for TCNN)
# ============================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ============================================================
# Subdirectories
# ============================================================
add_subdirectory(src/neural)
add_subdirectory(tests)
