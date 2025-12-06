cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
project(NeuralOptiXRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Debug build flags
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -lineinfo")

# # GPU Architecture Configuration
# # For GTX 980: set to 52, For RTX 2080 Ti: 75, For RTX 3090/4090: 86
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# endif()
set(CMAKE_CUDA_ARCHITECTURES 75)
set(TCNN_HALF_PRECISION 1)

# # Half precision configuration (must match how tiny-cuda-nn was built)
# # For sm_52 (GTX 980): use 0 (OFF)
# # For sm_75+ (RTX 2080 Ti and newer): can use 1 (ON)
# if(NOT DEFINED TCNN_HALF_PRECISION)
#     if(CMAKE_CUDA_ARCHITECTURES GREATER_EQUAL 70)
#     else()
#         set(TCNN_HALF_PRECISION 0)
#     endif()
# endif()

find_package(CUDAToolkit REQUIRED)

# ============================================================
# yaml-cpp integration (for Phase 4 scene loading)
# ============================================================
find_package(yaml-cpp REQUIRED)
message(STATUS "yaml-cpp found and available")

# ============================================================
# tinyobjloader integration (for mesh loading from OBJ files)
# ============================================================
include(FetchContent)
FetchContent_Declare(
    tinyobjloader
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG v2.0.0rc13
)
set(TINYOBJLOADER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TINYOBJLOADER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tinyobjloader)
message(STATUS "tinyobjloader fetched and made available")

# ============================================================
# OptiX SDK integration
# ============================================================
set(OptiX_INSTALL_DIR /opt/optix)
set(OptiX_INCLUDE ${OptiX_INSTALL_DIR}/include)

if(NOT EXISTS "${OptiX_INCLUDE}/optix.h")
    message(FATAL_ERROR "OptiX SDK not found at: ${OptiX_INSTALL_DIR}")
endif()

message(STATUS "OptiX SDK found at ${OptiX_INSTALL_DIR}")

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
target_compile_definitions(TCNN::tcnn INTERFACE
    TCNN_HALF_PRECISION=${TCNN_HALF_PRECISION}
    TCNN_MIN_GPU_ARCH=${CMAKE_CUDA_ARCHITECTURES}
)
target_compile_features(TCNN::tcnn INTERFACE cxx_std_17)

# Link actual libraries
target_link_libraries(TCNN::tcnn INTERFACE
    ${TCNN_LIB}
    ${TCNN_RESOURCES_LIB}
    ${FMT_LIB}
)



message(STATUS "tiny-cuda-nn loaded from ${TCNN_PREFIX}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TCNN Half Precision: ${TCNN_HALF_PRECISION}")

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
    ${CMAKE_CURRENT_SOURCE_DIR}/programs
    ${OptiX_INCLUDE}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# ============================================================
# Subdirectories
# ============================================================
add_subdirectory(src/neural)
add_subdirectory(src/optix)
add_subdirectory(programs)
add_subdirectory(tests)

# # ============================================================
# # Main renderer executable
# # ============================================================
add_executable(renderer src/main.cpp)
target_include_directories(renderer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(renderer
    neural_integration
    optix_host
    yaml-cpp
    tinyobjloader
    CUDA::nvrtc
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)

target_link_options(renderer PRIVATE "-Wl,--no-as-needed")

# Make sure PTX files are compiled before building renderer
add_dependencies(renderer raygen miss neural triangle)
