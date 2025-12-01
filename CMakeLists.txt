cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(NeuralOptiXRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures (adjust based on your GPU)
# GTX 980 has compute capability 5.2
# NOTE: tiny-cuda-nn's FullyFusedMLP requires tensor cores (sm_70+)
# If using GTX 980, you need to use CutlassMLP instead
set(CMAKE_CUDA_ARCHITECTURES 52)

find_package(CUDAToolkit REQUIRED)

# tiny-cuda-nn (assumes it's in external/ directory or installed)
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/tiny-cuda-nn")
    add_subdirectory(external/tiny-cuda-nn)
else()
    message(STATUS "tiny-cuda-nn not found in external/. Assuming system install.")
    # For system install, we need to find it or create an interface target
    if(NOT TARGET tiny-cuda-nn)
        # Check if installed to system paths
        find_path(TCNN_INCLUDE_DIR tiny-cuda-nn/common.h
            PATHS
                /usr/local/include
                /opt/tiny-cuda-nn/install/include
                /opt/tiny-cuda-nn/include
            NO_DEFAULT_PATH
        )
        if(TCNN_INCLUDE_DIR)
            message(STATUS "Found tiny-cuda-nn headers: ${TCNN_INCLUDE_DIR}")

            # Find the library files
            find_library(TCNN_LIB tiny-cuda-nn
                PATHS /opt/tiny-cuda-nn/build
                NO_DEFAULT_PATH
            )
            find_library(TCNN_RESOURCES_LIB tiny-cuda-nn-resources
                PATHS /opt/tiny-cuda-nn/build
                NO_DEFAULT_PATH
            )
            find_library(FMT_LIB fmt
                PATHS /opt/tiny-cuda-nn/build/dependencies/fmt
                NO_DEFAULT_PATH
            )

            if(TCNN_LIB)
                message(STATUS "Found tiny-cuda-nn library: ${TCNN_LIB}")
                add_library(tiny-cuda-nn INTERFACE)
                target_include_directories(tiny-cuda-nn INTERFACE ${TCNN_INCLUDE_DIR})
                target_compile_features(tiny-cuda-nn INTERFACE cxx_std_17)

                # Add required compile definitions for tiny-cuda-nn
                # Note: TCNN_HALF_PRECISION must match how tiny-cuda-nn was built
                # For GTX 980 (sm_52), use float precision (TCNN_HALF_PRECISION=0)
                target_compile_definitions(tiny-cuda-nn INTERFACE
                    TCNN_HALF_PRECISION=0
                    TCNN_MIN_GPU_ARCH=52
                )

                # Include tiny-cuda-nn dependencies
                set(TCNN_DEPS_BASE /opt/tiny-cuda-nn/dependencies)
                target_include_directories(tiny-cuda-nn INTERFACE
                    ${TCNN_DEPS_BASE}/fmt/include
                    ${TCNN_DEPS_BASE}
                    ${TCNN_DEPS_BASE}/cutlass/include
                )

                # Link the libraries
                target_link_libraries(tiny-cuda-nn INTERFACE
                    ${TCNN_LIB}
                    ${TCNN_RESOURCES_LIB}
                    ${FMT_LIB}
                )
            else()
                message(FATAL_ERROR "tiny-cuda-nn library not found in build directory.")
            endif()
        else()
            message(FATAL_ERROR "tiny-cuda-nn headers not found. Please install tiny-cuda-nn.")
        endif()
    endif()
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Add source files
add_subdirectory(src/neural)
add_subdirectory(tests)
