# HIPRT host-side library
# Includes Orochi for cross-platform GPU abstraction

# Orochi and its dependencies (CUEW/HIPEW for dynamic loading)
set(OROCHI_SRC ${HIPRT_INCLUDE}/Orochi/Orochi.cpp)
set(CUEW_SRC ${HIPRT_INCLUDE}/contrib/cuew/src/cuew.cpp)
set(HIPEW_SRC ${HIPRT_INCLUDE}/contrib/hipew/src/hipew.cpp)

add_library(hiprt_host STATIC
    hiprt_context.cpp
    geometry_builder.cpp
    scene_builder.cpp
    kernel_compiler.cpp
    ${OROCHI_SRC}
    ${CUEW_SRC}
    ${HIPEW_SRC}
)

target_include_directories(hiprt_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HIPRT_INCLUDE}
    ${HIPRT_INCLUDE}/contrib/cuew/include
    ${HIPRT_INCLUDE}/contrib/hipew/include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Enable CUEW for NVIDIA GPU support through Orochi
target_compile_definitions(hiprt_host PUBLIC
    OROCHI_ENABLE_CUEW
)

target_link_libraries(hiprt_host PUBLIC
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
)

# Set CUDA separable compilation for device code in .cu files
set_target_properties(hiprt_host PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Test executable for HIPRT context
add_executable(test_hiprt test_hiprt.cpp)
target_link_libraries(test_hiprt
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_hiprt PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for geometry builder
add_executable(test_geometry test_geometry.cpp)
target_link_libraries(test_geometry
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_geometry PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for scene builder
add_executable(test_scene test_scene.cpp)
target_link_libraries(test_scene
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_scene PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for kernel compiler
add_executable(test_kernel test_kernel.cpp)
target_link_libraries(test_kernel
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_kernel PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for custom intersection
add_executable(test_custom_intersection test_custom_intersection.cpp)
target_link_libraries(test_custom_intersection
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_custom_intersection PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for neural kernel compilation
add_executable(test_neural_kernel test_neural_kernel.cpp)
target_link_libraries(test_neural_kernel
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_neural_kernel PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for wall intersection
add_executable(test_wall_intersection test_wall_intersection.cpp)
target_link_libraries(test_wall_intersection
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_wall_intersection PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for two planes
add_executable(test_two_planes test_two_planes.cpp)
target_link_libraries(test_two_planes
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_two_planes PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)

# Test executable for mixed geometry (triangles + custom AABB)
add_executable(test_mixed_geometry test_mixed_geometry.cpp)
target_link_libraries(test_mixed_geometry
    hiprt_host
    ${HIPRT_LIBRARY}
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)
set_target_properties(test_mixed_geometry PROPERTIES
    BUILD_RPATH ${HIPRT_LIB_DIR}
    INSTALL_RPATH ${HIPRT_LIB_DIR}
)
