# OptiX host code library
add_library(optix_host
    context.cpp
    pipeline.cpp
    geometry.cpp
    sbt.cpp
)

target_link_libraries(optix_host
    CUDA::cudart
    CUDA::cuda_driver
)

target_include_directories(optix_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
