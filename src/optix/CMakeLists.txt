# OptiX host code library
add_library(optix_host
    context.cpp
    pipeline.cpp
    geometry.cpp
    sbt.cpp
    neural_params.cpp
    optix_stubs.cpp
)

target_link_libraries(optix_host
    CUDA::cudart
    CUDA::cuda_driver
    ${CMAKE_DL_LIBS}
)

target_include_directories(optix_host PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)
