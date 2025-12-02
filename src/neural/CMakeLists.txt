# Neural network integration library
add_library(neural_integration
    network.cpp
    network_kernels.cu
    weight_loader.cpp
    config.cpp
    ../utils/debug_utils.cu
)

# Mark network.cpp as CUDA source since it includes tiny-cuda-nn headers with device code
set_source_files_properties(network.cpp PROPERTIES
    LANGUAGE CUDA
    COMPILE_FLAGS "--extended-lambda"
)

target_include_directories(neural_integration PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(neural_integration PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc
    TCNN::tcnn
)

set_target_properties(neural_integration PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
