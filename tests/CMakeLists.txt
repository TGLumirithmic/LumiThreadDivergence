# Phase 1 test program
add_executable(test_network test_network.cu)

target_include_directories(test_network PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(test_network PRIVATE
    neural_integration
    CUDA::cudart
    TCNN::tcnn
)

set_target_properties(test_network PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Copy to output directory for convenience
install(TARGETS test_network DESTINATION bin)

# OptiX neural network test program
add_executable(test_optix_network test_optix_network.cu)

target_include_directories(test_optix_network PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/programs
    ${OptiX_INCLUDE}
)

target_link_libraries(test_optix_network PRIVATE
    neural_integration
    optix_host
    CUDA::cudart
)

set_target_properties(test_optix_network PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS test_optix_network DESTINATION bin)

# OptiX encoding test program (tests intermediate encodings)
add_executable(test_optix_encodings test_optix_encodings.cu)

target_include_directories(test_optix_encodings PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/programs
    ${OptiX_INCLUDE}
)

target_link_libraries(test_optix_encodings PRIVATE
    neural_integration
    optix_host
    CUDA::cudart
)

set_target_properties(test_optix_encodings PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS test_optix_encodings DESTINATION bin)
