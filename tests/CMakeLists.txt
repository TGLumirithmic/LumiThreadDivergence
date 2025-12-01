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
