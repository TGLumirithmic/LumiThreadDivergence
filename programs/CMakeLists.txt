# Compile OptiX device programs to PTX

# Function to compile CUDA to PTX
function(add_ptx_target target_name cuda_file)
    set(output_file "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${target_name}.ptx")

    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_CUDA_COMPILER}
            -ptx
            -arch=sm_${CMAKE_CUDA_ARCHITECTURES}
            -dc
            --use_fast_math
            -lineinfo
            -I${OptiX_INCLUDE}
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/..
            -I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
            ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_file}
            -o ${output_file}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${cuda_file}
                ${CMAKE_CURRENT_SOURCE_DIR}/common.h
        COMMENT "Compiling ${cuda_file} to PTX"
        VERBATIM
    )

    add_custom_target(${target_name} ALL DEPENDS ${output_file})
endfunction()

# Compile each program to PTX
add_ptx_target(raygen raygen.cu)
add_ptx_target(miss miss.cu)
add_ptx_target(neural neural_programs.cu)
