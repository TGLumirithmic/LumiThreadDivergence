# CUDA Debugging Guide

This document explains how to debug CUDA kernels step-by-step to track down NaN values and other numerical issues.

## Debug Tools Available

### 1. Debug Utilities (`src/utils/debug_utils.h`)

The debug utilities provide several functions to inspect CUDA buffers:

```cpp
#include "../utils/debug_utils.h"

// Print statistics (min, max, mean, NaN count, Inf count, first N values)
debug_utils::print_buffer_stats(d_buffer, size, "Buffer Name", max_print=10);

// Check for NaN/Inf values only
debug_utils::check_for_nan_inf(d_buffer, size, "Buffer Name");

// Print first N values
debug_utils::print_buffer_values(d_buffer, size, "Buffer Name", max_print=10);
```

### 2. Build with Debug Symbols

The project is configured to build with debug symbols:

```cmake
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G -O0 -lineinfo")
```

This enables:
- `-g`: Host code debug symbols
- `-G`: Device code debug symbols (enables cuda-gdb)
- `-O0`: No optimization for easier debugging
- `-lineinfo`: Line number information in kernels

## Current Debug Checkpoints in Inference

The `network.cpp::inference()` function has debug checkpoints at each step:

### STEP 0: Input Data
- Checks input positions (batch_size * 3)
- Checks input directions (batch_size * 3)
- Verifies no NaN/Inf in inputs

### STEP 0b: Encoding Weights
- Reports number of parameters in position encoder
- (Detailed weight checking disabled due to half-precision complexity)

### STEP 1: Position Encoding
- Checks encoded positions after hash grid lookup
- **CRITICAL**: This is where NaN first appears in your case

### STEP 2a: Padded Directions
- Checks direction input after padding to multiple of 16
- **CRITICAL**: Values are corrupted here (should be -1 to 1, but are ~1e-12)

### STEP 2b: Direction Encoding
- Checks direction encoding output
- **ALL NaN** due to corrupted input

### STEP 3: Concatenated Features
- Checks position + direction concatenation
- Inherits NaN from both sources

### STEP 4a/b/c: Decoder Outputs
- Checks visibility, normal, and depth outputs
- All NaN due to NaN inputs

## Interpreting the Output

### Example Output Analysis

```
>>> STEP 1 Result: Checking encoded positions <<<
[DEBUG] Encoded positions (size=8192)
  NaN count: 21              ← 21 NaN values found!
  Inf count: 0
  Zero count: 7804           ← Most values are zero
  First 10 values: 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
```

This tells you:
- **WHERE**: Position encoding (Step 1)
- **WHAT**: 21 NaN values out of 8192 total
- **MAGNITUDE**: 95% of values are zero, which is suspicious

### Example of Corrupted Data

```
>>> STEP 2a Result: Checking padded directions <<<
[DEBUG] Padded directions (size=4096)
  Min: -0.00732945          ← Should be around -1
  Max: 0.00780155           ← Should be around +1
  First 10 values: -1.91427e-12, 2.14034e-41, 0, 0, ...
```

This indicates a type conversion or memory corruption issue.

## Next Steps for Debugging

Based on the output from your inference, here are the issues to investigate:

### Issue 1: Position Encoding NaN

**Location**: After `encoding->inference()` in [network.cpp:360](../src/neural/network.cpp#L360)

**Possible Causes**:
1. Hash grid weights contain NaN
2. Hash table lookup is accessing invalid memory
3. Interpolation is dividing by zero
4. Weight loading corrupted the encoding parameters

**How to Debug**:
```cpp
// Add before encoding->inference():
// 1. Check if encoding weights are valid
auto* params = encoding->params();
// Copy first 100 params to host and print

// 2. Check hash indices being computed
// Add debug kernel to print hash values for first few samples

// 3. Check interpolation weights
// Add debug output in hash grid kernel
```

### Issue 2: Direction Padding Corruption

**Location**: After `pad_direction_kernel_float` in [network.cpp:390](../src/neural/network.cpp#L390)

**Possible Causes**:
1. Half-precision conversion is wrong (but TCNN_HALF_PRECISION=1, so should use half)
2. Kernel is reading from wrong memory location
3. Type mismatch between kernel and buffer allocation

**How to Debug**:
```cpp
// Add debug output in the kernel itself:
__global__ void pad_direction_kernel_float(...) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Input dir[0]: %f, %f, %f\n",
               input[0], input[1], input[2]);
        printf("Output dir[0]: %f, %f, %f\n",
               output[0], output[1], output[2]);
    }
    // ... rest of kernel
}
```

## Using cuda-gdb

With debug symbols enabled, you can use cuda-gdb:

```bash
cuda-gdb ./bin/test_network
(cuda-gdb) run ../data/models/weights.bin ../data/test/predictions.bin ../output
# When it stops at NaN:
(cuda-gdb) cuda thread
(cuda-gdb) cuda kernel
(cuda-gdb) print variable_name
```

## Useful CUDA Debugging Commands

```bash
# Check for CUDA errors
nvidia-smi

# Memory check with cuda-memcheck
cuda-memcheck ./bin/test_network args...

# Race condition detection
cuda-memcheck --tool racecheck ./bin/test_network args...

# Memory leak detection
cuda-memcheck --tool memcheck --leak-check full ./bin/test_network args...
```

## Adding New Debug Checkpoints

To add a new checkpoint in the inference pipeline:

```cpp
// After any CUDA operation
CUDA_CHECK(cudaDeviceSynchronize());

// Check the output buffer
debug_utils::print_buffer_stats(d_output, size, "Operation Name");
debug_utils::check_for_nan_inf(d_output, size, "Operation Name");
```

## Recommended Investigation Order

Based on the current output:

1. ✅ **Start**: Input data is clean
2. ❌ **First failure**: Position encoding (STEP 1) - **START HERE**
   - Check encoding weights for NaN
   - Inspect hash grid lookup logic
   - Verify interpolation doesn't have division by zero
3. ❌ **Second failure**: Direction padding (STEP 2a) - **FIX NEXT**
   - Check half-precision conversion
   - Verify buffer types match kernel expectations
   - Check if `d_dir_input_float_` is actually float or half

## Summary of Current Findings

**Root Causes Identified:**

1. **Position Encoding produces NaN** (21 out of 8192 values)
   - Likely: Encoding weights contain NaN or infinity
   - Likely: Hash table interpolation issue

2. **Direction Padding produces corrupt values** (values ~1e-12 instead of ±1)
   - Likely: Type mismatch (treating half as float or vice versa)
   - Likely: Wrong buffer being written to

**Next Actions:**

1. Inspect encoding weights for NaN values
2. Check hash grid interpolation for division by zero
3. Fix type mismatch in direction padding kernel
4. Verify all buffer allocations match their declared types
