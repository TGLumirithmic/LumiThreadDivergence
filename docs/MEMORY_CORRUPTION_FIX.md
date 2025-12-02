# Memory Corruption Fix - Summary

## Date: 2025-12-02

## Problem

The neural network inference was producing all NaN outputs. After extensive debugging, the root cause was identified as **memory corruption in weight loading**.

## Root Cause

**TCNN's `set_params()` stores pointers without copying data for hash grid encodings.**

The original code used temporary `tcnn::GPUMemory` objects that were destroyed after the weight loading function returned:

```cpp
// WRONG - Memory gets freed!
tcnn::GPUMemory<precision_t> gpu_params(n_params);  // Temporary object
gpu_params.copy_from_host(params_fp16.data());
encoding->set_params(gpu_params.data(), gpu_params.data(), gpu_params.data());
// gpu_params goes out of scope here and FREES THE MEMORY
return true;  // Encoding now has dangling pointers!
```

### Why This Happened

1. `tcnn::GPUMemory<T>` is an RAII wrapper that **owns** GPU memory
2. When `gpu_params` goes out of scope, its destructor calls `cudaFree()`
3. TCNN's `set_params()` **does NOT copy the data** for hash grid encodings - it stores the pointers
4. After the function returns, TCNN is holding **dangling pointers** to freed memory

This is because hash grid encodings don't implement `set_params_impl()`, so the pointers must persist throughout the object's lifetime.

## Solution

**Made GPUMemory objects persistent as class members.**

### Changes Made

1. **Added persistent parameter storage to network.h** ([network.h:45-50](../src/neural/network.h#L45-L50)):
   ```cpp
   // Persistent GPU memory for network parameters (must outlive the networks)
   void* position_encoder_params_ = nullptr;
   void* direction_encoder_params_ = nullptr;
   void* visibility_decoder_params_ = nullptr;
   void* normal_decoder_params_ = nullptr;
   void* depth_decoder_params_ = nullptr;
   ```

2. **Updated destructor to clean up** ([network.cpp:49-64](../src/neural/network.cpp#L49-L64)):
   ```cpp
   if (position_encoder_params_) {
       delete reinterpret_cast<tcnn::GPUMemory<precision_t>*>(position_encoder_params_);
   }
   // ... same for other 4 parameter buffers
   ```

3. **Updated weight loading to use persistent allocation** ([network.cpp:597-607](../src/neural/network.cpp#L597-L607)):
   ```cpp
   // CRITICAL: Allocate PERSISTENT GPU memory for parameters
   auto* gpu_params = new tcnn::GPUMemory<precision_t>(n_params);
   gpu_params->copy_from_host(params_fp16.data());

   // Store the persistent GPU memory pointer
   position_encoder_params_ = gpu_params;

   // Pass pointers to PERSISTENT memory (gpu_params will live until destructor)
   encoding->set_params(gpu_params->data(), gpu_params->data(), gpu_params->data());
   ```

## Evidence

**Before the fix:**
```
[CHECKPOINT] Verifying weights immediately after load:
  [0] = -0.12854004  ✅ Correct
  [1] = -0.10314941  ✅ Correct

[CHECKPOINT] After position encoder load:
  params() pointer: 0x7efceda00000
CUDA error: an illegal memory access was encountered  ❌ Memory freed!
```

**After the fix:**
```
Position encoding produces valid values:
  NaN count: 0 ✅
  Zero count: 0 ✅
  Min: -0.312, Max: 0.354

All decoders produce valid outputs:
  Visibility: Min=0, Max=1, 0 NaN ✅
  Normal: Min=-0.082, Max=0.083, 0 NaN ✅
  Depth: Min=0.005, Max=1.545, 0 NaN ✅
```

## Additional Fixes

While debugging, we also fixed:

1. **Half precision buffer allocation** - Buffers now allocate correct byte size based on `TCNN_HALF_PRECISION`:
   - `d_dir_input_float_`: 2 bytes/element when half precision, 4 bytes when float
   - `d_concatenated_float_`: Same as above

2. **Half precision debug support** - Added `print_device_values_half()` function for inspecting `__half*` buffers

## Impact

This bug caused:
- ✅ **FIXED**: Illegal memory access when reading parameters after loading
- ✅ **FIXED**: Garbage weights during inference (reading from freed memory)
- ✅ **FIXED**: NaN outputs because the neural network was using corrupted/random weights

## Testing

```bash
cd /workspaces/LumiThreadDivergence/build
make -j8
./bin/test_network ../data/models/weights.bin ../data/test/predictions.bin ../output
```

Results:
- ✅ Simple grid test (262,144 samples) completes successfully
- ✅ All outputs are valid (no NaN values)
- ✅ Visualization images written correctly

## Lesson Learned

**Never pass pointers to temporary objects to APIs that expect persistent memory!**

TCNN's `set_params()` is designed for scenarios where you manage parameter memory externally (e.g., during training with optimizers). It doesn't copy - it stores the pointers for efficient updates. For hash grid encodings specifically, the memory must persist for the lifetime of the encoding object.
