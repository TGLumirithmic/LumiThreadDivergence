# Critical Bug Found: Memory Corruption in Weight Loading

## Date: 2025-12-02

## Summary

**ROOT CAUSE IDENTIFIED**: The encoding weights are being corrupted because `tcnn::GPUMemory` temporary objects are being destroyed, freeing the GPU memory that TCNN's `set_params()` is referencing.

## The Bug

### Location
- `load_position_encoder_weights()` - [network.cpp:801-827](../src/neural/network.cpp#L801-L827)
- `load_direction_encoder_weights()` - Similar pattern
- `load_network_weights()` - Similar pattern

### Code Pattern
```cpp
// WRONG - Memory gets freed!
tcnn::GPUMemory<precision_t> gpu_params(n_params);  // Temporary object
gpu_params.copy_from_host(params_fp16.data());
encoding->set_params(gpu_params.data(), gpu_params.data(), gpu_params.data());
// gpu_params goes out of scope here and FREES THE MEMORY
return true;  // Encoding now has dangling pointers!
```

### Evidence

**Inside the function** (before `gpu_params` is destroyed):
```
[CHECKPOINT] Verifying weights immediately after load:
  [0] = -0.12854004  ✅ Correct
  [1] = -0.10314941  ✅ Correct
  [2] = -0.17224121  ✅ Correct
```

**After the function returns** (after `gpu_params` is destroyed):
```
[CHECKPOINT] After position encoder load:
  params() pointer: 0x7efceda00000
  inference_params() pointer: 0x7efceda00000
CUDA error: an illegal memory access was encountered  ❌ Memory freed!
```

The pointer address is the same, but accessing it causes illegal memory access because the memory was freed by `gpu_params`'s destructor.

## Why This Happens

1. `tcnn::GPUMemory<T>` is an RAII wrapper that **owns** GPU memory
2. When `gpu_params` goes out of scope, its destructor calls `cudaFree()`
3. TCNN's `set_params()` **does NOT copy the data** - it stores the pointers
4. After the function returns, TCNN is holding **dangling pointers** to freed memory

## Solutions

### Option 1: Keep GPUMemory Alive (Recommended)
Store the `GPUMemory` objects as class members so they persist:

```cpp
class NeuralNetwork {
private:
    // Add persistent GPU memory for parameters
    std::unique_ptr<tcnn::GPUMemory<precision_t>> position_encoder_params_;
    std::unique_ptr<tcnn::GPUMemory<precision_t>> direction_encoder_params_;
    // ...
};

// In load function:
position_encoder_params_ = std::make_unique<tcnn::GPUMemory<precision_t>>(n_params);
position_encoder_params_->copy_from_host(params_fp16.data());
encoding->set_params(
    position_encoder_params_->data(),
    position_encoder_params_->data(),
    position_encoder_params_->data()
);
```

### Option 2: Use TCNN's Internal Allocation
Let TCNN allocate its own parameter memory:

```cpp
// Get TCNN's internal parameter buffer
auto params_ptr = encoding->params();  // Returns pointer to TCNN's internal buffer
// Copy directly to TCNN's buffer
CUDA_CHECK(cudaMemcpy(params_ptr, params_fp16.data(),
                      n_params * sizeof(precision_t), cudaMemcpyHostToDevice));
```

### Option 3: Manual GPU Allocation
Allocate GPU memory manually and manage lifetime:

```cpp
precision_t* d_params = nullptr;
CUDA_CHECK(cudaMalloc(&d_params, n_params * sizeof(precision_t)));
CUDA_CHECK(cudaMemcpy(d_params, params_fp16.data(),
                      n_params * sizeof(precision_t), cudaMemcpyHostToDevice));
encoding->set_params(d_params, d_params, d_params);
// Store d_params pointer to free later in destructor
```

## Impact

This bug causes:
1. **Illegal memory access** when trying to read parameters after loading
2. **Garbage weights** during inference (reading from freed memory)
3. **NaN outputs** because the neural network is using corrupted/random weights

This explains why:
- Weights looked correct immediately after loading
- Weights became garbage (`-405.5`, `5376`) at inference time
- Position encoding produced NaN values
- All downstream operations failed

## Next Steps

1. Implement Option 1 (persistent GPUMemory members) - SAFEST
2. Test that weights persist correctly across function boundaries
3. Verify inference produces valid outputs
4. Remove temporary synchronization attempts (they don't help)

## Lesson Learned

**Never pass pointers to temporary objects to APIs that expect persistent memory!**

TCNN's `set_params()` is designed for scenarios where you manage parameter memory externally (e.g., during training with optimizers). It doesn't copy - it stores the pointers for efficient updates.
