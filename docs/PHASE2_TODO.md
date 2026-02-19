# Phase 2 Implementation TODO List

## Overview
Since tiny-cuda-nn cannot be called from OptiX device code, we need to reimplement the neural network inference using custom CUDA kernels that are compatible with OptiX programs. This document outlines the detailed tasks required.

---

## 1. Neural Network Architecture Analysis

### 1.1 Current Network Structure (from network.cpp)
- [x] **Position Encoder**: Hash grid encoding
  - 3D input (position)
  - 16 levels, 2 features per level
  - Log2 hash table size: 14
  - Base resolution: 16
  - Max resolution: 1024
  - Output: 32D (16 levels × 2 features)

- [x] **Direction Encoder** (optional): Fully-fused MLP
  - Input: 3D (direction), padded to 16D with ones (NOT zeros)
  - Architecture: 1 hidden layer, 16 neurons
  - Output: 16D

- [x] **Decoders**: Three separate fully-fused MLPs
  - Input: 32D (position) + 16D (direction) = 48D total
  - Visibility decoder: 3 hidden layers → 1D output (sigmoid)
  - Normal decoder: 3 hidden layers → 3D output (none)
  - Depth decoder: 3 hidden layers → 1D output (none)
  - Hidden layer width: 32 neurons
  - Activation: ReLU

### 1.2 Weight Format Understanding
- [ ] **Analyze weight storage format** in WeightLoader
  - Position encoder: `position_encoder.params` (hash table entries)
  - Direction encoder: `direction_encoder.params` (MLP weights)
  - Visibility decoder: `visibility_decoder.params`
  - Normal decoder: `normal_decoder.params`
  - Depth decoder: `depth_decoder.params`

- [ ] **Document weight organization**
  - How are MLP weights stored? (row-major vs column-major)
  - Layer-by-layer or flattened?
  - Precision: float32 or float16?

---

## 2. Custom CUDA Kernel Implementation

### 2.1 Hash Grid Encoding (`__device__` function)
- [ ] **Create `programs/neural_inference.cuh`** header file
  - Define hash grid parameters structure
    - Note that the logic in tcnn must be carefully considered as the hash grid will not necessarily be of size num_levelsxhash_table_sizexnum_features
    - hash table sizes at particular levels may be smaller based on the resolution at that level
  - Define MLP layer parameters structure
  - Declare all `__device__` functions

- [ ] **Implement hash function**
  ```cuda
  __device__ uint32_t hash_position(int3 grid_pos, uint32_t table_size)
  ```
  - Use spatial hash function (matching tiny-cuda-nn)
  - XOR-based hash commonly used

- [ ] **Implement trilinear interpolation**
  ```cuda
  __device__ void hash_encode(
      const float3& position,
      const HashGridParams& params,
      float* output
  )
  ```
  - For each of 16 levels:
    - Compute grid resolution
    - Find 8 corner grid cells
    - Hash each corner to lookup feature indices
    - Perform trilinear interpolation
    - Write to output buffer (32D total)

### 2.2 MLP Inference (`__device__` function)
- [ ] **Implement matrix-vector multiply**
  ```cuda
  __device__ void matmul(
      const float* input,
      const float* weights,
      const float* bias,
      float* output,
      int in_dim,
      int out_dim
  )
  ```

- [ ] **Implement activation functions**
  ```cuda
  __device__ float relu(float x)
  __device__ float sigmoid(float x)
  ```

- [ ] **Implement full MLP forward pass**
  ```cuda
  __device__ void mlp_forward(
      const float* input,
      const MLPParams& params,
      float* output,
      float* scratch_buffer  // for intermediate activations
  )
  ```
  - Loop through layers
  - Apply matmul → activation
  - Handle output activation (sigmoid for visibility, none for others)

### 2.3 Full Network Inference
- [ ] **Implement complete inference pipeline**
  ```cuda
  __device__ void neural_inference(
      const float3& position,
      const float3& direction,
      const NeuralNetworkParams& net_params,
      float& visibility,
      float3& normal,
      float& depth
  )
  ```
  - Step 1: Hash encode position → 32D
  - Step 2: MLP encode direction (pad to 16D) → 16D
  - Step 3: Concatenate → 48D
  - Step 4: Run three decoders in parallel (or sequentially)
  - Step 5: Write outputs

---

## 3. Weight Loading and Data Structures

### 3.1 Device-Side Data Structures
- [ ] **Create `src/optix/neural_params.h`**
  ```cpp
  struct HashGridParams {
      float* hash_table;      // Flattened hash table
      uint32_t n_levels;
      uint32_t n_features_per_level;
      uint32_t log2_hashmap_size;
      float base_resolution;
      float per_level_scale;
  };

  struct MLPParams {
      float** layer_weights;  // Array of weight matrices
      float** layer_biases;   // Array of bias vectors
      uint32_t n_layers;
      uint32_t* layer_in_dims;
      uint32_t* layer_out_dims;
      const char* output_activation;  // "relu", "sigmoid", "none"
  };

  struct NeuralNetworkParams {
      HashGridParams position_encoder;
      MLPParams direction_encoder;
      MLPParams visibility_decoder;
      MLPParams normal_decoder;
      MLPParams depth_decoder;
  };
  ```

### 3.2 Weight Conversion System
- [ ] **Create `src/optix/weight_converter.h/cpp`**
  - Function to extract weights from tiny-cuda-nn GPUMemory
  - Function to reorganize weights for OptiX access pattern
  - Function to allocate and copy to device memory

- [ ] **Implement hash table conversion**
  ```cpp
  float* convert_hash_table(const WeightLoader& loader);
  ```
  - Extract from `position_encoder.params`
  - Organize as [n_levels][table_size][n_features]

- [ ] **Implement MLP weight conversion**
  ```cpp
  MLPParams convert_mlp_weights(
      const WeightLoader& loader,
      const std::string& prefix,
      const NetworkConfig::DecoderConfig& config
  );
  ```
  - Extract layer-by-layer from `.params` tensor
  - Separate weights from biases (if interleaved)
  - Handle transpose if needed (row-major vs column-major)

### 3.3 Update LaunchParams
- [ ] **Modify `programs/common.h`**
  - Replace void pointers with `NeuralNetworkParams` structure
  - Ensure proper alignment for device access
  ```cpp
  struct LaunchParams {
      // ... existing fields ...
      NeuralNetworkParams neural_network;
  };
  ```

---

## 4. OptiX Integration

### 4.1 Update Closest-Hit Program
- [ ] **Implement `__closesthit__neural` in `programs/neural_programs.cu`**
  - Include `neural_inference.cuh`
  - Get hit position and ray direction
  - Call `neural_inference()` device function
  - Use outputs to compute final color:
    - Visibility: modulate brightness
    - Normal: compute lighting (simple diffuse for now)
    - Depth: could affect color or be visualized
  - Set ray payload with computed color

### 4.2 OptiX Pipeline Setup
- [ ] **Complete `src/optix/pipeline.cpp`**
  - Load PTX from compiled CUDA programs
  - Create module from PTX
  - Create program groups:
    - Raygen: `__raygen__rg`
    - Miss: `__miss__ms`
    - Hit group: `__intersection__neural` + `__closesthit__neural`
  - Link pipeline
  - Set stack sizes appropriately

### 4.3 Geometry and Acceleration Structure
- [ ] **Complete `src/optix/geometry.cpp`**
  - Build custom primitive AABB for neural asset
  - Create BLAS with `OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES`
  - For now: single AABB, no TLAS needed
  - Store `OptixTraversableHandle` in `LaunchParams`

### 4.4 Shader Binding Table (SBT)
- [ ] **Complete `src/optix/sbt.cpp`**
  - Create SBT records for:
    - Raygen program (1 record)
    - Miss program (1 record)
    - Hit group (1 record for neural asset)
  - Populate SBT with program headers
  - No per-instance data needed yet

---

## 5. Main Application Flow

### 5.1 Update `src/main.cpp`
- [ ] **Initialize OptiX context** (already done in `context.cpp`)
- [ ] **Load neural network weights**
  ```cpp
  WeightLoader loader;
  loader.load_from_file("path/to/weights.msgpack");
  ```
- [ ] **Convert weights for OptiX**
  ```cpp
  NeuralNetworkParams net_params = convert_weights(loader, config);
  ```
- [ ] **Setup OptiX pipeline and geometry**
  ```cpp
  Pipeline pipeline;
  pipeline.create(context, "path/to/programs.ptx");

  Geometry geometry;
  geometry.build_neural_asset_blas(context, neural_bounds);
  ```
- [ ] **Setup SBT**
  ```cpp
  SBT sbt;
  sbt.build(pipeline);
  ```
- [ ] **Prepare launch params**
  ```cpp
  LaunchParams params;
  params.frame_buffer = allocate_framebuffer(width, height);
  params.camera = setup_camera();
  params.traversable = geometry.get_handle();
  params.neural_network = net_params;
  ```
- [ ] **Launch OptiX**
  ```cpp
  optixLaunch(
      pipeline.get(),
      stream,
      &params, sizeof(LaunchParams),
      &sbt.get(),
      width, height, 1
  );
  ```
- [ ] **Save output image**
  ```cpp
  save_ppm("output.ppm", frame_buffer, width, height);
  ```

---

## 6. Testing and Validation

The file in data/test/predictions.bin can be read in with the PredictionReader in tests/neural_proxy_predictions.h to get the outputs and intermediate values produced by the tcnn implementation for comparison.

### 6.1 Unit Tests
- [ ] **Test hash grid encoding**
  - Compare against tiny-cuda-nn encoding output
  - Test cases: various 3D positions
  - Tolerance: < 1e-5 absolute error

- [ ] **Test MLP inference**
  - Create simple 2-layer MLP with known weights
  - Verify matmul correctness
  - Test activation functions

- [ ] **Test full network inference**
  - Run same inputs through tiny-cuda-nn and custom implementation
  - Compare visibility, normal, depth outputs
  - Tolerance: < 1e-4 (accounting for precision differences)

### 6.2 Integration Tests
- [ ] **Test OptiX pipeline**
  - Verify PTX compilation
  - Verify program group creation
  - Verify SBT setup

- [ ] **Test rendering**
  - Render simple test scene
  - Verify no CUDA errors
  - Verify output image is not black/corrupted

### 6.3 Visual Validation
- [ ] **Compare rendered images**
  - Render with tiny-cuda-nn (Phase 1 test harness)
  - Render with custom OptiX implementation
  - Visual comparison (should look nearly identical)
  - Quantitative: compute PSNR or MSE

---

## 7. Debugging and Optimization

### 7.1 Common Issues to Watch For
- [ ] **Memory alignment**
  - Ensure weight pointers are properly aligned
  - Check for unaligned memory access errors

- [ ] **Coordinate system mismatches**
  - Verify position is in correct coordinate space
  - Check AABB bounds match training bounds

- [ ] **Numerical precision**
  - Hash table lookups out of bounds
  - Trilinear interpolation edge cases
  - NaN/Inf propagation in MLPs

- [ ] **OptiX-specific issues**
  - Stack overflow (increase with `optixPipelineSetStackSize`)
  - Payload size limits
  - Attribute limits in intersection program

### 7.2 Performance Considerations (Future)
- [ ] Profile inference time per ray
- [ ] Consider batch inference if possible
- [ ] Optimize memory layout for coalesced access
- [ ] Consider shared memory for weights (limited applicability in OptiX)

---

## 8. Documentation

### 8.1 Code Documentation
- [ ] Document hash grid implementation details
- [ ] Document MLP weight layout assumptions
- [ ] Add comments to OptiX programs explaining pipeline

### 8.2 User Documentation
- [ ] Document weight export process from PyTorch
- [ ] Document how to run the renderer
- [ ] Document expected output and validation process

---

## Summary Checklist

**Critical Path Tasks:**
1. ✅ Understand current network architecture
2. ⬜ Implement hash grid encoding device function
3. ⬜ Implement MLP inference device function
4. ⬜ Create weight conversion system
5. ⬜ Update LaunchParams with neural network params
6. ⬜ Implement complete `__closesthit__neural` program
7. ⬜ Complete OptiX pipeline setup (pipeline.cpp)
8. ⬜ Complete geometry/BLAS setup (geometry.cpp)
9. ⬜ Complete SBT setup (sbt.cpp)
10. ⬜ Update main.cpp with full render loop
11. ⬜ Test and validate against tiny-cuda-nn output
12. ⬜ Render first successful image

**Estimated Complexity:**
- **Hash Grid Encoding**: Medium (requires careful implementation of spatial hash + interpolation)
- **MLP Inference**: Low-Medium (standard matmul + activation)
- **Weight Conversion**: Medium (need to understand tiny-cuda-nn's weight format)
- **OptiX Integration**: Medium-High (multiple components need to work together)
- **Testing/Validation**: High (critical to ensure correctness)

**Next Immediate Steps:**
1. Analyze weight format from WeightLoader (examine msgpack structure)
2. Implement and test hash grid encoding in isolation
3. Implement and test MLP inference in isolation
4. Create weight conversion utilities
5. Integrate into OptiX programs
