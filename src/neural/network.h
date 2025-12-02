#pragma once

#include "config.h"
#include "weight_loader.h"
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace neural {

// Neural network wrapper for tiny-cuda-nn
// Manages hash encoding + multi-decoder MLP architecture
class NeuralNetwork {
public:
    NeuralNetwork(const NetworkConfig& config);
    ~NeuralNetwork();

    // Initialize network from loaded weights
    bool initialize_from_weights(const WeightLoader& loader);

    // Run inference on a batch of 3D positions and directions
    // Inputs:
    //   - positions [batch_size, 3] (x, y, z coordinates)
    //   - directions [batch_size, 3] (dx, dy, dz direction vectors)
    // Outputs:
    //   - visibility [batch_size, 1] - opacity/visibility probability
    //   - normals [batch_size, 3] - surface normal vectors
    //   - depth [batch_size, 1] - distance/depth values
    void inference(
        const float* d_positions,
        const float* d_directions,
        float* d_visibility,
        float* d_normals,
        float* d_depth,
        uint32_t batch_size,
        cudaStream_t stream = nullptr
    );

    // Single point inference (for debugging)
    void inference_single(
        const float* position,   // [3] on host
        const float* direction,  // [3] on host
        float& visibility,       // output on host
        float* normal,           // [3] output on host
        float& depth             // output on host
    );

    // Get network configuration
    const NetworkConfig& config() const { return config_; }

    // Check if network is initialized
    bool is_initialized() const { return initialized_; }

    // Get device pointers (for passing to OptiX kernels)
    void* get_encoding_device_ptr() const;
    void* get_visibility_network_ptr() const;
    void* get_normal_network_ptr() const;
    void* get_depth_network_ptr() const;

private:
    NetworkConfig config_;
    bool initialized_ = false;

    // tiny-cuda-nn components (using forward declarations)
    // We'll use void* to avoid exposing tiny-cuda-nn headers
    void* encoding_ = nullptr;
    void* direction_encoder_ = nullptr;
    void* visibility_network_ = nullptr;
    void* normal_network_ = nullptr;
    void* depth_network_ = nullptr;

    // Persistent GPU memory for network parameters (must outlive the networks)
    // Using void* to avoid exposing tcnn::GPUMemory in header
    void* position_encoder_params_ = nullptr;
    void* direction_encoder_params_ = nullptr;
    void* visibility_decoder_params_ = nullptr;
    void* normal_decoder_params_ = nullptr;
    void* depth_decoder_params_ = nullptr;

    // Temporary device buffers for inference
    float* d_encoded_ = nullptr;            // Position encoding output (float)
    size_t encoded_buffer_size_ = 0;
    float* d_dir_input_float_ = nullptr;    // Direction input in float precision
    size_t dir_input_float_buffer_size_ = 0;
    float* d_dir_encoded_ = nullptr;        // Direction encoding output (float)
    size_t dir_encoded_buffer_size_ = 0;
    float* d_concatenated_float_ = nullptr; // Concatenated position + direction encoding (float precision for networks)
    size_t concatenated_float_buffer_size_ = 0;

    // Helper functions
    void allocate_buffers(uint32_t batch_size);
    void free_buffers();
    bool load_position_encoder_weights(const WeightLoader& loader);
    bool load_direction_encoder_weights(const WeightLoader& loader);
    bool load_network_weights(void* network, const WeightLoader& loader,
                             const std::string& prefix, const NetworkConfig::DecoderConfig& decoder_config);
};

} // namespace neural
