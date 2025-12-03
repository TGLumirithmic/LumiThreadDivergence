#pragma once

#include "../neural/config.h"
#include "../neural/weight_loader.h"
#include "../../programs/neural_types.h"
#include <cuda_runtime.h>
#include <memory>

namespace optix {

// Host-side container for neural network parameters
// Manages device memory and provides conversion from tiny-cuda-nn format
class NeuralNetworkParamsHost {
public:
    NeuralNetworkParamsHost(const neural::NetworkConfig& config);
    ~NeuralNetworkParamsHost();

    // Load and convert weights from WeightLoader
    bool load_from_weights(const neural::WeightLoader& loader);

    // Get device-side parameters (to pass to OptiX launch params)
    const NeuralNetworkParams& get_device_params() const { return d_params_; }

    // Check if loaded
    bool is_loaded() const { return loaded_; }

private:
    neural::NetworkConfig config_;
    bool loaded_ = false;

    // Device-side parameter structure
    NeuralNetworkParams d_params_;

    // Host-side storage for managing device memory
    // Hash grid
    float* d_hash_table_ = nullptr;
    uint32_t* d_hash_offsets_ = nullptr;
    size_t hash_table_size_ = 0;

    // Direction encoder layers
    MLPLayer* d_dir_encoder_layers_ = nullptr;
    uint32_t dir_encoder_n_layers_ = 0;

    // Visibility decoder layers
    MLPLayer* d_vis_decoder_layers_ = nullptr;
    uint32_t vis_decoder_n_layers_ = 0;

    // Normal decoder layers
    MLPLayer* d_norm_decoder_layers_ = nullptr;
    uint32_t norm_decoder_n_layers_ = 0;

    // Depth decoder layers
    MLPLayer* d_depth_decoder_layers_ = nullptr;
    uint32_t depth_decoder_n_layers_ = 0;

    // Activation strings (device memory)
    char* d_relu_str_ = nullptr;
    char* d_sigmoid_str_ = nullptr;
    char* d_none_str_ = nullptr;

    // Helper functions
    bool load_hash_encoding(const neural::WeightLoader& loader);
    bool load_mlp(
        const neural::WeightLoader& loader,
        const std::string& prefix,
        uint32_t n_layers,
        uint32_t hidden_dim,
        uint32_t input_dim,
        uint32_t output_dim,
        const std::string& output_activation,
        MLPLayer*& d_layers_out,
        uint32_t& n_layers_out
    );

    void free_device_memory();
    char* allocate_activation_string(const std::string& str);
};

} // namespace optix
