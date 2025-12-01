#pragma once

#include <cstdint>
#include <string>
#include <cmath>

namespace neural {

// Configuration for Instant-NGP style network with multiple decoders
struct NetworkConfig {
    // Hash encoding parameters
    uint32_t n_levels = 16;              // Number of hash grid levels
    uint32_t n_features_per_level = 2;   // Features per level
    uint32_t log2_hashmap_size = 14;     // Hash table size (2^14)
    float base_resolution = 16.0f;       // Base/coarsest resolution
    float max_resolution = 2048.0f;      // Maximum/finest resolution

    // Compute per-level scale from base and max resolution
    float compute_per_level_scale() const {
        return std::exp((std::log(max_resolution) - std::log(base_resolution)) / (n_levels - 1));
    }

    // Decoder outputs (multiple decoders from shared encoding)
    struct DecoderConfig {
        std::string name;
        uint32_t n_output_dims;
        std::string output_activation;
        uint32_t n_decoder_layers = 4;  // Number of hidden layers in decoder MLP (not counting output layer)
    };

    // Three decoders:
    // 1. Visibility decoder (any-hit) - 1D output
    // 2. Normal decoder - 3D output (nx, ny, nz)
    // 3. Depth decoder (closest-hit) - 1D output
    DecoderConfig visibility_decoder{"visibility", 1, "Sigmoid"};  // [0,1] probability
    DecoderConfig normal_decoder{"normal", 3, "None"};             // normalized vector
    DecoderConfig depth_decoder{"depth", 1, "None"};               // distance value

    // Direction encoder parameters
    bool use_direction_encoder = true;   // Whether to use direction encoding
    uint32_t direction_input_dims = 3;   // Direction vector (dx, dy, dz)
    uint32_t direction_hidden_dim = 16;  // Hidden dimension for direction encoder
    uint32_t direction_n_hidden_layers = 1; // Number of hidden layers in direction encoder

    // Decoder MLP parameters
    uint32_t n_neurons = 32;             // Neurons per hidden layer in decoders

    // Input
    uint32_t n_input_dims = 3;           // 3D position (x, y, z)

    // Activation function for hidden layers
    std::string activation = "ReLU";

    // Get total encoding output dimension (position encoding only)
    uint32_t encoding_n_output_dims() const {
        return n_levels * n_features_per_level;
    }

    // Get direction encoder output dimension
    uint32_t direction_encoder_n_output_dims() const {
        return use_direction_encoder ? direction_hidden_dim : 0;
    }

    // Get total decoder input dimension (position + direction encoding)
    uint32_t decoder_input_dims() const {
        return encoding_n_output_dims() + direction_encoder_n_output_dims();
    }

    // Print configuration
    void print() const;

    // Load from file (for future use)
    static NetworkConfig from_json(const std::string& json_path);

    // Default Instant-NGP config with multi-decoder setup
    static NetworkConfig instant_ngp_default();
};

} // namespace neural
