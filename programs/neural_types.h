#pragma once

#include <cstdint>

// ============================================================================
// Data Structures for Neural Network Parameters (Host and Device)
// ============================================================================

// Hash grid encoding parameters
struct HashGridParams {
    float* hash_table;           // Flattened hash table [total_entries * n_features_per_level]
    uint32_t* offset_table;      // Offset into hash_table for each level [n_levels]
    uint32_t n_levels;           // Number of multi-resolution levels (16)
    uint32_t n_features_per_level; // Features per level (2)
    uint32_t log2_hashmap_size;  // Log2 of hash table size (14 -> 16384 entries)
    float base_resolution;       // Coarsest grid resolution (16.0)
    float per_level_scale;       // Scale factor between levels
};

// MLP layer parameters
struct MLPLayer {
    float* weights;  // Weight matrix [out_dim, in_dim] (row-major)
    float* biases;   // Bias vector [out_dim] - will be nullptr for FullyFusedMLP
    uint32_t in_dim;
    uint32_t out_dim;
};

// Complete MLP parameters
struct MLPParams {
    MLPLayer* layers;            // Array of layer parameters
    uint32_t n_layers;           // Total number of layers (hidden + output)
    const char* output_activation; // "relu", "sigmoid", "none"
};

// Complete neural network parameters for OptiX
struct NeuralNetworkParams {
    HashGridParams hash_encoding;
    MLPParams direction_encoder;
    MLPParams visibility_decoder;
    MLPParams normal_decoder;
    MLPParams depth_decoder;

    // Scratch buffer for intermediate computations (allocated per-launch)
    float* scratch_buffer;
    uint32_t scratch_buffer_size;
};
