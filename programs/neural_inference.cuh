#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "neural_types.h"
#include "divergence_profiling.cuh"

// ============================================================================
// Activation Functions
// ============================================================================

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ __half relu(__half x) {
    return __hmax(__float2half(0.0f), x);
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ __half sigmoid(__half x) {
    return __float2half(1.0f) / (__float2half(1.0f) + hexp(__hneg(x)));
}

// ============================================================================
// Hash Grid Encoding
// ============================================================================

// Spatial hash function (matching tiny-cuda-nn implementation)
// Uses prime number multiplication and XOR
__device__ __forceinline__ uint32_t hash_grid_index(
    uint32_t x,
    uint32_t y,
    uint32_t z,
    uint32_t hashmap_size
) {
    // Prime numbers for spatial hashing
    constexpr uint32_t primes[3] = {1u, 2654435761u, 805459861u};

    uint32_t result = 0;
    result ^= x * primes[0];
    result ^= y * primes[1];
    result ^= z * primes[2];

    // Modulo by hashmap size (power of 2, so we can use bitwise AND)
    return result;
}

// Hash grid encoding with trilinear interpolation
// Input: 3D position (x, y, z) in [0, 1]^3
// Output: encoded features (n_levels * n_features_per_level dimensions)
// div_counter: optional pointer to accumulate divergence measurements
__device__ __forceinline__ void hash_encode(
    const float3& position,
    const HashGridParams& params,
    float* output,
    unsigned int* div_counter = nullptr
) {
    // For each level in the hash grid
    for (uint32_t level = 0; level < params.n_levels; ++level) {
        // Compute grid resolution for this level
        float scale = params.base_resolution * powf(params.per_level_scale, (float)level) - 1.0;
        uint32_t grid_resolution = (uint32_t)ceilf(scale) + 1;
        uint32_t grid_volume = grid_resolution*grid_resolution*grid_resolution;


        // Scale position to grid coordinates
        float3 pos_scaled = make_float3(
            position.x * scale + 0.5f,
            position.y * scale + 0.5f,
            position.z * scale + 0.5f
        );

        // Get integer grid coordinates (floor)
        uint32_t x0 = (uint32_t)floorf(pos_scaled.x);
        uint32_t y0 = (uint32_t)floorf(pos_scaled.y);
        uint32_t z0 = (uint32_t)floorf(pos_scaled.z);

        // Get fractional part for interpolation
        float fx = pos_scaled.x - (float)x0;
        float fy = pos_scaled.y - (float)y0;
        float fz = pos_scaled.z - (float)z0;

        // Get hash table size for this level (may be smaller than max)
        // offset_table stores offsets in units of grid points, not features
        uint32_t level_offset_grid_points = params.offset_table[level];
        uint32_t hashmap_size = params.offset_table[level+1] - level_offset_grid_points;
        uint32_t level_offset_features = level_offset_grid_points * params.n_features_per_level;

        // For each feature in this level
        for (uint32_t f = 0; f < params.n_features_per_level; ++f) {
            // Trilinear interpolation: sample 8 corners of the cube
            float values[8];

            for (int i = 0; i < 8; ++i) {
                uint32_t dx = (i & 1);
                uint32_t dy = (i & 2) >> 1;
                uint32_t dz = (i & 4) >> 2;

                uint32_t hash_idx = 0;
                bool use_direct_index = (grid_volume <= hashmap_size);

                // Measure divergence: direct indexing vs hashing
                if (div_counter != nullptr) {
                    *div_counter += measure_divergence(use_direct_index);
                }

                // If grid fits into hashmap at this resolution, directly index
                if (use_direct_index) {
                    hash_idx = x0 + dx;
                    hash_idx += (y0 + dy) * grid_resolution;
                    hash_idx += (z0 + dz) * grid_resolution * grid_resolution;
                } else {
                    // Hash the corner coordinates
                    hash_idx = hash_grid_index(
                        x0 + dx,
                        y0 + dy,
                        z0 + dz,
                        hashmap_size
                    );
                }

                hash_idx = hash_idx % hashmap_size;

                // Lookup feature value from hash table (offset is in features, hash_idx is in grid points)
                uint32_t table_idx = level_offset_features + hash_idx * params.n_features_per_level + f;
                values[i] = params.hash_table[table_idx];
            }

            // Trilinear interpolation
            float c00 = values[0] * (1.0f - fx) + values[1] * fx;
            float c01 = values[2] * (1.0f - fx) + values[3] * fx;
            float c10 = values[4] * (1.0f - fx) + values[5] * fx;
            float c11 = values[6] * (1.0f - fx) + values[7] * fx;

            float c0 = c00 * (1.0f - fy) + c01 * fy;
            float c1 = c10 * (1.0f - fy) + c11 * fy;

            float result = c0 * (1.0f - fz) + c1 * fz;

            // Write to output
            output[level * params.n_features_per_level + f] = result;
        }
    }
}

// ============================================================================
// MLP Forward Pass
// ============================================================================

// Matrix-vector multiplication: output = weights * input + bias
// weights is [out_dim, in_dim] in row-major order (PyTorch format)
// bias can be nullptr for FullyFusedMLP (no bias)
__device__ __forceinline__ void matmul_add_bias(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    uint32_t in_dim,
    uint32_t out_dim
) {
    for (uint32_t i = 0; i < out_dim; ++i) {
        float sum = (bias != nullptr) ? bias[i] : 0.0f;
        for (uint32_t j = 0; j < in_dim; ++j) {
            // Row-major indexing: weights[out_idx, in_idx] = weights[out_idx * in_dim + in_idx]
            sum += weights[i * in_dim + j] * input[j];
        }
        output[i] = sum;
    }
}

// Half precision version of matmul
__device__ __forceinline__ void matmul_add_bias_fp16(
    const __half* input,
    const __half* weights,
    const __half* bias,
    __half* output,
    uint32_t in_dim,
    uint32_t out_dim
) {
    const __half zero = __float2half(0.0f);
    for (uint32_t i = 0; i < out_dim; ++i) {
        __half sum = (bias != nullptr) ? bias[i] : zero;
        for (uint32_t j = 0; j < in_dim; ++j) {
            // Row-major indexing: weights[out_idx, in_idx] = weights[out_idx * in_dim + in_idx]
            sum = __hadd(sum, __hmul(weights[i * in_dim + j], input[j]));
        }
        output[i] = sum;
    }
}

// Apply activation function element-wise
__device__ __forceinline__ void apply_activation(
    float* data,
    uint32_t size,
    const char* activation
) {
    // Simple string comparison for activation type
    // In device code, we compare by checking first character
    char act_type = activation[0];

    if (act_type == 'r' || act_type == 'R') {  // ReLU
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = relu(data[i]);
        }
    } else if (act_type == 's' || act_type == 'S') {  // Sigmoid
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = sigmoid(data[i]);
        }
    }
    // else: "none" or "None" - no activation
}

// Half precision version of apply_activation
__device__ __forceinline__ void apply_activation_fp16(
    __half* data,
    uint32_t size,
    const char* activation
) {
    char act_type = activation[0];

    if (act_type == 'r' || act_type == 'R') {  // ReLU
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = relu(data[i]);
        }
    } else if (act_type == 's' || act_type == 'S') {  // Sigmoid
        for (uint32_t i = 0; i < size; ++i) {
            data[i] = sigmoid(data[i]);
        }
    }
    // else: "none" or "None" - no activation
}

// Full MLP forward pass
// input: input vector
// params: MLP parameters (layers, activation)
// output: output vector
// scratch: temporary buffer for intermediate activations (must be large enough)
// div_counter: optional pointer to accumulate divergence measurements
__device__ __forceinline__ void mlp_forward(
    const float* input,
    const MLPParams& params,
    float* output,
    float* scratch,
    unsigned int* div_counter = nullptr
) {
    const float* current_input = input;
    float* layer_output = scratch;

    // Forward through all layers
    for (uint32_t l = 0; l < params.n_layers; ++l) {
        const MLPLayer& layer = params.layers[l];

        // Matrix multiply + bias
        // NOTE: FullyFusedMLPs do not have a bias vector, so it will always be zero
        matmul_add_bias(
            current_input,
            layer.weights,
            layer.biases,
            layer_output,
            layer.in_dim,
            layer.out_dim
        );

        // Apply activation
        bool is_hidden_layer = (l < params.n_layers - 1);

        // Measure divergence: hidden vs output layer activation
        if (div_counter != nullptr) {
            *div_counter += measure_divergence(is_hidden_layer);
        }

        if (is_hidden_layer) {
            // Hidden layers: use ReLU
            apply_activation(layer_output, layer.out_dim, "ReLU");
        } else {
            // Output layer: use specified activation
            apply_activation(layer_output, layer.out_dim, params.output_activation);
        }

        // Output becomes input for next layer
        current_input = layer_output;
        layer_output += layer.out_dim;  // Move to next section of scratch buffer
    }

    // Copy final output
    const MLPLayer& last_layer = params.layers[params.n_layers - 1];
    for (uint32_t i = 0; i < last_layer.out_dim; ++i) {
        output[i] = current_input[i];
    }
}

// Hybrid MLP forward pass: fp32 weights, fp16 computation
// This matches tiny-cuda-nn's behavior more closely
__device__ __forceinline__ void mlp_forward_fp16(
    const float* input,
    const MLPParams& params,
    float* output,
    float* scratch  // Will use as fp16 buffer
) {
    __half* fp16_scratch = reinterpret_cast<__half*>(scratch);

    // Convert input to fp16
    const MLPLayer& first_layer = params.layers[0];
    __half* fp16_input = fp16_scratch;
    for (uint32_t i = 0; i < first_layer.in_dim; ++i) {
        fp16_input[i] = __float2half(input[i]);
    }

    __half* current_input_fp16 = fp16_input;
    __half* layer_output_fp16 = fp16_scratch + first_layer.in_dim;

    // Forward through all layers in fp16
    for (uint32_t l = 0; l < params.n_layers; ++l) {
        const MLPLayer& layer = params.layers[l];

        // Convert weights to fp16 on the fly and compute
        const __half zero = __float2half(0.0f);
        for (uint32_t i = 0; i < layer.out_dim; ++i) {
            __half sum = (layer.biases != nullptr) ? __float2half(layer.biases[i]) : zero;
            for (uint32_t j = 0; j < layer.in_dim; ++j) {
                __half w = __float2half(layer.weights[i * layer.in_dim + j]);
                sum = __hadd(sum, __hmul(w, current_input_fp16[j]));
            }
            layer_output_fp16[i] = sum;
        }

        // Apply activation in fp16
        if (l < params.n_layers-1) {
            apply_activation_fp16(layer_output_fp16, layer.out_dim, "ReLU");
        }
        else {
            apply_activation_fp16(layer_output_fp16, layer.out_dim, params.output_activation);
        }

        // Output becomes input for next layer
        current_input_fp16 = layer_output_fp16;
        layer_output_fp16 += layer.out_dim;
    }

    // Convert final output back to fp32
    const MLPLayer& last_layer = params.layers[params.n_layers-1];
    for (uint32_t i = 0; i < last_layer.out_dim; ++i) {
        output[i] = __half2float(current_input_fp16[i]);
    }
}

// ============================================================================
// Complete Neural Network Inference
// ============================================================================

// Full inference pipeline
// position: 3D position in [0, 1]^3
// direction: 3D direction vector (normalized)
// net_params: all network parameters
// Output: visibility, normal (3D), depth
// div_hash: optional pointer to accumulate hash encoding divergence
// div_mlp: optional pointer to accumulate MLP divergence
__device__ __forceinline__ void neural_inference(
    const float3& position,
    const float3& direction,
    const NeuralNetworkParams& net_params,
    float& visibility,
    float3& normal,
    float& depth,
    unsigned int* div_hash = nullptr,
    unsigned int* div_mlp = nullptr
) {
    // Allocate local scratch buffer (on stack)
    // Memory layout:
    // position_encoding: scratch[0..31] (32 floats)
    // direction_input: scratch[64..79] (16 floats)
    // direction_encoding: scratch[80..95] (16 floats)
    // dir_scratch: scratch[96..239] (144 floats for 5-layer MLP)
    // concatenated: scratch[240..287] (48 floats)
    // decoder_scratch: scratch[288..431] (144 floats for 5-layer decoder)
    // Total: 432 floats minimum
    float scratch[512];  // Increased from 256 to 512 to avoid overflow

    // Step 1: Hash encode position -> 32D
    float* position_encoding = scratch;
    hash_encode(position, net_params.hash_encoding, position_encoding, div_hash);
    uint32_t pos_encoding_dim = net_params.hash_encoding.n_levels *
                                 net_params.hash_encoding.n_features_per_level;

    // Step 2: Encode direction (pad to 16D, then run through MLP) -> 16D
    float* direction_input = scratch + 64;
    direction_input[0] = direction.x;
    direction_input[1] = direction.y;
    direction_input[2] = direction.z;

    // Pad with ones (matching PyTorch training configuration)
    for (uint32_t i = 3; i < 16; ++i) {
        direction_input[i] = 1.0f;
    }

    float* direction_encoding = scratch + 80;
    float* dir_scratch = scratch + 96;
    mlp_forward(direction_input, net_params.direction_encoder, direction_encoding, dir_scratch, div_mlp);

    // Step 3: Concatenate position + direction encodings -> 48D
    float* concatenated = scratch + 240;  // Updated offset
    for (uint32_t i = 0; i < pos_encoding_dim; ++i) {
        concatenated[i] = position_encoding[i];
    }
    uint32_t dir_encoding_dim = net_params.direction_encoder.layers[
        net_params.direction_encoder.n_layers - 1
    ].out_dim;
    for (uint32_t i = 0; i < dir_encoding_dim; ++i) {
        concatenated[pos_encoding_dim + i] = direction_encoding[i];
    }

    // Step 4: Run three decoders with fp16 precision
    float* decoder_scratch = scratch + 288;  // Updated offset

    // Visibility decoder (48D -> 32D -> 32D -> 32D -> 1D with sigmoid)
    float vis_output[16];
    mlp_forward(concatenated, net_params.visibility_decoder, vis_output, decoder_scratch, div_mlp);
    visibility = vis_output[0];

    // Normal decoder (48D -> 32D -> 32D -> 32D -> 3D)
    float norm_output[16];
    mlp_forward(concatenated, net_params.normal_decoder, norm_output, decoder_scratch, div_mlp);
    normal = make_float3(norm_output[0], norm_output[1], norm_output[2]);

    // Depth decoder (48D -> 32D -> 32D -> 32D -> 1D)
    float depth_output[16];
    mlp_forward(concatenated, net_params.depth_decoder, depth_output, decoder_scratch, div_mlp);
    depth = depth_output[0];
}
