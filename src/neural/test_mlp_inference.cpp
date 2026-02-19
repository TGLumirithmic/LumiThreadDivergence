/**
 * test_mlp_inference.cpp
 *
 * Test harness for the MLP and hash grid encoding functions extracted from
 * kernel_source.h. Loads weights from data/models/weights.bin and tests
 * inference on a grid of points on a unit cube face with directions pointing
 * into the cube.
 *
 * Output: A PNG image showing visibility predictions across the cube face.
 */

#include "weight_loader.h"
#include "config.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>

// ============================================================================
// Hash Grid Encoding Parameters (matching kernel_source.h structures)
// ============================================================================

struct HashGridParams {
    std::vector<float> hash_table;
    std::vector<uint32_t> offset_table;
    uint32_t n_levels;
    uint32_t n_features_per_level;
    uint32_t log2_hashmap_size;
    float base_resolution;
    float per_level_scale;
};

struct MLPLayer {
    std::vector<float> weights;
    std::vector<float> biases;
    uint32_t in_dim;
    uint32_t out_dim;
};

struct MLPParams {
    std::vector<MLPLayer> layers;
    char output_activation; // 'r' = relu, 's' = sigmoid, 'n' = none
};

struct NeuralNetworkParams {
    HashGridParams hash_encoding;
    MLPParams direction_encoder;
    MLPParams visibility_decoder;
    MLPParams normal_decoder;
    MLPParams depth_decoder;
};

// ============================================================================
// Activation Functions
// ============================================================================

inline float relu(float x) {
    return std::max(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

// ============================================================================
// Hash Grid Encoding (CPU implementation matching kernel_source.h)
// ============================================================================

inline uint32_t hash_grid_index(uint32_t x, uint32_t y, uint32_t z, uint32_t hashmap_size) {
    constexpr uint32_t primes[3] = {1u, 2654435761u, 805459861u};
    uint32_t result = 0;
    result ^= x * primes[0];
    result ^= y * primes[1];
    result ^= z * primes[2];
    return result;
}

void hash_encode(
    float px, float py, float pz,
    const HashGridParams& params,
    std::vector<float>& output
) {
    output.resize(params.n_levels * params.n_features_per_level);

    for (uint32_t level = 0; level < params.n_levels; ++level) {
        float scale = params.base_resolution * std::pow(params.per_level_scale, (float)level) - 1.0f;
        uint32_t grid_resolution = (uint32_t)std::ceil(scale) + 1;
        uint32_t grid_volume = grid_resolution * grid_resolution * grid_resolution;

        float pos_scaled_x = px * scale + 0.5f;
        float pos_scaled_y = py * scale + 0.5f;
        float pos_scaled_z = pz * scale + 0.5f;

        uint32_t x0 = (uint32_t)std::floor(pos_scaled_x);
        uint32_t y0 = (uint32_t)std::floor(pos_scaled_y);
        uint32_t z0 = (uint32_t)std::floor(pos_scaled_z);

        float fx = pos_scaled_x - (float)x0;
        float fy = pos_scaled_y - (float)y0;
        float fz = pos_scaled_z - (float)z0;

        uint32_t level_offset_grid_points = params.offset_table[level];
        uint32_t hashmap_size = params.offset_table[level + 1] - level_offset_grid_points;
        uint32_t level_offset_features = level_offset_grid_points * params.n_features_per_level;

        for (uint32_t f = 0; f < params.n_features_per_level; ++f) {
            float values[8];

            for (int i = 0; i < 8; ++i) {
                uint32_t dx = (i & 1);
                uint32_t dy = (i & 2) >> 1;
                uint32_t dz = (i & 4) >> 2;

                bool use_direct_index = (grid_volume <= hashmap_size);

                uint32_t hash_idx;
                if (use_direct_index) {
                    hash_idx = (x0 + dx) + (y0 + dy) * grid_resolution +
                               (z0 + dz) * grid_resolution * grid_resolution;
                } else {
                    hash_idx = hash_grid_index(x0 + dx, y0 + dy, z0 + dz, hashmap_size);
                }

                hash_idx = hash_idx % hashmap_size;
                uint32_t table_idx = level_offset_features + hash_idx * params.n_features_per_level + f;

                if (table_idx < params.hash_table.size()) {
                    values[i] = params.hash_table[table_idx];
                } else {
                    values[i] = 0.0f;
                }
            }

            // Trilinear interpolation
            float c00 = values[0] * (1.0f - fx) + values[1] * fx;
            float c01 = values[2] * (1.0f - fx) + values[3] * fx;
            float c10 = values[4] * (1.0f - fx) + values[5] * fx;
            float c11 = values[6] * (1.0f - fx) + values[7] * fx;
            float c0 = c00 * (1.0f - fy) + c01 * fy;
            float c1 = c10 * (1.0f - fy) + c11 * fy;
            float result = c0 * (1.0f - fz) + c1 * fz;

            output[level * params.n_features_per_level + f] = result;
        }
    }
}

// ============================================================================
// MLP Forward Pass (CPU implementation matching kernel_source.h)
// ============================================================================

void matmul_add_bias(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    std::vector<float>& output,
    uint32_t in_dim,
    uint32_t out_dim
) {
    output.resize(out_dim);
    for (uint32_t i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < in_dim; ++j) {
            sum += weights[i * in_dim + j] * input[j];
        }
        output[i] = sum + bias[i];
    }
}

void apply_activation(std::vector<float>& data, char act_type) {
    if (act_type == 'r' || act_type == 'R') {
        for (auto& v : data) v = relu(v);
    } else if (act_type == 's' || act_type == 'S') {
        for (auto& v : data) v = sigmoid(v);
    }
}

void mlp_forward(
    const std::vector<float>& input,
    const MLPParams& params,
    std::vector<float>& output
) {
    std::vector<float> current_input = input;
    std::vector<float> layer_output;

    for (size_t l = 0; l < params.layers.size(); ++l) {
        const MLPLayer& layer = params.layers[l];

        matmul_add_bias(current_input, layer.weights, layer.biases,
                        layer_output, layer.in_dim, layer.out_dim);

        bool is_hidden_layer = (l < params.layers.size() - 1);

        if (is_hidden_layer) {
            apply_activation(layer_output, 'R');
        } else {
            apply_activation(layer_output, params.output_activation);
        }

        current_input = layer_output;
    }

    output = layer_output;
}

// ============================================================================
// Neural Inference (CPU implementation matching kernel_source.h)
// ============================================================================

void neural_inference(
    float px, float py, float pz,
    float dx, float dy, float dz,
    const NeuralNetworkParams& net_params,
    float& visibility,
    float& nx, float& ny, float& nz,
    float& depth
) {
    // Hash encode position
    std::vector<float> position_encoding;
    hash_encode(px, py, pz, net_params.hash_encoding, position_encoding);
    uint32_t pos_encoding_dim = net_params.hash_encoding.n_levels *
                                net_params.hash_encoding.n_features_per_level;

    // Encode direction
    std::vector<float> direction_input(16, 1.0f);
    direction_input[0] = dx;
    direction_input[1] = dy;
    direction_input[2] = dz;

    std::vector<float> direction_encoding;
    if (!net_params.direction_encoder.layers.empty()) {
        mlp_forward(direction_input, net_params.direction_encoder, direction_encoding);
    }

    // Concatenate encodings
    std::vector<float> concatenated;
    concatenated.reserve(pos_encoding_dim + direction_encoding.size());
    for (size_t i = 0; i < pos_encoding_dim; ++i) {
        concatenated.push_back(position_encoding[i]);
    }
    for (size_t i = 0; i < direction_encoding.size(); ++i) {
        concatenated.push_back(direction_encoding[i]);
    }

    // Visibility decoder
    std::vector<float> vis_output;
    mlp_forward(concatenated, net_params.visibility_decoder, vis_output);
    visibility = vis_output.empty() ? 0.0f : vis_output[0];

    // Normal decoder
    std::vector<float> norm_output;
    mlp_forward(concatenated, net_params.normal_decoder, norm_output);
    if (norm_output.size() >= 3) {
        nx = norm_output[0];
        ny = norm_output[1];
        nz = norm_output[2];
    } else {
        nx = ny = nz = 0.0f;
    }

    // Depth decoder
    std::vector<float> depth_output;
    mlp_forward(concatenated, net_params.depth_decoder, depth_output);
    depth = depth_output.empty() ? 0.0f : depth_output[0];
}

// ============================================================================
// Weight Loading from binary file
// ============================================================================

bool load_network_params(
    const neural::WeightLoader& loader,
    NeuralNetworkParams& params,
    const neural::NetworkConfig& config
) {
    // Setup hash encoding parameters
    params.hash_encoding.n_levels = config.n_levels;
    params.hash_encoding.n_features_per_level = config.n_features_per_level;
    params.hash_encoding.log2_hashmap_size = config.log2_hashmap_size;
    params.hash_encoding.base_resolution = config.base_resolution;
    params.hash_encoding.per_level_scale = config.compute_per_level_scale();

    // Load position encoder hash table
    const neural::Tensor* pos_params = loader.get_tensor("position_encoder.params");
    if (pos_params) {
        params.hash_encoding.hash_table = pos_params->data;
        std::cout << "  Loaded position encoder: " << pos_params->data.size() << " parameters" << std::endl;
    } else {
        std::cerr << "  Warning: Could not find position_encoder.params" << std::endl;
        // Initialize with zeros
        uint32_t total_features = (1 << config.log2_hashmap_size) * config.n_levels * config.n_features_per_level;
        params.hash_encoding.hash_table.resize(total_features, 0.0f);
    }

    // Build offset table for hash grid levels
    params.hash_encoding.offset_table.resize(config.n_levels + 1);
    uint32_t offset = 0;
    for (uint32_t level = 0; level <= config.n_levels; ++level) {
        params.hash_encoding.offset_table[level] = offset;
        if (level < config.n_levels) {
            float scale = config.base_resolution * std::pow(params.hash_encoding.per_level_scale, (float)level) - 1.0f;
            uint32_t grid_resolution = (uint32_t)std::ceil(scale) + 1;
            uint32_t grid_volume = grid_resolution * grid_resolution * grid_resolution;
            uint32_t hashmap_size = 1 << config.log2_hashmap_size;
            offset += std::min(grid_volume, hashmap_size);
        }
    }

    // Load direction encoder
    const neural::Tensor* dir_params = loader.get_tensor("direction_encoder.params");
    if (dir_params && config.use_direction_encoder) {
        // Build direction encoder MLP structure
        // Input: 16 (padded from 3), Hidden: direction_hidden_dim, Output: direction_hidden_dim
        uint32_t padded_input = ((config.direction_input_dims + 15) / 16) * 16;

        // For FullyFusedMLP, parameters are packed: weights followed by biases
        // Layer sizes for n_hidden_layers + 1 output layer
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes;
        layer_sizes.push_back({padded_input, config.direction_hidden_dim});
        for (uint32_t i = 0; i < config.direction_n_hidden_layers; ++i) {
            layer_sizes.push_back({config.direction_hidden_dim, config.direction_hidden_dim});
        }

        std::cout << "  Loaded direction encoder: " << dir_params->data.size() << " parameters" << std::endl;

        // Create layers
        size_t param_offset = 0;
        for (const auto& ls : layer_sizes) {
            MLPLayer layer;
            layer.in_dim = ls.first;
            layer.out_dim = ls.second;

            size_t weight_size = ls.first * ls.second;
            size_t bias_size = ls.second;

            layer.weights.resize(weight_size);
            layer.biases.resize(bias_size);

            for (size_t i = 0; i < weight_size && (param_offset + i) < dir_params->data.size(); ++i) {
                layer.weights[i] = dir_params->data[param_offset + i];
            }
            param_offset += weight_size;

            for (size_t i = 0; i < bias_size && (param_offset + i) < dir_params->data.size(); ++i) {
                layer.biases[i] = dir_params->data[param_offset + i];
            }
            param_offset += bias_size;

            params.direction_encoder.layers.push_back(layer);
        }
        params.direction_encoder.output_activation = 'n'; // None
    }

    // Helper function to load decoder params
    auto load_decoder = [&](const std::string& name, MLPParams& mlp,
                           const neural::NetworkConfig::DecoderConfig& dec_config,
                           uint32_t input_dim) {
        const neural::Tensor* dec_params = loader.get_tensor(name + ".params");
        if (!dec_params) {
            std::cerr << "  Warning: Could not find " << name << ".params" << std::endl;
            return;
        }

        std::cout << "  Loaded " << name << ": " << dec_params->data.size() << " parameters" << std::endl;

        // Build layer structure: n_decoder_layers hidden layers + 1 output layer
        std::vector<std::pair<uint32_t, uint32_t>> layer_sizes;
        layer_sizes.push_back({input_dim, config.n_neurons});
        for (uint32_t i = 1; i < dec_config.n_decoder_layers; ++i) {
            layer_sizes.push_back({config.n_neurons, config.n_neurons});
        }
        layer_sizes.push_back({config.n_neurons, dec_config.n_output_dims});

        size_t param_offset = 0;
        for (const auto& ls : layer_sizes) {
            MLPLayer layer;
            layer.in_dim = ls.first;
            layer.out_dim = ls.second;

            size_t weight_size = ls.first * ls.second;
            size_t bias_size = ls.second;

            layer.weights.resize(weight_size, 0.0f);
            layer.biases.resize(bias_size, 0.0f);

            for (size_t i = 0; i < weight_size && (param_offset + i) < dec_params->data.size(); ++i) {
                layer.weights[i] = dec_params->data[param_offset + i];
            }
            param_offset += weight_size;

            for (size_t i = 0; i < bias_size && (param_offset + i) < dec_params->data.size(); ++i) {
                layer.biases[i] = dec_params->data[param_offset + i];
            }
            param_offset += bias_size;

            mlp.layers.push_back(layer);
        }

        // Set output activation
        if (dec_config.output_activation == "Sigmoid") {
            mlp.output_activation = 's';
        } else if (dec_config.output_activation == "ReLU") {
            mlp.output_activation = 'r';
        } else {
            mlp.output_activation = 'n';
        }
    };

    // Calculate decoder input dimension
    uint32_t decoder_input_dim = config.encoding_n_output_dims();
    if (config.use_direction_encoder) {
        decoder_input_dim += config.direction_encoder_n_output_dims();
    }

    load_decoder("visibility_decoder", params.visibility_decoder, config.visibility_decoder, decoder_input_dim);
    load_decoder("normal_decoder", params.normal_decoder, config.normal_decoder, decoder_input_dim);
    load_decoder("depth_decoder", params.depth_decoder, config.depth_decoder, decoder_input_dim);

    return true;
}

// ============================================================================
// PPM Image Writer
// ============================================================================

void write_ppm(const std::string& filename, const std::vector<uint8_t>& pixels, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), pixels.size());
    file.close();
    std::cout << "Wrote image to " << filename << std::endl;
}

// ============================================================================
// Main Test
// ============================================================================

int main(int argc, char** argv) {
    std::string weights_path = "data/models/weights.bin";
    std::string output_path = "mlp_inference_test.ppm";

    if (argc > 1) {
        weights_path = argv[1];
    }
    if (argc > 2) {
        output_path = argv[2];
    }

    std::cout << "=== MLP Inference Test ===" << std::endl;
    std::cout << "Loading weights from: " << weights_path << std::endl;

    // Load weights
    neural::WeightLoader loader;
    if (!loader.load_from_file(weights_path)) {
        std::cerr << "FAILED: Could not load weights from " << weights_path << std::endl;
        return 1;
    }

    loader.print_summary();

    // Create network config (using defaults matching training)
    neural::NetworkConfig config = neural::NetworkConfig::instant_ngp_default();
    config.print();

    // Load network parameters
    NeuralNetworkParams net_params;
    std::cout << "\nLoading network parameters..." << std::endl;
    if (!load_network_params(loader, net_params, config)) {
        std::cerr << "FAILED: Could not load network parameters" << std::endl;
        return 1;
    }

    // Generate test points on unit cube face (z=0 face, looking into +z)
    const int width = 256;
    const int height = 256;
    std::vector<uint8_t> pixels(width * height * 3);

    std::cout << "\nRunning inference on " << width << "x" << height << " grid..." << std::endl;

    // Direction pointing into the cube (+z direction)
    float dx = 0.0f, dy = 0.0f, dz = 1.0f;

    // Statistics
    float min_vis = 1.0f, max_vis = 0.0f;
    float min_depth = 1e10f, max_depth = -1e10f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Map pixel to unit cube face [0,1]x[0,1] at z=0
            float px = (float)x / (float)(width - 1);
            float py = (float)y / (float)(height - 1);
            float pz = 0.0f;

            // Run inference
            float visibility, nx, ny, nz, depth;
            neural_inference(px, py, pz, dx, dy, dz, net_params,
                           visibility, nx, ny, nz, depth);

            // Track statistics
            min_vis = std::min(min_vis, visibility);
            max_vis = std::max(max_vis, visibility);
            min_depth = std::min(min_depth, depth);
            max_depth = std::max(max_depth, depth);

            // Visualize outputs as RGB
            // R = visibility (0-1 scaled to 0-255)
            // G = (nx + 1) / 2 (normal x component mapped from [-1,1] to [0,1])
            // B = depth (normalized)
            int idx = (y * width + x) * 3;

            // Clamp visibility to [0,1]
            visibility = std::max(0.0f, std::min(1.0f, visibility));

            pixels[idx + 0] = (uint8_t)(visibility * 255.0f);  // R = visibility
            pixels[idx + 1] = (uint8_t)(((nx + 1.0f) / 2.0f) * 255.0f);  // G = normal x
            pixels[idx + 2] = (uint8_t)(((ny + 1.0f) / 2.0f) * 255.0f);  // B = normal y
        }
    }

    std::cout << "\nInference statistics:" << std::endl;
    std::cout << "  Visibility range: [" << min_vis << ", " << max_vis << "]" << std::endl;
    std::cout << "  Depth range: [" << min_depth << ", " << max_depth << "]" << std::endl;

    // Write output image
    write_ppm(output_path, pixels, width, height);

    // Also write separate images for each output
    std::vector<uint8_t> vis_pixels(width * height * 3);
    std::vector<uint8_t> norm_pixels(width * height * 3);
    std::vector<uint8_t> depth_pixels(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float px = (float)x / (float)(width - 1);
            float py = (float)y / (float)(height - 1);
            float pz = 0.0f;

            float visibility, nx, ny, nz, depth;
            neural_inference(px, py, pz, dx, dy, dz, net_params,
                           visibility, nx, ny, nz, depth);

            int idx = (y * width + x) * 3;

            // Visibility image (grayscale)
            uint8_t v = (uint8_t)(std::max(0.0f, std::min(1.0f, visibility)) * 255.0f);
            vis_pixels[idx + 0] = v;
            vis_pixels[idx + 1] = v;
            vis_pixels[idx + 2] = v;

            // Normal image (RGB = XYZ mapped from [-1,1] to [0,255])
            norm_pixels[idx + 0] = (uint8_t)(((nx + 1.0f) / 2.0f) * 255.0f);
            norm_pixels[idx + 1] = (uint8_t)(((ny + 1.0f) / 2.0f) * 255.0f);
            norm_pixels[idx + 2] = (uint8_t)(((nz + 1.0f) / 2.0f) * 255.0f);

            // Depth image (normalize to [0,1] based on range)
            float depth_norm = (max_depth > min_depth) ?
                              (depth - min_depth) / (max_depth - min_depth) : 0.5f;
            depth_norm = std::max(0.0f, std::min(1.0f, depth_norm));
            uint8_t d = (uint8_t)(depth_norm * 255.0f);
            depth_pixels[idx + 0] = d;
            depth_pixels[idx + 1] = d;
            depth_pixels[idx + 2] = d;
        }
    }

    // Write separate output images
    std::string base_path = output_path.substr(0, output_path.find_last_of('.'));
    write_ppm(base_path + "_visibility.ppm", vis_pixels, width, height);
    write_ppm(base_path + "_normal.ppm", norm_pixels, width, height);
    write_ppm(base_path + "_depth.ppm", depth_pixels, width, height);

    std::cout << "\n=== MLP Inference Test PASSED ===" << std::endl;
    return 0;
}
