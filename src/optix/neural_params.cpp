#include "neural_params.h"
#include "../utils/error.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <limits>

namespace optix {

// Helper functions matching tiny-cuda-nn logic
static uint32_t next_multiple(uint32_t val, uint32_t divisor) {
    return ((val + divisor - 1) / divisor) * divisor;
}

static uint32_t powi(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

static float grid_scale(uint32_t level, float log2_per_level_scale, float base_resolution) {
    return std::exp2(level * log2_per_level_scale) * base_resolution - 1.0f;
}

static uint32_t grid_resolution(float scale) {
    return (uint32_t)std::ceil(scale) + 1;
}

NeuralNetworkParamsHost::NeuralNetworkParamsHost(const neural::NetworkConfig& config)
    : config_(config) {
    // Initialize device params to zero
    std::memset(&d_params_, 0, sizeof(NeuralNetworkParams));

    // Allocate activation strings on device
    d_relu_str_ = allocate_activation_string("ReLU");
    d_sigmoid_str_ = allocate_activation_string("Sigmoid");
    d_none_str_ = allocate_activation_string("None");
}

NeuralNetworkParamsHost::~NeuralNetworkParamsHost() {
    free_device_memory();
}

char* NeuralNetworkParamsHost::allocate_activation_string(const std::string& str) {
    char* d_str = nullptr;
    size_t len = str.length() + 1;
    CUDA_CHECK(cudaMalloc(&d_str, len));
    CUDA_CHECK(cudaMemcpy(d_str, str.c_str(), len, cudaMemcpyHostToDevice));
    return d_str;
}

void NeuralNetworkParamsHost::free_device_memory() {
    // Free hash grid
    if (d_hash_table_) cudaFree(d_hash_table_);
    if (d_hash_offsets_) cudaFree(d_hash_offsets_);

    // Helper lambda to free MLP layers
    auto free_mlp = [](MLPLayer* d_layers, uint32_t n_layers) {
        if (!d_layers) return;

        // First, copy layer pointers back to host to free individual allocations
        std::vector<MLPLayer> h_layers(n_layers);
        cudaMemcpy(h_layers.data(), d_layers, n_layers * sizeof(MLPLayer), cudaMemcpyDeviceToHost);

        for (uint32_t i = 0; i < n_layers; ++i) {
            if (h_layers[i].weights) cudaFree(h_layers[i].weights);
            if (h_layers[i].biases) cudaFree(h_layers[i].biases);  // Safe even if nullptr
        }

        cudaFree(d_layers);
    };

    free_mlp(d_dir_encoder_layers_, dir_encoder_n_layers_);
    free_mlp(d_vis_decoder_layers_, vis_decoder_n_layers_);
    free_mlp(d_norm_decoder_layers_, norm_decoder_n_layers_);
    free_mlp(d_depth_decoder_layers_, depth_decoder_n_layers_);

    // Free activation strings
    if (d_relu_str_) cudaFree(d_relu_str_);
    if (d_sigmoid_str_) cudaFree(d_sigmoid_str_);
    if (d_none_str_) cudaFree(d_none_str_);

    loaded_ = false;
}

bool NeuralNetworkParamsHost::load_from_weights(const neural::WeightLoader& loader) {
    if (!loader.is_loaded()) {
        std::cerr << "WeightLoader has no weights loaded" << std::endl;
        return false;
    }

    std::cout << "\n=== Converting Weights for OptiX ===" << std::endl;

    // Load hash encoding
    if (!load_hash_encoding(loader)) {
        std::cerr << "Failed to load hash encoding weights" << std::endl;
        return false;
    }

    // Calculate layer dimensions
    uint32_t pos_encoding_dim = config_.encoding_n_output_dims();  // 32
    uint32_t dir_encoding_dim = config_.direction_encoder_n_output_dims();  // 16
    uint32_t decoder_input_dim = pos_encoding_dim + dir_encoding_dim;  // 48

    // Load direction encoder (3 padded to 16 -> 16D hidden -> 16D output)
    if (!load_mlp(
        loader,
        "direction_encoder",
        config_.direction_n_hidden_layers,  // hidden + output layer
        config_.direction_hidden_dim,  // 16
        16,  // padded input dimension
        config_.direction_hidden_dim,  // 16
        "None",
        d_dir_encoder_layers_,
        dir_encoder_n_layers_
    )) {
        std::cerr << "Failed to load direction encoder weights" << std::endl;
        return false;
    }

    // Load visibility decoder (48D -> 32D -> 32D -> 32D -> 1D)
    if (!load_mlp(
        loader,
        "visibility_decoder",
        config_.visibility_decoder.n_decoder_layers,  // 3 hidden + 1 output = 4 total
        config_.n_neurons,  // 32
        decoder_input_dim,  // 48
        config_.visibility_decoder.n_output_dims,  // 1
        config_.visibility_decoder.output_activation,  // "Sigmoid"
        d_vis_decoder_layers_,
        vis_decoder_n_layers_
    )) {
        std::cerr << "Failed to load visibility decoder weights" << std::endl;
        return false;
    }

    // Load normal decoder (48D -> 32D -> 32D -> 32D -> 3D)
    if (!load_mlp(
        loader,
        "normal_decoder",
        config_.normal_decoder.n_decoder_layers,  // 3 hidden + 1 output = 4 total
        config_.n_neurons,  // 32
        decoder_input_dim,  // 48
        config_.normal_decoder.n_output_dims,  // 3
        config_.normal_decoder.output_activation,  // "None"
        d_norm_decoder_layers_,
        norm_decoder_n_layers_
    )) {
        std::cerr << "Failed to load normal decoder weights" << std::endl;
        return false;
    }

    // Load depth decoder (48D -> 32D -> 32D -> 32D -> 1D)
    if (!load_mlp(
        loader,
        "depth_decoder",
        config_.depth_decoder.n_decoder_layers,  // 3 hidden + 1 output = 4 total
        config_.n_neurons,  // 32
        decoder_input_dim,  // 48
        config_.depth_decoder.n_output_dims,  // 1
        config_.depth_decoder.output_activation,  // "None"
        d_depth_decoder_layers_,
        depth_decoder_n_layers_
    )) {
        std::cerr << "Failed to load depth decoder weights" << std::endl;
        return false;
    }

    // Setup MLPParams structures
    d_params_.direction_encoder.layers = d_dir_encoder_layers_;
    d_params_.direction_encoder.n_layers = dir_encoder_n_layers_;
    d_params_.direction_encoder.output_activation = d_none_str_;

    d_params_.visibility_decoder.layers = d_vis_decoder_layers_;
    d_params_.visibility_decoder.n_layers = vis_decoder_n_layers_;
    if (config_.visibility_decoder.output_activation[0] == 'R' || config_.visibility_decoder.output_activation[0] == 'r')
        d_params_.visibility_decoder.output_activation = d_relu_str_;
    else if (config_.visibility_decoder.output_activation[0] == 'S' || config_.visibility_decoder.output_activation[0] == 's')
        d_params_.visibility_decoder.output_activation = d_sigmoid_str_;
    else
        d_params_.visibility_decoder.output_activation = d_none_str_;

    d_params_.normal_decoder.layers = d_norm_decoder_layers_;
    d_params_.normal_decoder.n_layers = norm_decoder_n_layers_;
    d_params_.normal_decoder.output_activation = d_none_str_;

    d_params_.depth_decoder.layers = d_depth_decoder_layers_;
    d_params_.depth_decoder.n_layers = depth_decoder_n_layers_;
    d_params_.depth_decoder.output_activation = d_none_str_;

    loaded_ = true;
    std::cout << "===================================\n" << std::endl;
    return true;
}

bool NeuralNetworkParamsHost::load_hash_encoding(const neural::WeightLoader& loader) {
    std::cout << "Loading hash encoding weights..." << std::endl;

    // Get hash table from loader
    const neural::Tensor* hash_tensor = loader.get_tensor("position_encoder.params");
    if (!hash_tensor) {
        std::cerr << "  Error: Could not find position_encoder.params" << std::endl;
        return false;
    }

    std::cout << "  Hash table total size: " << hash_tensor->data.size() << " elements" << std::endl;

    // Calculate offset table using tiny-cuda-nn's logic
    // This matches the GridEncodingTemplated constructor
    constexpr uint32_t N_POS_DIMS = 3;  // 3D positions
    constexpr uint32_t N_FEATURES_PER_LEVEL = 2;  // Matches config

    std::vector<uint32_t> h_offsets(config_.n_levels + 1);  // +1 for total size at end
    uint32_t offset = 0;

    float log2_per_level_scale = std::log2(config_.compute_per_level_scale());

    for (uint32_t i = 0; i < config_.n_levels; ++i) {
        // Compute resolution at this level
        float scale = grid_scale(i, log2_per_level_scale, config_.base_resolution);
        uint32_t resolution = grid_resolution(scale);

        // Compute number of params at this level (before hashing)
        uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
        float dense_params = std::pow((float)resolution, (float)N_POS_DIMS);
        uint32_t params_in_level = (dense_params > (float)max_params) ? max_params : powi(resolution, N_POS_DIMS);

        // Align to 8 elements
        params_in_level = next_multiple(params_in_level, 8u);

        // For hash grid: use min of dense params or hash table size
        uint32_t hashmap_size = 1u << config_.log2_hashmap_size;
        params_in_level = std::min(params_in_level, hashmap_size);

        h_offsets[i] = offset;
        offset += params_in_level;

        std::cout << "  Level " << i << ": resolution=" << resolution
                  << ", params_in_level=" << params_in_level
                  << ", offset=" << h_offsets[i] << std::endl;
    }

    h_offsets[config_.n_levels] = offset;
    uint32_t total_params = offset * N_FEATURES_PER_LEVEL;

    std::cout << "  Total params (features): " << total_params
              << " (expected: " << hash_tensor->data.size() << ")" << std::endl;

    if (total_params != hash_tensor->data.size()) {
        std::cerr << "  Warning: Parameter count mismatch!" << std::endl;
        std::cerr << "    Calculated: " << total_params << std::endl;
        std::cerr << "    Loaded: " << hash_tensor->data.size() << std::endl;
    }

    // Allocate device memory for hash table
    hash_table_size_ = hash_tensor->data.size();
    CUDA_CHECK(cudaMalloc(&d_hash_table_, hash_table_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(
        d_hash_table_,
        hash_tensor->data.data(),
        hash_table_size_ * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    // Allocate device memory for offset table
    CUDA_CHECK(cudaMalloc(&d_hash_offsets_, (config_.n_levels + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(
        d_hash_offsets_,
        h_offsets.data(),
        (config_.n_levels + 1) * sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));

    // Setup HashGridParams
    d_params_.hash_encoding.hash_table = d_hash_table_;
    d_params_.hash_encoding.offset_table = d_hash_offsets_;
    d_params_.hash_encoding.n_levels = config_.n_levels;
    d_params_.hash_encoding.n_features_per_level = config_.n_features_per_level;
    d_params_.hash_encoding.log2_hashmap_size = config_.log2_hashmap_size;
    d_params_.hash_encoding.base_resolution = config_.base_resolution;
    d_params_.hash_encoding.per_level_scale = config_.compute_per_level_scale();

    std::cout << "  Hash encoding loaded successfully" << std::endl;
    std::cout << "    Levels: " << config_.n_levels << std::endl;
    std::cout << "    Features per level: " << config_.n_features_per_level << std::endl;
    std::cout << "    Log2 hashmap size: " << config_.log2_hashmap_size << std::endl;
    std::cout << "    Base resolution: " << config_.base_resolution << std::endl;
    std::cout << "    Per-level scale: " << config_.compute_per_level_scale() << std::endl;
    std::cout << "    Hash offsets" << std::endl;
    for (int i = 0; i < config_.n_levels; i++) {
        std::cout << "        Level: " << i << " - " << h_offsets[i] << std::endl;
    }
    return true;
}

bool NeuralNetworkParamsHost::load_mlp(
    const neural::WeightLoader& loader,
    const std::string& prefix,
    uint32_t n_layers,
    uint32_t hidden_dim,
    uint32_t input_dim,
    uint32_t output_dim,
    const std::string& output_activation,
    MLPLayer*& d_layers_out,
    uint32_t& n_layers_out
) {
    std::cout << "Loading MLP weights for " << prefix << "..." << std::endl;
    std::cout << "  Layers: " << n_layers << std::endl;

    // Get flattened params from tiny-cuda-nn
    const neural::Tensor* params_tensor = loader.get_tensor(prefix + ".params");
    if (!params_tensor) {
        std::cerr << "  Error: Could not find " << prefix << ".params" << std::endl;
        return false;
    }

    std::cout << "  Total parameters: " << params_tensor->data.size() << std::endl;

    // Allocate host-side layer structures
    std::vector<MLPLayer> h_layers(n_layers+1);

    // Parse parameters into layers
    // FullyFusedMLP stores only weights (no biases): [layer0_weights, layer1_weights, ...]
    size_t offset = 0;

    uint32_t padded_output_dim = (uint32_t)((output_dim + 15) / 16) * 16;  // Align output dim to 16 for FullyFusedMLP

    for (uint32_t l = 0; l < n_layers+1; ++l) {
        uint32_t layer_in_dim = (l == 0) ? input_dim : hidden_dim;
        uint32_t layer_out_dim = (l == n_layers) ? padded_output_dim : hidden_dim;

        h_layers[l].in_dim = layer_in_dim;
        h_layers[l].out_dim = layer_out_dim;

        // Calculate sizes (FullyFusedMLP has no bias)
        size_t weight_size = layer_out_dim * layer_in_dim;

        std::cout << "  Layer " << l << ": " << layer_in_dim << " -> " << layer_out_dim << std::endl;
        std::cout << "    Weights: " << weight_size << " (no bias for FullyFusedMLP)" << std::endl;

        // Check bounds
        if (offset + weight_size > params_tensor->data.size()) {
            std::cerr << "  Error: Parameter size mismatch at layer " << l << std::endl;
            std::cerr << "    Expected offset + " << weight_size
                      << " <= " << params_tensor->data.size() << std::endl;
            return false;
        }

        // Allocate device memory for weights only
        CUDA_CHECK(cudaMalloc(&h_layers[l].weights, weight_size * sizeof(float)));
        h_layers[l].biases = nullptr;  // No bias for FullyFusedMLP

        // Copy weights directly without transposing for testing
        CUDA_CHECK(cudaMemcpy(
            h_layers[l].weights,
            params_tensor->data.data() + offset,
            weight_size * sizeof(float),
            cudaMemcpyHostToDevice
        ));
        offset += weight_size;
    }
    std::cout << "Output activation: " << output_activation << std::endl;

    std::cout << "  Used " << offset << " / " << params_tensor->data.size() << " parameters" << std::endl;

    // Allocate device memory for layer array (n_layers+1 because we have n hidden + 1 output)
    CUDA_CHECK(cudaMalloc(&d_layers_out, (n_layers+1) * sizeof(MLPLayer)));
    CUDA_CHECK(cudaMemcpy(
        d_layers_out,
        h_layers.data(),
        (n_layers+1) * sizeof(MLPLayer),
        cudaMemcpyHostToDevice
    ));

    n_layers_out = n_layers + 1;

    // Checkpoint: Print first 10 weights from Layer 0 after loading
    if (prefix == "direction_encoder") {
        std::cout << "\n  === Checkpoint 0: Direction Encoder Weights (after host load) ===" << std::endl;
        std::cout << "  Layer 0: in_dim=" << h_layers[0].in_dim << ", out_dim=" << h_layers[0].out_dim << std::endl;
        std::vector<float> h_weights(10);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), h_layers[0].weights, 10 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 10; ++i) {
            std::cout << "    weights[" << i << "] = " << h_weights[i] << std::endl;
        }
    }
    else if (prefix == "visibility_decoder") {
        std::cout << "\n  === Checkpoint 0: Visibility Decoder Weights (after host load) ===" << std::endl;
        std::cout << "  Layer 0: in_dim=" << h_layers[0].in_dim << ", out_dim=" << h_layers[0].out_dim << std::endl;
        std::vector<float> h_weights(10);
        CUDA_CHECK(cudaMemcpy(h_weights.data(), h_layers[0].weights, 10 * sizeof(float), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 10; ++i) {
            std::cout << "    weights[" << i << "] = " << h_weights[i] << std::endl;
        }
    }

    std::cout << "  " << prefix << " loaded successfully" << std::endl;
    return true;
}

} // namespace optix
