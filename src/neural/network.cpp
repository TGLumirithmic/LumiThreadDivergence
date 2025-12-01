#include "network.h"
#include "network_kernels.h"
#include "../utils/error.h"
#include "../utils/cuda_utils.h"
#include <iostream>
#include <vector>

// Include tiny-cuda-nn headers (includes CUDA headers including half precision)
#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/config.h>
#include <tiny-cuda-nn/encoding.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/gpu_memory.h>

namespace neural {

using namespace tcnn;
// GTX 980 (sm_52) doesn't support half precision in tiny-cuda-nn
// Use float precision for compatibility
using precision_t = float;

NeuralNetwork::NeuralNetwork(const NetworkConfig& config)
    : config_(config), initialized_(false) {
}

NeuralNetwork::~NeuralNetwork() {
    free_buffers();

    // Clean up tiny-cuda-nn objects
    if (encoding_) {
        delete reinterpret_cast<Encoding<precision_t>*>(encoding_);
    }
    if (direction_encoder_) {
        delete reinterpret_cast<Network<precision_t>*>(direction_encoder_);
    }
    if (visibility_network_) {
        delete reinterpret_cast<Network<precision_t>*>(visibility_network_);
    }
    if (normal_network_) {
        delete reinterpret_cast<Network<precision_t>*>(normal_network_);
    }
    if (depth_network_) {
        delete reinterpret_cast<Network<precision_t>*>(depth_network_);
    }
}

bool NeuralNetwork::initialize_from_weights(const WeightLoader& loader) {
    if (!loader.is_loaded()) {
        std::cerr << "WeightLoader has no weights loaded" << std::endl;
        return false;
    }

    std::cout << "\n=== Initializing Neural Network ===" << std::endl;
    config_.print();

    try {
        // Create hash encoding configuration
        nlohmann::json encoding_config = {
            {"otype", "HashGrid"},
            {"n_dims_to_encode", config_.n_input_dims},
            {"n_levels", config_.n_levels},
            {"n_features_per_level", config_.n_features_per_level},
            {"log2_hashmap_size", config_.log2_hashmap_size},
            {"base_resolution", config_.base_resolution},
            {"per_level_scale", config_.compute_per_level_scale()}
        };

        std::cout << "\nCreating hash encoding..." << std::endl;
        encoding_ = create_encoding<precision_t>(config_.n_input_dims, encoding_config);

        uint32_t encoding_output_dims = config_.encoding_n_output_dims();
        std::cout << "Encoding output dimensions: " << encoding_output_dims << std::endl;

        // Create direction encoder if enabled
        uint32_t direction_output_dims = 0;
        if (config_.use_direction_encoder) {
            std::cout << "\nCreating direction encoder..." << std::endl;

            // Pad input dimension to multiple of 16 for tensor cores (matching training)
            uint32_t padded_input_dims = ((config_.direction_input_dims + 15) / 16) * 16;

            nlohmann::json dir_encoder_config = {
                {"otype", "CutlassMLP"},
                {"activation", config_.activation},
                {"n_neurons", config_.direction_hidden_dim},
                {"n_hidden_layers", config_.direction_n_hidden_layers},
                {"n_input_dims", padded_input_dims},  // Use padded dimension
                {"n_output_dims", config_.direction_hidden_dim},
                {"output_activation", "None"}
            };
            direction_encoder_ = create_network<precision_t>(dir_encoder_config);
            if (!direction_encoder_) {
                std::cerr << "  ERROR: Failed to create direction encoder!" << std::endl;
                return false;
            }
            direction_output_dims = config_.direction_encoder_n_output_dims();
            std::cout << "  Direction encoder created (input: " << config_.direction_input_dims
                      << "D padded to " << padded_input_dims << "D, output: " << direction_output_dims << "D)" << std::endl;
            std::cout << "Created network type: " << reinterpret_cast<Network<precision_t>*>(direction_encoder_)->name() << std::endl;
        }

        // Calculate total decoder input dimension
        uint32_t decoder_input_dims = encoding_output_dims + direction_output_dims;
        std::cout << "\nTotal decoder input dimensions: " << decoder_input_dims
                  << " (" << encoding_output_dims << " position + "
                  << direction_output_dims << " direction)" << std::endl;

        // Create three decoder networks
        std::cout << "\nCreating decoder networks..." << std::endl;

        // Visibility decoder (1D output with sigmoid)
        nlohmann::json vis_config = {
            {"otype", "CutlassMLP"},
            {"activation", config_.activation},
            {"n_neurons", config_.n_neurons},
            {"n_hidden_layers", config_.visibility_decoder.n_decoder_layers},
            {"n_input_dims", decoder_input_dims},
            {"n_output_dims", config_.visibility_decoder.n_output_dims},
            {"output_activation", config_.visibility_decoder.output_activation}
        };
        visibility_network_ = create_network<precision_t>(vis_config);
        std::cout << "  Visibility decoder created ("
                  << config_.visibility_decoder.n_decoder_layers << " hidden + 1 output = "
                  << (config_.visibility_decoder.n_decoder_layers + 1) << " layers, "
                  << config_.visibility_decoder.n_output_dims << "D output)" << std::endl;
        std::vector<std::pair<uint32_t, uint32_t>> layer_size_vector = reinterpret_cast<Network<precision_t>*>(visibility_network_)->layer_sizes();
        for (auto it = layer_size_vector.begin(); it != layer_size_vector.end(); it++) {
            std::cout << it->first << ", " << it->second << std::endl;
        }

        // Normal decoder (3D output)
        nlohmann::json norm_config = {
            {"otype", "CutlassMLP"},
            {"activation", config_.activation},
            {"n_neurons", config_.n_neurons},
            {"n_hidden_layers", config_.normal_decoder.n_decoder_layers},
            {"n_input_dims", decoder_input_dims},
            {"n_output_dims", config_.normal_decoder.n_output_dims},
            {"output_activation", config_.normal_decoder.output_activation}
        };
        normal_network_ = create_network<precision_t>(norm_config);
        std::cout << "  Normal decoder created ("
                  << config_.normal_decoder.n_decoder_layers << " hidden + 1 output = "
                  << (config_.normal_decoder.n_decoder_layers + 1) << " layers, "
                  << config_.normal_decoder.n_output_dims << "D output)" << std::endl;

        // Depth decoder (1D output)
        nlohmann::json depth_config = {
            {"otype", "CutlassMLP"},
            {"activation", config_.activation},
            {"n_neurons", config_.n_neurons},
            {"n_hidden_layers", config_.depth_decoder.n_decoder_layers},
            {"n_input_dims", decoder_input_dims},
            {"n_output_dims", config_.depth_decoder.n_output_dims},
            {"output_activation", config_.depth_decoder.output_activation}
        };
        depth_network_ = create_network<precision_t>(depth_config);
        std::cout << "  Depth decoder created ("
                  << config_.depth_decoder.n_decoder_layers << " hidden + 1 output = "
                  << (config_.depth_decoder.n_decoder_layers + 1) << " layers, "
                  << config_.depth_decoder.n_output_dims << "D output)" << std::endl;

        // Load weights into networks
        std::cout << "\nLoading weights into networks..." << std::endl;

        bool weights_loaded = true;

        // Load position encoder weights
        weights_loaded &= load_position_encoder_weights(loader);

        // Load direction encoder weights if enabled
        if (config_.use_direction_encoder) {
            weights_loaded &= load_direction_encoder_weights(loader);
        }

        weights_loaded &= load_network_weights(
            visibility_network_,
            loader,
            "visibility_decoder",
            config_.visibility_decoder
        );

        weights_loaded &= load_network_weights(
            normal_network_,
            loader,
            "normal_decoder",
            config_.normal_decoder
        );

        weights_loaded &= load_network_weights(
            depth_network_,
            loader,
            "depth_decoder",
            config_.depth_decoder
        );

        if (!weights_loaded) {
            std::cerr << "Warning: Some weights failed to load. Using random initialization for missing weights." << std::endl;
        } else {
            std::cout << "  All weights loaded successfully!" << std::endl;
        }

        initialized_ = true;
        std::cout << "\nNetwork initialization complete!" << std::endl;
        std::cout << "===================================\n" << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Error initializing network: " << e.what() << std::endl;
        return false;
    }
}

void NeuralNetwork::allocate_buffers(uint32_t batch_size) {
    // Allocate position encoding buffer (float)
    size_t pos_encoding_size = batch_size * config_.encoding_n_output_dims();
    if (d_encoded_ == nullptr || pos_encoding_size > encoded_buffer_size_) {
        if (d_encoded_) cuda_utils::free_device(d_encoded_);
        d_encoded_ = cuda_utils::allocate_device<float>(pos_encoding_size);
        encoded_buffer_size_ = pos_encoding_size;
    }

    // Allocate direction buffers if direction encoder is enabled
    if (config_.use_direction_encoder) {
        // Direction input buffer (half precision for network input, padded to multiple of 16)
        uint32_t padded_dir_input_dims = ((config_.direction_input_dims + 15) / 16) * 16;
        size_t dir_input_size = batch_size * padded_dir_input_dims;
        if (d_dir_input_float_ == nullptr || dir_input_size > dir_input_float_buffer_size_) {
            if (d_dir_input_float_) cuda_utils::free_device(d_dir_input_float_);
            d_dir_input_float_ = cuda_utils::allocate_device<float>(dir_input_size);
            dir_input_float_buffer_size_ = dir_input_size;
        }

        // Direction encoding output buffer (float)
        size_t dir_encoding_size = batch_size * config_.direction_encoder_n_output_dims();
        if (d_dir_encoded_ == nullptr || dir_encoding_size > dir_encoded_buffer_size_) {
            if (d_dir_encoded_) cuda_utils::free_device(d_dir_encoded_);
            d_dir_encoded_ = cuda_utils::allocate_device<float>(dir_encoding_size);
            dir_encoded_buffer_size_ = dir_encoding_size;
        }
    }

    // Allocate concatenated buffer for decoder input (half precision)
    size_t concat_half_size = batch_size * config_.decoder_input_dims();
    if (d_concatenated_float_ == nullptr || concat_half_size > concatenated_float_buffer_size_) {
        if (d_concatenated_float_) cuda_utils::free_device(d_concatenated_float_);
        d_concatenated_float_ = cuda_utils::allocate_device<float>(concat_half_size);
        concatenated_float_buffer_size_ = concat_half_size;
    }
}

void NeuralNetwork::free_buffers() {
    if (d_encoded_) {
        cuda_utils::free_device(d_encoded_);
        d_encoded_ = nullptr;
        encoded_buffer_size_ = 0;
    }
    if (d_dir_input_float_) {
        cuda_utils::free_device(d_dir_input_float_);
        d_dir_input_float_ = nullptr;
        dir_input_float_buffer_size_ = 0;
    }
    if (d_dir_encoded_) {
        cuda_utils::free_device(d_dir_encoded_);
        d_dir_encoded_ = nullptr;
        dir_encoded_buffer_size_ = 0;
    }
    if (d_concatenated_float_) {
        cuda_utils::free_device(d_concatenated_float_);
        d_concatenated_float_ = nullptr;
        concatenated_float_buffer_size_ = 0;
    }
}

void NeuralNetwork::inference(
    const float* d_positions,
    const float* d_directions,
    float* d_visibility,
    float* d_normals,
    float* d_depth,
    uint32_t batch_size,
    cudaStream_t stream
) {
    if (!initialized_) {
        throw std::runtime_error("Network not initialized");
    }

    if (batch_size % BATCH_SIZE_GRANULARITY != 0) {
        throw std::runtime_error("Batch size must be a multiple of " + std::to_string(BATCH_SIZE_GRANULARITY));
    }

    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] At entry " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Failed at inference start");
    }

    std::cout << "[INFO] Starting inference, batch_size=" << batch_size << std::endl;

    // Allocate encoding buffers if needed
    allocate_buffers(batch_size);
 
    std::cout << "[DEBUG] Allocated buffers: "
              << "d_encoded_=" << encoded_buffer_size_ 
              << ", d_dir_input_float_=" << dir_input_float_buffer_size_
              << ", d_dir_encoded_=" << dir_encoded_buffer_size_
              << ", d_concatenated_float_=" << concatenated_float_buffer_size_
              << std::endl;

    auto* encoding = reinterpret_cast<Encoding<precision_t>*>(encoding_);
    auto* vis_network = reinterpret_cast<Network<precision_t>*>(visibility_network_);
    auto* norm_network = reinterpret_cast<Network<precision_t>*>(normal_network_);
    auto* depth_network = reinterpret_cast<Network<precision_t>*>(depth_network_);

    encoding->set_jit_fusion(false);

    uint32_t pos_encoding_dims = config_.encoding_n_output_dims();
    uint32_t decoder_input_dims = config_.decoder_input_dims();

    // Step 1: Encode positions
    std::cout << "[INFO] Step 1: Position encoding, n_input_dims=" << config_.n_input_dims
            << ", pos_encoding_dims=" << pos_encoding_dims
            << ", batch_size=" << batch_size << std::endl;

    // Allocate temporary buffer for positions (same size)
    float* d_positions_tmp = nullptr;
    size_t positions_bytes = batch_size * config_.n_input_dims * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_positions_tmp, positions_bytes));

    // Try copying d_positions to d_positions_tmp (device-to-device)
    std::cout << "[DEBUG] Copying d_positions to temporary buffer..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_positions_tmp, d_positions, positions_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[DEBUG] Copy succeeded" << std::endl;

    // Create GPUMatrixDynamic objects using the temporary buffer
    GPUMatrixDynamic<float> pos_input(d_positions_tmp, config_.n_input_dims, batch_size);
    GPUMatrixDynamic<float> pos_output(d_encoded_, pos_encoding_dims, batch_size);

    // Run the encoder
    std::cout << "[DEBUG] Running encoding->inference..." << std::endl;
    encoding->inference_mixed_precision_impl(stream, pos_input, pos_output, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[DEBUG] Encoding inference completed" << std::endl;

    // Free temporary buffer
    CUDA_CHECK(cudaFree(d_positions_tmp));


    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] After position encoding: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Position encoding failed");
    }

    // Step 2: Encode directions if enabled
    if (config_.use_direction_encoder && direction_encoder_) {
        auto* dir_encoder = reinterpret_cast<Network<precision_t>*>(direction_encoder_);
        uint32_t dir_encoding_dims = config_.direction_encoder_n_output_dims();
        uint32_t padded_dir_input_dims = ((config_.direction_input_dims + 15) / 16) * 16;

        std::cout << "[INFO] Step 2: Pad directions, padded_dir_input_dims=" << padded_dir_input_dims
                  << ", batch_size=" << batch_size << std::endl;

        uint32_t threads = 256;
        uint32_t blocks = (batch_size + threads - 1) / threads;
        pad_direction_kernel_float<<<blocks, threads, 0, stream>>>(
            d_directions,
            d_dir_input_float_,
            batch_size,
            config_.direction_input_dims,
            padded_dir_input_dims
        );
        if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] After pad_direction_kernel: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Pad direction kernel failed");
        }

        std::cout << "[INFO] Step 2b: Direction encoder inference" << std::endl;
        GPUMatrixDynamic<precision_t> dir_input(d_dir_input_float_, padded_dir_input_dims, batch_size);
        GPUMatrixDynamic<float> dir_output(d_dir_encoded_, dir_encoding_dims, batch_size);
        dir_encoder->inference(stream, dir_input, dir_output);

        if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] After direction encoder inference: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Direction encoder inference failed");
        }

        // Step 3: Concatenate position + direction encodings
        std::cout << "[INFO] Step 3: Concatenate encodings, pos=" << pos_encoding_dims
                  << ", dir=" << dir_encoding_dims << std::endl;

        concatenate_encodings_kernel_float<<<blocks, threads, 0, stream>>>(
            d_encoded_,
            d_dir_encoded_,
            d_concatenated_float_,
            batch_size,
            pos_encoding_dims,
            dir_encoding_dims
        );
        if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] After concatenate_encodings_kernel: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Concatenate encodings kernel failed");
        }
    } else {
        // No direction encoder, just copy position encoding
        std::cout << "[INFO] Step 2: No direction encoder, copying position encoding" << std::endl;
        CUDA_CHECK(cudaMemcpyAsync(
            d_concatenated_float_,
            d_encoded_,
            batch_size * pos_encoding_dims * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ));
        if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "[CUDA ERROR] After memcpy of position encoding: " << cudaGetErrorString(err) << std::endl;
            throw std::runtime_error("Memcpy of position encoding failed");
        }
    }

    // Step 4: Run decoders
    std::cout << "[INFO] Step 4: Decoder inference" << std::endl;

    GPUMatrixDynamic<precision_t> decoder_in(d_concatenated_float_, decoder_input_dims, batch_size);
    GPUMatrixDynamic<float> vis_output(d_visibility, config_.visibility_decoder.n_output_dims, batch_size);
    GPUMatrixDynamic<float> norm_output(d_normals, config_.normal_decoder.n_output_dims, batch_size);
    GPUMatrixDynamic<float> depth_output(d_depth, config_.depth_decoder.n_output_dims, batch_size);

    vis_network->inference(stream, decoder_in, vis_output);
    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] After visibility decoder: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Visibility decoder failed");
    }

    norm_network->inference(stream, decoder_in, norm_output);
    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] After normal decoder: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Normal decoder failed");
    }

    depth_network->inference(stream, decoder_in, depth_output);
    if (!stream) CUDA_CHECK(cudaDeviceSynchronize());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] After depth decoder: " << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("Depth decoder failed");
    }

    std::cout << "[INFO] Inference complete for batch_size=" << batch_size << std::endl;
}


void NeuralNetwork::inference_single(
    const float* position,
    const float* direction,
    float& visibility,
    float* normal,
    float& depth
) {
    // tiny-cuda-nn requires batch sizes to be multiples of 128
    // So we pad to 128 and only use the first result
    constexpr uint32_t BATCH_SIZE = 128;

    // Allocate device memory for padded batch
    float* d_pos = cuda_utils::allocate_device<float>(BATCH_SIZE * 3);
    float* d_dir = cuda_utils::allocate_device<float>(BATCH_SIZE * 3);
    float* d_vis = cuda_utils::allocate_device<float>(BATCH_SIZE);
    float* d_norm = cuda_utils::allocate_device<float>(BATCH_SIZE * 3);
    float* d_dep = cuda_utils::allocate_device<float>(BATCH_SIZE);

    // Initialize buffers with zeros
    CUDA_CHECK(cudaMemset(d_pos, 0, BATCH_SIZE * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dir, 0, BATCH_SIZE * 3 * sizeof(float)));

    // Copy single input to first element of batch
    cuda_utils::copy_to_device(d_pos, position, 3);
    cuda_utils::copy_to_device(d_dir, direction, 3);

    // Run inference on padded batch
    inference(d_pos, d_dir, d_vis, d_norm, d_dep, BATCH_SIZE);

    // Copy only the first result back to host
    cuda_utils::copy_to_host(&visibility, d_vis, 1);
    cuda_utils::copy_to_host(normal, d_norm, 3);
    cuda_utils::copy_to_host(&depth, d_dep, 1);

    // Cleanup
    cuda_utils::free_device(d_pos);
    cuda_utils::free_device(d_dir);
    cuda_utils::free_device(d_vis);
    cuda_utils::free_device(d_norm);
    cuda_utils::free_device(d_dep);
}

void* NeuralNetwork::get_encoding_device_ptr() const {
    return encoding_;
}

void* NeuralNetwork::get_visibility_network_ptr() const {
    return visibility_network_;
}

void* NeuralNetwork::get_normal_network_ptr() const {
    return normal_network_;
}

void* NeuralNetwork::get_depth_network_ptr() const {
    return depth_network_;
}

bool NeuralNetwork::load_position_encoder_weights(const WeightLoader& loader) {
    if (!encoding_) {
        std::cerr << "  Error: Position encoder is null" << std::endl;
        return false;
    }

    auto* encoding = reinterpret_cast<Encoding<precision_t>*>(encoding_);
    std::cout << "  Loading weights for position_encoder..." << std::endl;

    try {
        auto params = encoding->params();
        size_t n_params = encoding->n_params();

        std::cout << "    Encoding expects " << n_params << " parameters" << std::endl;

        // Load parameters as a single flat chunk
        std::string weight_name = "position_encoder.params";
        const Tensor* weight_tensor = loader.get_tensor(weight_name);

        if (!weight_tensor) {
            std::cerr << "    Error: Could not find tensor " << weight_name << std::endl;
            return false;
        }

        std::cout << "    Loading " << weight_tensor->data.size() << " parameters from " << weight_name << std::endl;

        // Check size and handle mismatch
        size_t params_to_load = std::min(weight_tensor->data.size(), n_params);

        if (weight_tensor->data.size() != n_params) {
            std::cerr << "    Warning: Parameter count mismatch. Loaded tensor has "
                      << weight_tensor->data.size() << " but encoding expects " << n_params << std::endl;

            if (weight_tensor->data.size() > n_params) {
                std::cout << "    Truncating to " << n_params << " parameters" << std::endl;
            } else {
                std::cout << "    Padding with zeros to " << n_params << " parameters" << std::endl;
            }
        }

        // Copy parameters
        std::vector<float> params_fp32(n_params, 0.0f);  // Initialize with zeros for padding
        for (size_t i = 0; i < params_to_load; ++i) {
            params_fp32[i] = weight_tensor->data[i];
        }

        // Convert to half precision
        std::vector<precision_t> params_fp16(n_params);
        for (size_t i = 0; i < n_params; ++i) {
            params_fp16[i] = (precision_t)params_fp32[i];
        }

        std::cout << "    Copying " << n_params << " parameters to device..." << std::endl;

        // Use GPUMemory to upload parameters
        tcnn::GPUMemory<precision_t> gpu_params(n_params);
        gpu_params.copy_from_host(params_fp16.data());
        // Set both training and inference params to the same values
        encoding->set_params(gpu_params.data(), gpu_params.data(), gpu_params.data());

        std::cout << "    Successfully loaded " << params_to_load << " parameters" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "    Error loading position encoder weights: " << e.what() << std::endl;
        return false;
    }
}

bool NeuralNetwork::load_direction_encoder_weights(const WeightLoader& loader) {
    if (!direction_encoder_) {
        std::cerr << "  Error: Direction encoder is null" << std::endl;
        return false;
    }

    auto* network = reinterpret_cast<Network<precision_t>*>(direction_encoder_);
    std::cout << "  Loading weights for direction_encoder..." << std::endl;

    try {
        auto params = network->params();
        size_t n_params = network->n_params();

        std::cout << "    Network expects " << n_params << " parameters" << std::endl;

        // Load parameters as a single flat chunk
        std::string weight_name = "direction_encoder.params";
        const Tensor* weight_tensor = loader.get_tensor(weight_name);

        if (!weight_tensor) {
            std::cerr << "    Error: Could not find tensor " << weight_name << std::endl;
            return false;
        }

        std::cout << "    Loading " << weight_tensor->data.size() << " parameters from " << weight_name << std::endl;

        // Check size and handle mismatch
        size_t params_to_load = std::min(weight_tensor->data.size(), n_params);

        if (weight_tensor->data.size() != n_params) {
            std::cerr << "    Warning: Parameter count mismatch. Loaded tensor has "
                      << weight_tensor->data.size() << " but network expects " << n_params << std::endl;

            if (weight_tensor->data.size() > n_params) {
                std::cout << "    Truncating to " << n_params << " parameters" << std::endl;
            } else {
                std::cout << "    Padding with zeros to " << n_params << " parameters" << std::endl;
            }
        }

        // Copy parameters
        std::vector<float> params_fp32(n_params, 0.0f);  // Initialize with zeros for padding
        for (size_t i = 0; i < params_to_load; ++i) {
            params_fp32[i] = weight_tensor->data[i];
        }

        // Convert to half precision
        std::vector<precision_t> params_fp16(n_params);
        for (size_t i = 0; i < n_params; ++i) {
            params_fp16[i] = (precision_t)params_fp32[i];
        }

        std::cout << "    Copying " << n_params << " parameters to device..." << std::endl;

        // Use GPUMemory to upload parameters
        tcnn::GPUMemory<precision_t> gpu_params(n_params);
        gpu_params.copy_from_host(params_fp16.data());
        // Set both training and inference params to the same values
        network->set_params(gpu_params.data(), gpu_params.data(), gpu_params.data());

        std::cout << "    Successfully loaded " << params_to_load << " parameters" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "    Error loading direction encoder weights: " << e.what() << std::endl;
        return false;
    }
}

bool NeuralNetwork::load_network_weights(
    void* network_ptr,
    const WeightLoader& loader,
    const std::string& prefix,
    const NetworkConfig::DecoderConfig& decoder_config
) {
    if (!network_ptr) {
        std::cerr << "  Error: Network pointer is null for " << prefix << std::endl;
        return false;
    }

    auto* network = reinterpret_cast<Network<precision_t>*>(network_ptr);

    std::cout << "  Loading weights for " << prefix << " decoder..." << std::endl;

    try {
        // Get network parameters
        auto params = network->params();
        size_t n_params = network->n_params();

        std::cout << "    Network expects " << n_params << " parameters" << std::endl;

        // Load parameters as a single flat chunk
        std::string weight_name = prefix + ".params";
        const Tensor* weight_tensor = loader.get_tensor(weight_name);

        if (!weight_tensor) {
            std::cerr << "    Error: Could not find tensor " << weight_name << std::endl;
            return false;
        }

        std::cout << "    Loading " << weight_tensor->data.size() << " parameters from " << weight_name << std::endl;

        // Check size and handle mismatch
        size_t params_to_load = std::min(weight_tensor->data.size(), n_params);

        if (weight_tensor->data.size() != n_params) {
            std::cerr << "    Warning: Parameter count mismatch. Loaded tensor has "
                      << weight_tensor->data.size() << " but network expects " << n_params << std::endl;

            if (weight_tensor->data.size() > n_params) {
                std::cout << "    Truncating to " << n_params << " parameters" << std::endl;
            } else {
                std::cout << "    Padding with zeros to " << n_params << " parameters" << std::endl;
            }
        }

        // Copy parameters
        std::vector<float> params_fp32(n_params, 0.0f);  // Initialize with zeros for padding
        for (size_t i = 0; i < params_to_load; ++i) {
            params_fp32[i] = weight_tensor->data[i];
        }

        // Convert to half precision
        std::vector<precision_t> params_fp16(n_params);
        for (size_t i = 0; i < n_params; ++i) {
            params_fp16[i] = (precision_t)params_fp32[i];
        }

        // Use GPUMemory to upload parameters
        tcnn::GPUMemory<precision_t> gpu_params(n_params);
        gpu_params.copy_from_host(params_fp16.data());
        // Set both training and inference params to the same values
        network->set_params(gpu_params.data(), gpu_params.data(), gpu_params.data());

        std::cout << "    Successfully loaded " << params_to_load << " parameters" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "    Error loading weights: " << e.what() << std::endl;
        return false;
    }
}

} // namespace neural
