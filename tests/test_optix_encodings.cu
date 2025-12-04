/**
 * OptiX Encoding Test: Compare intermediate encodings
 *
 * This test compares the hash grid and direction encodings from the OptiX
 * implementation against the ground truth saved in predictions.bin
 */

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <vector>

#include "../src/neural/weight_loader.h"
#include "../src/neural/config.h"
#include "../src/optix/neural_params.h"
#include "../src/utils/cuda_utils.h"
#include "../src/utils/error.h"
#include "../src/utils/debug_utils.h"
#include "../programs/neural_inference.cuh"
#include "neural_proxy_predictions.h"

// Kernel to test hash encoding
__global__ void test_hash_encoding_kernel(
    const float* positions,
    const HashGridParams hash_params,
    float* hash_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float3 position = make_float3(
        positions[idx * 3 + 0],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
    );

    float hash_output[32];
    hash_encode(position, hash_params, hash_output);

    for (int i = 0; i < 32; ++i) {
        hash_out[idx * 32 + i] = hash_output[i];
    }
}

// Kernel to print weights and test a single forward pass
__global__ void print_weights_kernel(const MLPParams dir_encoder) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Checkpoint 2: Weights seen by kernel ===\n");
        printf("n_layers: %u\n", dir_encoder.n_layers);
        printf("Layer 0 in_dim: %u, out_dim: %u\n",
               dir_encoder.layers[0].in_dim,
               dir_encoder.layers[0].out_dim);
        printf("First 10 weights from Layer 0:\n");
        for (int i = 0; i < 10; ++i) {
            printf("  weights[%d] = %f\n", i, dir_encoder.layers[0].weights[i]);
        }

        // Test a simple forward pass with known input
        printf("\n=== Testing Forward Pass ===\n");
        float test_input[16];
        for (int i = 0; i < 3; ++i) test_input[i] = 0.5f;  // Simple test values
        for (int i = 3; i < 16; ++i) test_input[i] = 1.0f;  // Padding

        printf("Input: [0.5, 0.5, 0.5, 1.0, ...]\n");

        float output[16];
        float scratch[128];
        mlp_forward_fp16(test_input, dir_encoder, output, scratch);

        printf("Output (first 4):\n");
        for (int i = 0; i < 4; ++i) {
            printf("  output[%d] = %f\n", i, output[i]);
        }
    }
}

// Kernel to test direction encoding
__global__ void test_direction_encoding_kernel(
    const float* directions,
    const MLPParams dir_encoder,
    float* dir_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float3 direction = make_float3(
        directions[idx * 3 + 0],
        directions[idx * 3 + 1],
        directions[idx * 3 + 2]
    );

    float direction_input[16];
    direction_input[0] = direction.x;
    direction_input[1] = direction.y;
    direction_input[2] = direction.z;
    for (int i = 3; i < 16; ++i) {
        direction_input[i] = 1.0f;
    }

    float direction_encoding[16];
    float scratch[128];
    mlp_forward_fp16(direction_input, dir_encoder, direction_encoding, scratch);

    for (int i = 0; i < 16; ++i) {
        dir_out[idx * 16 + i] = direction_encoding[i];
    }
}

// Kernel to test visibility decoder using OptiX encoders
__global__ void test_visibility_decoder_kernel(
    const float* positions,
    const float* directions,
    const HashGridParams hash_params,
    const MLPParams dir_encoder,
    const MLPParams vis_decoder,
    float* visibility_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float3 position = make_float3(
        positions[idx * 3 + 0],
        positions[idx * 3 + 1],
        positions[idx * 3 + 2]
    );

    float3 direction = make_float3(
        directions[idx * 3 + 0],
        directions[idx * 3 + 1],
        directions[idx * 3 + 2]
    );

    // Hash encoding
    float hash_output[32];
    hash_encode(position, hash_params, hash_output);

    // Direction encoding
    float direction_input[16];
    direction_input[0] = direction.x;
    direction_input[1] = direction.y;
    direction_input[2] = direction.z;
    for (int i = 3; i < 16; ++i) {
        direction_input[i] = 1.0f;
    }
    float direction_encoding[16];
    float dir_scratch[256];
    mlp_forward_fp16(direction_input, dir_encoder, direction_encoding, dir_scratch);

    // Concatenate encodings (32 + 16 = 48)
    float concatenated[48];
    for (int i = 0; i < 32; ++i) {
        concatenated[i] = hash_output[i];
    }
    for (int i = 0; i < 16; ++i) {
        concatenated[32 + i] = direction_encoding[i];
    }

    // Run visibility decoder with fp16
    float vis_output[16];  // Output is padded to 16
    float vis_scratch[256];
    mlp_forward_fp16(concatenated, vis_decoder, vis_output, vis_scratch);

    // First element is the visibility logit
    visibility_out[idx] = vis_output[0];
}

// Kernel to test visibility decoder using ground truth encodings
__global__ void test_visibility_decoder_with_gt_encodings_kernel(
    const float* gt_hash_encodings,
    const float* gt_dir_encodings,
    const MLPParams vis_decoder,
    float* visibility_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Use ground truth encodings directly
    float concatenated[48];
    for (int i = 0; i < 32; ++i) {
        concatenated[i] = gt_hash_encodings[idx * 32 + i];
    }
    for (int i = 0; i < 16; ++i) {
        concatenated[32 + i] = gt_dir_encodings[idx * 16 + i];
    }

    // Run visibility decoder with fp16
    float vis_output[16];
    float vis_scratch[256];
    mlp_forward_fp16(concatenated, vis_decoder, vis_output, vis_scratch);

    for (int i = 0; i < 16; ++i) {
        // First element is the visibility logit
        visibility_out[idx*16 + i] = vis_output[i];
    }
}

// Kernel to test visibility decoder using GT hash + predicted direction
__global__ void test_visibility_decoder_with_gt_hash_pred_dir_kernel(
    const float* gt_hash_encodings,
    const float* pred_dir_encodings,
    const MLPParams vis_decoder,
    float* visibility_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Use GT hash + predicted direction encodings
    float concatenated[48];
    for (int i = 0; i < 32; ++i) {
        concatenated[i] = gt_hash_encodings[idx * 32 + i];
    }
    for (int i = 0; i < 16; ++i) {
        concatenated[32 + i] = pred_dir_encodings[idx * 16 + i];
    }

    // Run visibility decoder with fp16
    float vis_output[16];
    float vis_scratch[256];
    mlp_forward_fp16(concatenated, vis_decoder, vis_output, vis_scratch);

    for (int i = 0; i < 16; ++i) {
        visibility_out[idx*16 + i] = vis_output[i];
    }
}

// Kernel to test visibility decoder using predicted hash + GT direction
__global__ void test_visibility_decoder_with_pred_hash_gt_dir_kernel(
    const float* pred_hash_encodings,
    const float* gt_dir_encodings,
    const MLPParams vis_decoder,
    float* visibility_out,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Use predicted hash + GT direction encodings
    float concatenated[48];
    for (int i = 0; i < 32; ++i) {
        concatenated[i] = pred_hash_encodings[idx * 32 + i];
    }
    for (int i = 0; i < 16; ++i) {
        concatenated[32 + i] = gt_dir_encodings[idx * 16 + i];
    }

    // Run visibility decoder with fp16
    float vis_output[16];
    float vis_scratch[256];
    mlp_forward_fp16(concatenated, vis_decoder, vis_output, vis_scratch);

    for (int i = 0; i < 16; ++i) {
        visibility_out[idx*16 + i] = vis_output[i];
    }
}

int main(int argc, char** argv) {
    std::cout << "=== OptiX Encoding Test ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> <predictions.bin>" << std::endl;
        return 1;
    }

    std::string weights_file = argv[1];
    std::string predictions_file = argv[2];

    try {
        cuda_utils::print_device_info();

        // Load weights
        std::cout << "\n=== Loading Weights ===" << std::endl;
        neural::WeightLoader loader;
        if (!loader.load_from_file(weights_file)) {
            std::cerr << "Failed to load weights" << std::endl;
            return 1;
        }

        // Convert to OptiX format
        neural::NetworkConfig config = neural::NetworkConfig::instant_ngp_default();
        config.visibility_decoder.output_activation = "None";
        optix::NeuralNetworkParamsHost neural_params(config);
        if (!neural_params.load_from_weights(loader)) {
            std::cerr << "Failed to convert weights" << std::endl;
            return 1;
        }
        std::cout << "Weights loaded and converted" << std::endl;

        // Read predictions
        std::cout << "\n=== Loading Predictions ===" << std::endl;
        neural_proxy::PredictionReader reader(predictions_file);
        const auto& header = reader.header();

        std::cout << "Samples: " << header.num_samples << std::endl;
        std::cout << "Has hash grid: " << (header.has_hash_grid ? "yes" : "no") << std::endl;
        std::cout << "Has direction encodings: " << (header.has_direction_encodings ? "yes" : "no") << std::endl;

        if (!header.has_hash_grid || !header.has_direction_encodings) {
            std::cerr << "Predictions file must have intermediate encodings" << std::endl;
            return 1;
        }

        // Load all samples
        auto samples = reader.read_all();
        const int total_samples = samples.size();
        const int batch_size = 4096;  // Process in batches for efficiency
        std::cout << "\nTesting all " << total_samples << " samples in batches of " << batch_size << std::endl;

        // Allocate device memory for batch processing
        float* d_positions = cuda_utils::allocate_device<float>(batch_size * 3);
        float* d_directions = cuda_utils::allocate_device<float>(batch_size * 3);
        float* d_hash_out = cuda_utils::allocate_device<float>(batch_size * 32);
        float* d_dir_out = cuda_utils::allocate_device<float>(batch_size * 16);
        float* d_visibility_out = cuda_utils::allocate_device<float>(batch_size);
        float* d_visibility_gt_enc_out = cuda_utils::allocate_device<float>(batch_size*16);
        float* d_visibility_gt_hash_pred_dir_out = cuda_utils::allocate_device<float>(batch_size*16);
        float* d_visibility_pred_hash_gt_dir_out = cuda_utils::allocate_device<float>(batch_size*16);
        float* d_gt_hash_encodings = cuda_utils::allocate_device<float>(batch_size * 32);
        float* d_gt_dir_encodings = cuda_utils::allocate_device<float>(batch_size * 16);

        // Host buffers for batch processing
        std::vector<float> h_positions(batch_size * 3);
        std::vector<float> h_directions(batch_size * 3);
        std::vector<float> h_hash_out(batch_size * 32);
        std::vector<float> h_dir_out(batch_size * 16);
        std::vector<float> h_visibility_out(batch_size);
        std::vector<float> h_visibility_gt_enc_out(batch_size*16);
        std::vector<float> h_visibility_gt_hash_pred_dir_out(batch_size*16);
        std::vector<float> h_visibility_pred_hash_gt_dir_out(batch_size*16);
        std::vector<float> h_gt_hash_encodings(batch_size * 32);
        std::vector<float> h_gt_dir_encodings(batch_size * 16);

        // Aggregate statistics
        const int n_hash_levels = 16;
        const int features_per_level = 2;

        double total_hash_error = 0.0;
        double total_dir_error = 0.0;
        double total_vis_error = 0.0;
        double total_vis_gt_enc_error = 0.0;  // Visibility error with GT encodings
        double total_vis_gt_hash_pred_dir_error = 0.0;  // Visibility error with GT hash + pred dir
        double total_vis_pred_hash_gt_dir_error = 0.0;  // Visibility error with pred hash + GT dir
        double max_hash_error = 0.0;
        double max_dir_error = 0.0;
        double max_vis_error = 0.0;
        double max_vis_gt_enc_error = 0.0;
        double max_vis_gt_hash_pred_dir_error = 0.0;
        double max_vis_pred_hash_gt_dir_error = 0.0;
        int correct_predictions = 0;
        int correct_predictions_gt_enc = 0;
        int correct_predictions_gt_hash_pred_dir = 0;
        int correct_predictions_pred_hash_gt_dir = 0;

        // Per-level hash grid statistics
        std::vector<double> per_level_hash_error(n_hash_levels, 0.0);
        std::vector<double> per_level_max_error(n_hash_levels, 0.0);

        // Process all samples in batches
        std::cout << "\n=== Processing All Samples ===" << std::endl;
        const int threads = 256;

        for (int batch_start = 0; batch_start < total_samples; batch_start += batch_size) {
            int current_batch_size = std::min(batch_size, total_samples - batch_start);
            int blocks = (current_batch_size + threads - 1) / threads;

            // Prepare batch inputs
            for (int i = 0; i < current_batch_size; ++i) {
                int sample_idx = batch_start + i;
                h_positions[i * 3 + 0] = (samples[sample_idx].origin[0] + 1.0f) / 2.0f;
                h_positions[i * 3 + 1] = (samples[sample_idx].origin[1] + 1.0f) / 2.0f;
                h_positions[i * 3 + 2] = (samples[sample_idx].origin[2] + 1.0f) / 2.0f;

                h_directions[i * 3 + 0] = samples[sample_idx].direction[0];
                h_directions[i * 3 + 1] = samples[sample_idx].direction[1];
                h_directions[i * 3 + 2] = samples[sample_idx].direction[2];

                // Prepare ground truth encodings
                for (int j = 0; j < 32; ++j) {
                    h_gt_hash_encodings[i * 32 + j] = samples[sample_idx].hash_grid[j];
                }
                for (int j = 0; j < 16; ++j) {
                    h_gt_dir_encodings[i * 16 + j] = samples[sample_idx].direction_encodings[j];
                }
            }

            // Copy to device
            cuda_utils::copy_to_device(d_positions, h_positions.data(), current_batch_size * 3);
            cuda_utils::copy_to_device(d_directions, h_directions.data(), current_batch_size * 3);
            cuda_utils::copy_to_device(d_gt_hash_encodings, h_gt_hash_encodings.data(), current_batch_size * 32);
            cuda_utils::copy_to_device(d_gt_dir_encodings, h_gt_dir_encodings.data(), current_batch_size * 16);

            // Test hash encoding
            test_hash_encoding_kernel<<<blocks, threads>>>(
                d_positions,
                neural_params.get_device_params().hash_encoding,
                d_hash_out,
                current_batch_size
            );
            CUDA_SYNC_CHECK();

            // Test direction encoding
            test_direction_encoding_kernel<<<blocks, threads>>>(
                d_directions,
                neural_params.get_device_params().direction_encoder,
                d_dir_out,
                current_batch_size
            );
            CUDA_SYNC_CHECK();

            // Test visibility decoder with OptiX encoders
            test_visibility_decoder_kernel<<<blocks, threads>>>(
                d_positions,
                d_directions,
                neural_params.get_device_params().hash_encoding,
                neural_params.get_device_params().direction_encoder,
                neural_params.get_device_params().visibility_decoder,
                d_visibility_out,
                current_batch_size
            );
            CUDA_SYNC_CHECK();

            // Test visibility decoder with ground truth encodings
            test_visibility_decoder_with_gt_encodings_kernel<<<blocks, threads>>>(
                d_gt_hash_encodings,
                d_gt_dir_encodings,
                neural_params.get_device_params().visibility_decoder,
                d_visibility_gt_enc_out,
                current_batch_size
            );
            CUDA_SYNC_CHECK();

            // Test visibility decoder with GT hash + predicted direction
            test_visibility_decoder_with_gt_hash_pred_dir_kernel<<<blocks, threads>>>(
                d_gt_hash_encodings,
                d_dir_out,
                neural_params.get_device_params().visibility_decoder,
                d_visibility_gt_hash_pred_dir_out,
                current_batch_size
            );
            CUDA_SYNC_CHECK();

            // Test visibility decoder with predicted hash + GT direction
            test_visibility_decoder_with_pred_hash_gt_dir_kernel<<<blocks, threads>>>(
                d_hash_out,
                d_gt_dir_encodings,
                neural_params.get_device_params().visibility_decoder,
                d_visibility_pred_hash_gt_dir_out,
                current_batch_size
            );
            CUDA_SYNC_CHECK();

            // Copy results back
            cuda_utils::copy_to_host(h_hash_out.data(), d_hash_out, current_batch_size * 32);
            cuda_utils::copy_to_host(h_dir_out.data(), d_dir_out, current_batch_size * 16);
            cuda_utils::copy_to_host(h_visibility_out.data(), d_visibility_out, current_batch_size);
            cuda_utils::copy_to_host(h_visibility_gt_enc_out.data(), d_visibility_gt_enc_out, current_batch_size*16);
            cuda_utils::copy_to_host(h_visibility_gt_hash_pred_dir_out.data(), d_visibility_gt_hash_pred_dir_out, current_batch_size*16);
            cuda_utils::copy_to_host(h_visibility_pred_hash_gt_dir_out.data(), d_visibility_pred_hash_gt_dir_out, current_batch_size*16);

            // Aggregate statistics for this batch
            for (int i = 0; i < current_batch_size; ++i) {
                int sample_idx = batch_start + i;

                // Hash encoding errors (per-level tracking)
                for (int j = 0; j < 32; ++j) {
                    // float gt = samples[sample_idx].hash_grid[j];
                    float gt = h_gt_hash_encodings[i * 32 + j];
                    float pred = h_hash_out[i * 32 + j];
                    double error = std::abs(gt - pred);
                    total_hash_error += error;
                    max_hash_error = std::max(max_hash_error, error);

                    // Track per-level statistics
                    int level = j / features_per_level;
                    per_level_hash_error[level] += error;
                    per_level_max_error[level] = std::max(per_level_max_error[level], error);
                }

                // Direction encoding errors
                for (int j = 0; j < 16; ++j) {
                    float gt = h_gt_dir_encodings[i * 16 + j];
                    float pred = h_dir_out[i * 16 + j];
                    double error = std::abs(gt - pred);
                    total_dir_error += error;
                    max_dir_error = std::max(max_dir_error, error);
                }

                // Visibility errors and accuracy (with OptiX encoders)
                // Note: Network now outputs raw logits (not probabilities)
                float gt_logit = samples[sample_idx].visibility_logit;
                float pred_logit = h_visibility_out[i];  // Direct logit output

                // Convert logits to probabilities for accuracy check
                float gt_prob = 1.0f / (1.0f + std::exp(-gt_logit));
                float pred_prob = 1.0f / (1.0f + std::exp(-pred_logit));

                bool gt_visible = gt_logit >= 0.0f;  // Threshold logit at 0 (equivalent to prob >= 0.5)
                bool pred_visible = pred_logit >= 0.0f;
                if (gt_visible == pred_visible) {
                    correct_predictions++;
                }

                double error = std::abs(gt_logit - pred_logit);
                total_vis_error += error;
                max_vis_error = std::max(max_vis_error, error);

                // Visibility errors and accuracy (with GT encodings)
                float pred_gt_enc_logit = h_visibility_gt_enc_out[i*16];  // Direct logit output

                // if (i == 0) {
                //     std::cout << "Full vis output" << std::endl;
                //     std::cout << "GT vis " << gt_logit << std::endl;
                //     for (int j = 0; j < 16; ++j)
                //     {
                //         std::cout << "    Prediction at dim " << j << ": " << h_visibility_gt_enc_out[i*16 + j] << std::endl;
                //     }
                // }

                bool pred_gt_enc_visible = pred_gt_enc_logit >= 0.0f;
                if (gt_visible == pred_gt_enc_visible) {
                    correct_predictions_gt_enc++;
                }

                double error_gt_enc = std::abs(gt_logit - pred_gt_enc_logit);
                total_vis_gt_enc_error += error_gt_enc;
                max_vis_gt_enc_error = std::max(max_vis_gt_enc_error, error_gt_enc);

                // Visibility errors and accuracy (with GT hash + predicted direction)
                float pred_gt_hash_pred_dir_logit = h_visibility_gt_hash_pred_dir_out[i*16];  // Direct logit output
                bool pred_gt_hash_pred_dir_visible = pred_gt_hash_pred_dir_logit >= 0.0f;
                if (gt_visible == pred_gt_hash_pred_dir_visible) {
                    correct_predictions_gt_hash_pred_dir++;
                }

                double error_gt_hash_pred_dir = std::abs(gt_logit - pred_gt_hash_pred_dir_logit);
                total_vis_gt_hash_pred_dir_error += error_gt_hash_pred_dir;
                max_vis_gt_hash_pred_dir_error = std::max(max_vis_gt_hash_pred_dir_error, error_gt_hash_pred_dir);

                // Visibility errors and accuracy (with predicted hash + GT direction)
                float pred_pred_hash_gt_dir_logit = h_visibility_pred_hash_gt_dir_out[i*16];  // Direct logit output
                bool pred_pred_hash_gt_dir_visible = pred_pred_hash_gt_dir_logit >= 0.0f;
                if (gt_visible == pred_pred_hash_gt_dir_visible) {
                    correct_predictions_pred_hash_gt_dir++;
                }

                double error_pred_hash_gt_dir = std::abs(gt_logit - pred_pred_hash_gt_dir_logit);
                total_vis_pred_hash_gt_dir_error += error_pred_hash_gt_dir;
                max_vis_pred_hash_gt_dir_error = std::max(max_vis_pred_hash_gt_dir_error, error_pred_hash_gt_dir);
            }

            // Progress indicator
            if ((batch_start + current_batch_size) % 10000 == 0 || batch_start + current_batch_size == total_samples) {
                std::cout << "  Processed " << (batch_start + current_batch_size) << "/" << total_samples << " samples\r" << std::flush;
            }
        }
        std::cout << std::endl;

        // Print aggregate results
        std::cout << "\n=== Hash Encoding Results (All Samples) ===" << std::endl;
        std::cout << "  Overall Mean L1 error: " << (total_hash_error / (total_samples * 32)) << std::endl;
        std::cout << "  Overall Max error: " << max_hash_error << std::endl;

        std::cout << "\n  Per-Level Statistics:" << std::endl;
        std::cout << "  Level | Mean Error | Max Error" << std::endl;
        std::cout << "  ------|------------|----------" << std::endl;
        for (int level = 0; level < n_hash_levels; ++level) {
            double mean_error = per_level_hash_error[level] / (total_samples * features_per_level);
            std::cout << "  " << std::setw(5) << level
                      << " | " << std::setw(10) << std::fixed << std::setprecision(6) << mean_error
                      << " | " << std::setw(9) << std::fixed << std::setprecision(6) << per_level_max_error[level]
                      << std::endl;
        }

        std::cout << "\n=== Direction Encoding Results (All Samples) ===" << std::endl;
        std::cout << "  Mean L1 error: " << (total_dir_error / (total_samples * 16)) << std::endl;
        std::cout << "  Max error: " << max_dir_error << std::endl;

        std::cout << "\n=== Visibility Decoder Results (with OptiX Encoders) ===" << std::endl;
        std::cout << "  Mean L1 error (logits): " << (total_vis_error / total_samples) << std::endl;
        std::cout << "  Max error: " << max_vis_error << std::endl;
        std::cout << "  Accuracy: " << (100.0 * correct_predictions / total_samples) << "% ("
                  << correct_predictions << "/" << total_samples << ")" << std::endl;

        std::cout << "\n=== Visibility Decoder Results (with Ground Truth Encodings) ===" << std::endl;
        std::cout << "  Mean L1 error (logits): " << (total_vis_gt_enc_error / total_samples) << std::endl;
        std::cout << "  Max error: " << max_vis_gt_enc_error << std::endl;
        std::cout << "  Accuracy: " << (100.0 * correct_predictions_gt_enc / total_samples) << "% ("
                  << correct_predictions_gt_enc << "/" << total_samples << ")" << std::endl;

        std::cout << "\n=== Visibility Decoder Results (with GT Hash + Predicted Direction) ===" << std::endl;
        std::cout << "  Mean L1 error (logits): " << (total_vis_gt_hash_pred_dir_error / total_samples) << std::endl;
        std::cout << "  Max error: " << max_vis_gt_hash_pred_dir_error << std::endl;
        std::cout << "  Accuracy: " << (100.0 * correct_predictions_gt_hash_pred_dir / total_samples) << "% ("
                  << correct_predictions_gt_hash_pred_dir << "/" << total_samples << ")" << std::endl;

        std::cout << "\n=== Visibility Decoder Results (with Predicted Hash + GT Direction) ===" << std::endl;
        std::cout << "  Mean L1 error (logits): " << (total_vis_pred_hash_gt_dir_error / total_samples) << std::endl;
        std::cout << "  Max error: " << max_vis_pred_hash_gt_dir_error << std::endl;
        std::cout << "  Accuracy: " << (100.0 * correct_predictions_pred_hash_gt_dir / total_samples) << "% ("
                  << correct_predictions_pred_hash_gt_dir << "/" << total_samples << ")" << std::endl;

        std::cout << "\n=== Error Contribution Analysis ===" << std::endl;
        std::cout << "  Total encoder error contribution: "
                  << (total_vis_error - total_vis_gt_enc_error) / total_samples << std::endl;
        std::cout << "  Hash encoder error contribution: "
                  << (total_vis_pred_hash_gt_dir_error - total_vis_gt_enc_error) / total_samples << std::endl;
        std::cout << "  Direction encoder error contribution: "
                  << (total_vis_gt_hash_pred_dir_error - total_vis_gt_enc_error) / total_samples << std::endl;
        std::cout << "  Decoder-only error: " << (total_vis_gt_enc_error / total_samples) << std::endl;
        std::cout << "\n  Accuracy Comparison:" << std::endl;
        std::cout << "    Both predicted encoders: " << (100.0 * correct_predictions / total_samples) << "%" << std::endl;
        std::cout << "    GT hash + pred direction: " << (100.0 * correct_predictions_gt_hash_pred_dir / total_samples) << "%" << std::endl;
        std::cout << "    Pred hash + GT direction: " << (100.0 * correct_predictions_pred_hash_gt_dir / total_samples) << "%" << std::endl;
        std::cout << "    Both GT encoders: " << (100.0 * correct_predictions_gt_enc / total_samples) << "%" << std::endl;
        std::cout << "\n  Accuracy Gains from Perfect Encodings:" << std::endl;
        std::cout << "    Perfect hash (vs both predicted): "
                  << ((correct_predictions_gt_hash_pred_dir - correct_predictions) * 100.0 / total_samples) << "%" << std::endl;
        std::cout << "    Perfect direction (vs both predicted): "
                  << ((correct_predictions_pred_hash_gt_dir - correct_predictions) * 100.0 / total_samples) << "%" << std::endl;
        std::cout << "    Both perfect (vs both predicted): "
                  << ((correct_predictions_gt_enc - correct_predictions) * 100.0 / total_samples) << "%" << std::endl;

        // Cleanup
        cuda_utils::free_device(d_positions);
        cuda_utils::free_device(d_directions);
        cuda_utils::free_device(d_hash_out);
        cuda_utils::free_device(d_dir_out);
        cuda_utils::free_device(d_visibility_out);
        cuda_utils::free_device(d_visibility_gt_enc_out);
        cuda_utils::free_device(d_visibility_gt_hash_pred_dir_out);
        cuda_utils::free_device(d_visibility_pred_hash_gt_dir_out);
        cuda_utils::free_device(d_gt_hash_encodings);
        cuda_utils::free_device(d_gt_dir_encodings);

        std::cout << "\n=== Test Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
