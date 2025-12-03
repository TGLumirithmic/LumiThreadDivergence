/**
 * OptiX Encoding Test: Compare intermediate encodings
 *
 * This test compares the hash grid and direction encodings from the OptiX
 * implementation against the ground truth saved in predictions.bin
 */

#include <iostream>
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
        mlp_forward(test_input, dir_encoder, output, scratch);

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
    mlp_forward(direction_input, dir_encoder, direction_encoding, scratch);

    for (int i = 0; i < 16; ++i) {
        dir_out[idx * 16 + i] = direction_encoding[i];
    }
}

// Kernel to test visibility decoder
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
    float dir_scratch[128];
    mlp_forward(direction_input, dir_encoder, direction_encoding, dir_scratch);

    // Concatenate encodings (32 + 16 = 48)
    float concatenated[48];
    for (int i = 0; i < 32; ++i) {
        concatenated[i] = hash_output[i];
    }
    for (int i = 0; i < 16; ++i) {
        concatenated[32 + i] = direction_encoding[i];
    }

    // Debug: print first sample's concatenated input
    if (idx == 0) {
        printf("\n[Vis Decoder Debug] Sample 0 concatenated input (first 8 values):\n");
        for (int i = 0; i < 8; ++i) {
            printf("  concat[%d] = %f\n", i, concatenated[i]);
        }
        printf("[Vis Decoder Debug] Output activation: %s\n", vis_decoder.output_activation);
        printf("[Vis Decoder Debug] Number of layers: %u\n", vis_decoder.n_layers);
    }

    // Run visibility decoder
    float vis_output[16];  // Output is padded to 16
    float vis_scratch[128];
    mlp_forward(concatenated, vis_decoder, vis_output, vis_scratch);

    // Debug: print first sample's output
    if (idx == 0) {
        printf("\n[Vis Decoder Debug] Sample 0 output (first 4 values):\n");
        for (int i = 0; i < 4; ++i) {
            printf("  vis_output[%d] = %f\n", i, vis_output[i]);
        }
    }

    // First element is the visibility logit
    visibility_out[idx] = vis_output[0];
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

        // Test a small batch
        int test_batch = 10;
        auto samples = reader.read_all();
        std::cout << "\nTesting first " << test_batch << " samples" << std::endl;

        // Prepare inputs
        std::vector<float> h_positions(test_batch * 3);
        std::vector<float> h_directions(test_batch * 3);

        for (int i = 0; i < test_batch; ++i) {
            h_positions[i * 3 + 0] = (samples[i].origin[0] + 1.0f) / 2.0f;
            h_positions[i * 3 + 1] = (samples[i].origin[1] + 1.0f) / 2.0f;
            h_positions[i * 3 + 2] = (samples[i].origin[2] + 1.0f) / 2.0f;

            h_directions[i * 3 + 0] = samples[i].direction[0];
            h_directions[i * 3 + 1] = samples[i].direction[1];
            h_directions[i * 3 + 2] = samples[i].direction[2];
        }

        // Allocate device memory
        float* d_positions = cuda_utils::allocate_device<float>(test_batch * 3);
        float* d_directions = cuda_utils::allocate_device<float>(test_batch * 3);
        float* d_hash_out = cuda_utils::allocate_device<float>(test_batch * 32);
        float* d_dir_out = cuda_utils::allocate_device<float>(test_batch * 16);

        cuda_utils::copy_to_device(d_positions, h_positions.data(), test_batch * 3);
        cuda_utils::copy_to_device(d_directions, h_directions.data(), test_batch * 3);

        // Checkpoint 1: Print first 10 direction encoder weights from device
        std::cout << "\n=== Checkpoint 1: Direction Encoder Weights (from device pointer) ===" << std::endl;
        const auto& dir_params = neural_params.get_device_params().direction_encoder;

        // Copy the MLPLayer structure back from device to get the weight pointer
        MLPLayer h_layer0;
        CUDA_CHECK(cudaMemcpy(&h_layer0, dir_params.layers, sizeof(MLPLayer), cudaMemcpyDeviceToHost));
        std::cout << "Layer 0: in_dim=" << h_layer0.in_dim << ", out_dim=" << h_layer0.out_dim << std::endl;
        debug_utils::print_buffer_values(h_layer0.weights, 10, "Dir Encoder Layer 0 Weights (from device)");

        // Test hash encoding
        std::cout << "\n=== Testing Hash Encoding ===" << std::endl;
        int threads = 256;
        int blocks = (test_batch + threads - 1) / threads;

        test_hash_encoding_kernel<<<blocks, threads>>>(
            d_positions,
            neural_params.get_device_params().hash_encoding,
            d_hash_out,
            test_batch
        );
        CUDA_SYNC_CHECK();

        std::vector<float> h_hash_out(test_batch * 32);
        cuda_utils::copy_to_host(h_hash_out.data(), d_hash_out, test_batch * 32);

        // Compare with ground truth
        double total_hash_error = 0.0;
        double max_hash_error = 0.0;

        for (int i = 0; i < test_batch; ++i) {
            std::cout << "\nSample " << i << ":" << std::endl;
            std::cout << "  Position: (" << h_positions[i*3] << ", " << h_positions[i*3+1] << ", " << h_positions[i*3+2] << ")" << std::endl;

            double sample_error = 0.0;
            for (int j = 0; j < 32; ++j) {
                float gt = samples[i].hash_grid[j];
                float pred = h_hash_out[i * 32 + j];
                double error = std::abs(gt - pred);
                sample_error += error;
                total_hash_error += error;
                max_hash_error = std::max(max_hash_error, error);

                if (j < 32) {  // Print first 4 features
                    std::cout << "    [" << j << "] GT: " << gt << ", Pred: " << pred << ", Error: " << error << std::endl;
                }
            }
            std::cout << "  Mean error: " << (sample_error / 32.0) << std::endl;
        }

        std::cout << "\nHash Encoding Results:" << std::endl;
        std::cout << "  Total mean L1 error: " << (total_hash_error / (test_batch * 32)) << std::endl;
        std::cout << "  Max error: " << max_hash_error << std::endl;

        // Test direction encoding
        std::cout << "\n=== Testing Direction Encoding ===" << std::endl;

        // Launch kernel to print weights as seen from device
        print_weights_kernel<<<1, 1>>>(neural_params.get_device_params().direction_encoder);
        CUDA_SYNC_CHECK();

        test_direction_encoding_kernel<<<blocks, threads>>>(
            d_directions,
            neural_params.get_device_params().direction_encoder,
            d_dir_out,
            test_batch
        );
        CUDA_SYNC_CHECK();

        std::vector<float> h_dir_out(test_batch * 16);
        cuda_utils::copy_to_host(h_dir_out.data(), d_dir_out, test_batch * 16);

        // Compare with ground truth
        double total_dir_error = 0.0;
        double max_dir_error = 0.0;

        for (int i = 0; i < test_batch; ++i) {
            std::cout << "\nSample " << i << ":" << std::endl;
            std::cout << "  Direction: (" << h_directions[i*3] << ", " << h_directions[i*3+1] << ", " << h_directions[i*3+2] << ")" << std::endl;

            double sample_error = 0.0;
            for (int j = 0; j < 16; ++j) {
                float gt = samples[i].direction_encodings[j];
                float pred = h_dir_out[i * 16 + j];
                double error = std::abs(gt - pred);
                sample_error += error;
                total_dir_error += error;
                max_dir_error = std::max(max_dir_error, error);

                if (j < 4) {  // Print first 4 features
                    std::cout << "    [" << j << "] GT: " << gt << ", Pred: " << pred << ", Error: " << error << std::endl;
                }
            }
            std::cout << "  Mean error: " << (sample_error / 16.0) << std::endl;
        }

        std::cout << "\nDirection Encoding Results:" << std::endl;
        std::cout << "  Total mean L1 error: " << (total_dir_error / (test_batch * 16)) << std::endl;
        std::cout << "  Max error: " << max_dir_error << std::endl;

        // Test visibility decoder
        std::cout << "\n=== Testing Visibility Decoder ===" << std::endl;

        float* d_visibility_out = cuda_utils::allocate_device<float>(test_batch);

        test_visibility_decoder_kernel<<<blocks, threads>>>(
            d_positions,
            d_directions,
            neural_params.get_device_params().hash_encoding,
            neural_params.get_device_params().direction_encoder,
            neural_params.get_device_params().visibility_decoder,
            d_visibility_out,
            test_batch
        );
        CUDA_SYNC_CHECK();

        std::vector<float> h_visibility_out(test_batch);
        cuda_utils::copy_to_host(h_visibility_out.data(), d_visibility_out, test_batch);

        // Compare with ground truth
        double total_vis_error = 0.0;
        double max_vis_error = 0.0;
        int correct_predictions = 0;

        for (int i = 0; i < test_batch; ++i) {
            float gt_logit = samples[i].visibility_logit;
            float pred_prob = h_visibility_out[i];  // This is a probability (after sigmoid)

            // Convert prediction probability back to logit for comparison
            // logit = log(prob / (1 - prob))
            float pred_logit;
            if (pred_prob <= 0.0f) {
                pred_logit = -100.0f;  // Very negative logit
            } else if (pred_prob >= 1.0f) {
                pred_logit = 100.0f;  // Very positive logit
            } else {
                pred_logit = std::log(pred_prob / (1.0f - pred_prob));
            }

            // Convert GT logit to probability
            float gt_prob = 1.0f / (1.0f + std::exp(-gt_logit));

            // Check if prediction is correct (threshold at 0.5)
            bool gt_visible = gt_prob >= 0.5f;
            bool pred_visible = pred_prob >= 0.5f;
            if (gt_visible == pred_visible) {
                correct_predictions++;
            }

            double error = std::abs(gt_logit - pred_logit);
            total_vis_error += error;
            max_vis_error = std::max(max_vis_error, error);

            std::cout << "\nSample " << i << ":" << std::endl;
            std::cout << "  GT logit: " << gt_logit << " (prob: " << gt_prob << ")" << std::endl;
            std::cout << "  Pred prob: " << pred_prob << " (logit: " << pred_logit << ")" << std::endl;
            std::cout << "  Error: " << error << std::endl;
            std::cout << "  Correct: " << (gt_visible == pred_visible ? "yes" : "no") << std::endl;
        }

        std::cout << "\nVisibility Decoder Results:" << std::endl;
        std::cout << "  Mean L1 error (logits): " << (total_vis_error / test_batch) << std::endl;
        std::cout << "  Max error: " << max_vis_error << std::endl;
        std::cout << "  Accuracy: " << (100.0 * correct_predictions / test_batch) << "% ("
                  << correct_predictions << "/" << test_batch << ")" << std::endl;

        // Cleanup
        cuda_utils::free_device(d_positions);
        cuda_utils::free_device(d_directions);
        cuda_utils::free_device(d_hash_out);
        cuda_utils::free_device(d_dir_out);
        cuda_utils::free_device(d_visibility_out);

        std::cout << "\n=== Test Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
