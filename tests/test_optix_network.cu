/**
 * OptiX Network Test: Compare OptiX neural inference against ground truth
 *
 * This test program validates that the OptiX implementation of neural inference
 * produces the same results as the standalone tiny-cuda-nn version by:
 * 1. Loading PyTorch weights into OptiX neural network format
 * 2. Running test queries from predictions.bin
 * 3. Comparing outputs with ground truth predictions
 */

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "../src/neural/network.h"
#include "../src/neural/weight_loader.h"
#include "../src/neural/config.h"
#include "../src/optix/neural_params.h"
#include "../src/utils/cuda_utils.h"
#include "../src/utils/error.h"
#include "../programs/neural_inference.cuh"
#include "neural_proxy_predictions.h"

// Kernel to run OptiX neural inference on a batch of samples
__global__ void batch_inference_kernel(
    const float* positions,          // [batch_size, 3]
    const float* directions,         // [batch_size, 3]
    const NeuralNetworkParams net_params,
    float* visibility_out,           // [batch_size]
    float* normals_out,              // [batch_size, 3]
    float* depth_out,                // [batch_size]
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    // Load position and direction
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

    // Run neural inference
    float visibility;
    float3 normal;
    float depth;

    neural_inference(
        position,
        direction,
        net_params,
        visibility,
        normal,
        depth
    );

    // Store results
    visibility_out[idx] = visibility;
    normals_out[idx * 3 + 0] = normal.x;
    normals_out[idx * 3 + 1] = normal.y;
    normals_out[idx * 3 + 2] = normal.z;
    depth_out[idx] = depth;
}

// Test with ground truth predictions (batched)
void test_optix_with_predictions(
    const optix::NeuralNetworkParamsHost& neural_params,
    const std::string& prediction_file,
    int batch_size = 256
) {
    std::cout << "\n=== Testing OptiX Network with Ground Truth Predictions ===" << std::endl;

    try {
        neural_proxy::PredictionReader reader(prediction_file);
        const auto& header = reader.header();

        std::cout << "Prediction file: " << prediction_file << std::endl;
        std::cout << "Mesh: " << header.mesh_name << std::endl;
        std::cout << "Samples: " << header.num_samples << std::endl;
        std::cout << "Has visibility: " << (header.has_visibility ? "yes" : "no") << std::endl;
        std::cout << "Has depth: " << (header.has_depth ? "yes" : "no") << std::endl;
        std::cout << "Has normal: " << (header.has_normal ? "yes" : "no") << std::endl;

        // Read all samples
        auto samples = reader.read_all();
        std::cout << "Loaded " << samples.size() << " samples" << std::endl;

        // Allocate host and device buffers for batch processing
        std::vector<float> h_positions(batch_size * 3);
        std::vector<float> h_directions(batch_size * 3);
        std::vector<float> h_visibility(batch_size);
        std::vector<float> h_depth(batch_size);
        std::vector<float> h_normals(batch_size * 3);

        float* d_positions = cuda_utils::allocate_device<float>(batch_size * 3);
        float* d_directions = cuda_utils::allocate_device<float>(batch_size * 3);
        float* d_visibility = cuda_utils::allocate_device<float>(batch_size);
        float* d_normals = cuda_utils::allocate_device<float>(batch_size * 3);
        float* d_depth = cuda_utils::allocate_device<float>(batch_size);

        // Metrics accumulators
        int total_visibility_samples = 0;
        int correct_visibility_predictions = 0;
        double total_depth_l1_loss = 0.0;
        int total_depth_samples = 0;
        double total_cosine_similarity = 0.0;
        int total_normal_samples = 0;

        int total_samples = samples.size();
        for (int start = 0; start < total_samples; start += batch_size) {
            int current_batch = std::min(batch_size, total_samples - start);

            // Fill host buffers for this batch
            for (int i = 0; i < current_batch; ++i) {
                const auto& sample = samples[start + i];
                // Normalize positions to [0, 1] (assuming they're in [-1, 1])
                h_positions[i * 3 + 0] = (sample.origin[0] + 1.0f) / 2.0f;
                h_positions[i * 3 + 1] = (sample.origin[1] + 1.0f) / 2.0f;
                h_positions[i * 3 + 2] = (sample.origin[2] + 1.0f) / 2.0f;

                h_directions[i * 3 + 0] = sample.direction[0];
                h_directions[i * 3 + 1] = sample.direction[1];
                h_directions[i * 3 + 2] = sample.direction[2];
            }

            // Copy to device
            cuda_utils::copy_to_device(d_positions, h_positions.data(), batch_size * 3);
            cuda_utils::copy_to_device(d_directions, h_directions.data(), batch_size * 3);

            // Launch kernel
            int threads_per_block = 256;
            int num_blocks = (current_batch + threads_per_block - 1) / threads_per_block;

            batch_inference_kernel<<<num_blocks, threads_per_block>>>(
                d_positions,
                d_directions,
                neural_params.get_device_params(),
                d_visibility,
                d_normals,
                d_depth,
                current_batch
            );
            CUDA_SYNC_CHECK();

            // Copy results back
            cuda_utils::copy_to_host(h_visibility.data(), d_visibility, batch_size);
            cuda_utils::copy_to_host(h_normals.data(), d_normals, batch_size * 3);
            cuda_utils::copy_to_host(h_depth.data(), d_depth, batch_size);

            // Aggregate metrics for this batch
            for (int i = 0; i < current_batch; ++i) {
                const auto& sample = samples[start + i];

                // Visibility accuracy (thresholded at 0.5)
                if (header.has_visibility) {
                    float gt_vis = 1.0f / (1.0f + std::exp(-sample.visibility_logit));
                    bool pred_visible = h_visibility[i] >= 0.5f;
                    bool gt_visible = gt_vis >= 0.5f;
                    if (pred_visible == gt_visible) {
                        correct_visibility_predictions++;
                    }
                    total_visibility_samples++;
                }

                // Depth L1 loss
                if (header.has_depth) {
                    total_depth_l1_loss += std::abs(h_depth[i] - sample.depth);
                    total_depth_samples++;
                }

                // Normal cosine similarity (with normalization)
                if (header.has_normal) {
                    // Normalize predicted normal
                    float pred_norm = std::sqrt(h_normals[i * 3 + 0] * h_normals[i * 3 + 0] +
                                               h_normals[i * 3 + 1] * h_normals[i * 3 + 1] +
                                               h_normals[i * 3 + 2] * h_normals[i * 3 + 2]);
                    float pred_nx = h_normals[i * 3 + 0] / (pred_norm + 1e-8f);
                    float pred_ny = h_normals[i * 3 + 1] / (pred_norm + 1e-8f);
                    float pred_nz = h_normals[i * 3 + 2] / (pred_norm + 1e-8f);

                    // Normalize GT normal
                    float gt_norm = std::sqrt(sample.normal[0] * sample.normal[0] +
                                             sample.normal[1] * sample.normal[1] +
                                             sample.normal[2] * sample.normal[2]);
                    float gt_nx = sample.normal[0] / (gt_norm + 1e-8f);
                    float gt_ny = sample.normal[1] / (gt_norm + 1e-8f);
                    float gt_nz = sample.normal[2] / (gt_norm + 1e-8f);

                    // Compute cosine similarity (dot product of normalized vectors)
                    float cosine_sim = pred_nx * gt_nx + pred_ny * gt_ny + pred_nz * gt_nz;
                    total_cosine_similarity += cosine_sim;
                    total_normal_samples++;
                }
            }
        }

        // Print aggregated metrics
        std::cout << "\n=== OptiX Implementation Metrics ===" << std::endl;

        if (total_visibility_samples > 0) {
            float accuracy = (float)correct_visibility_predictions / total_visibility_samples;
            std::cout << "Visibility Accuracy: " << accuracy
                      << " (" << correct_visibility_predictions << "/" << total_visibility_samples << ")" << std::endl;
        }

        if (total_depth_samples > 0) {
            double mean_l1_loss = total_depth_l1_loss / total_depth_samples;
            std::cout << "Depth L1 Loss: " << mean_l1_loss << std::endl;
        }

        if (total_normal_samples > 0) {
            double mean_cosine_similarity = total_cosine_similarity / total_normal_samples;
            std::cout << "Normal Cosine Similarity: " << mean_cosine_similarity << std::endl;
        }

        // Cleanup device memory
        cuda_utils::free_device(d_positions);
        cuda_utils::free_device(d_directions);
        cuda_utils::free_device(d_visibility);
        cuda_utils::free_device(d_normals);
        cuda_utils::free_device(d_depth);

    } catch (const std::exception& e) {
        std::cerr << "Error testing with predictions: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "=== OptiX Neural Network Test ===" << std::endl;

    // Parse arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> <predictions.bin>" << std::endl;
        std::cerr << "\nweights.bin: Binary weight file (convert from .pth using scripts/convert_checkpoint.py)" << std::endl;
        std::cerr << "predictions.bin: Ground truth predictions for validation" << std::endl;
        return 1;
    }

    std::string weights_file = argv[1];
    std::string predictions_file = argv[2];

    try {
        // Print CUDA device info
        cuda_utils::print_device_info();

        // Load weights
        std::cout << "\n=== Loading Weights ===" << std::endl;
        neural::WeightLoader loader;
        if (!loader.load_from_file(weights_file)) {
            std::cerr << "Failed to load weights from: " << weights_file << std::endl;
            return 1;
        }
        loader.print_summary();

        // Create network configuration
        neural::NetworkConfig config = neural::NetworkConfig::instant_ngp_default();

        // Convert to OptiX format
        std::cout << "\n=== Converting to OptiX Format ===" << std::endl;
        optix::NeuralNetworkParamsHost neural_params(config);
        if (!neural_params.load_from_weights(loader)) {
            std::cerr << "Failed to convert weights to OptiX format" << std::endl;
            return 1;
        }
        std::cout << "Successfully converted weights to OptiX format" << std::endl;

        // Run tests
        test_optix_with_predictions(neural_params, predictions_file);

        std::cout << "\n=== Test Complete ===" << std::endl;
        std::cout << "\nNote: Compare these metrics with the output from test_network" << std::endl;
        std::cout << "to verify the OptiX implementation matches the tiny-cuda-nn version." << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
