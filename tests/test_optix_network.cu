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
#include <cstdio>

#include "../src/neural/network.h"
#include "../src/neural/weight_loader.h"
#include "../src/neural/config.h"
#include "../src/optix/neural_params.h"
#include "../src/utils/cuda_utils.h"
#include "../src/utils/error.h"
#include "../programs/neural_inference.cuh"
#include "neural_proxy_predictions.h"

// Simple PPM image writer for visualization
void write_ppm(const std::string& filename, const float* data,
               int width, int height, int channels) {
    FILE* fp = fopen(filename.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    fprintf(fp, "P6\n%d %d\n255\n", width, height);

    std::vector<unsigned char> pixels(width * height * 3);

    for (int i = 0; i < width * height; ++i) {
        for (int c = 0; c < 3 && c < channels; ++c) {
            float val = data[i * channels + c];
            // Clamp to [0, 1] and convert to byte
            val = std::max(0.0f, std::min(1.0f, val));
            pixels[i * 3 + c] = static_cast<unsigned char>(val * 255.0f);
        }
        // If grayscale, replicate to RGB
        if (channels == 1) {
            pixels[i * 3 + 1] = pixels[i * 3 + 0];
            pixels[i * 3 + 2] = pixels[i * 3 + 0];
        }
    }

    fwrite(pixels.data(), 1, pixels.size(), fp);
    fclose(fp);

    std::cout << "Wrote image: " << filename << std::endl;
}

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

            // std::cout << "About to launch with: " << std::endl;
            // std::cout << "    Direction encoder.n_layers: " << neural_params.get_device_params().direction_encoder.n_layers << std::endl;
            // std::cout << "    Visibility decoder.n_layers: " << neural_params.get_device_params().visibility_decoder.n_layers << std::endl;
            // std::cout << "    Depth decoder.n_layers: " << neural_params.get_device_params().depth_decoder.n_layers << std::endl;
            // std::cout << "    Normal decoder.n_layers: " << neural_params.get_device_params().normal_decoder.n_layers << std::endl;

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

// Simple grid test (no ground truth)
void test_simple_grid(
    const optix::NeuralNetworkParamsHost& neural_params,
    const std::string& output_dir
) {
    std::cout << "\n=== Simple Grid Test ===" << std::endl;

    const int width = 512;
    const int height = 512;

    // Generate test positions in a 2D grid
    std::vector<float> h_positions(width * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            // Map pixel coordinates to [0, 1] range
            h_positions[idx + 0] = x / float(width);   // x
            h_positions[idx + 1] = y / float(height);  // y
            h_positions[idx + 2] = 0.0f;               // z at middle plane
        }
    }

    std::cout << "Generated " << (width * height) << " test positions" << std::endl;

    // Generate default directions (all pointing forward in +Z)
    std::vector<float> h_directions(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        h_directions[i * 3 + 0] = 0.0f;  // dx
        h_directions[i * 3 + 1] = 0.0f;  // dy
        h_directions[i * 3 + 2] = 1.0f;  // dz
    }

    // Allocate device memory
    float* d_positions = cuda_utils::allocate_device<float>(width * height * 3);
    float* d_directions = cuda_utils::allocate_device<float>(width * height * 3);
    float* d_visibility = cuda_utils::allocate_device<float>(width * height);
    float* d_normals = cuda_utils::allocate_device<float>(width * height * 3);
    float* d_depth = cuda_utils::allocate_device<float>(width * height);

    // Copy inputs to device
    cuda_utils::copy_to_device(d_positions, h_positions.data(), width * height * 3);
    cuda_utils::copy_to_device(d_directions, h_directions.data(), width * height * 3);

    // Run inference
    std::cout << "Running inference..." << std::endl;
    int threads_per_block = 256;
    int num_blocks = (width * height + threads_per_block - 1) / threads_per_block;

        // std::cout << "About to launch with: " << std::endl;
        // std::cout << "    Direction encoder.n_layers: " << neural_params.get_device_params().direction_encoder.n_layers << std::endl;
        // std::cout << "    Visibility decoder.n_layers: " << neural_params.get_device_params().visibility_decoder.n_layers << std::endl;
        // std::cout << "    Depth decoder.n_layers: " << neural_params.get_device_params().depth_decoder.n_layers << std::endl;
        // std::cout << "    Normal decoder.n_layers: " << neural_params.get_device_params().normal_decoder.n_layers << std::endl;

    batch_inference_kernel<<<num_blocks, threads_per_block>>>(
        d_positions,
        d_directions,
        neural_params.get_device_params(),
        d_visibility,
        d_normals,
        d_depth,
        width * height
    );
    CUDA_SYNC_CHECK();

    // Copy results back
    std::vector<float> visibility(width * height);
    std::vector<float> normals(width * height * 3);
    std::vector<float> depth(width * height);

    cuda_utils::copy_to_host(visibility.data(), d_visibility, width * height);
    cuda_utils::copy_to_host(normals.data(), d_normals, width * height * 3);
    cuda_utils::copy_to_host(depth.data(), d_depth, width * height);

    // Threshold visibility
    std::vector<float> visibility_threshold(width * height);
    for (int i = 0; i < width * height; ++i) {
        visibility_threshold[i] = (visibility[i] >= 0.5f) ? 1.0f : 0.0f;
        // visibility_threshold[i] = (1.0f / (1.0f + std::exp(-visibility[i])));  // Keep raw visibility for visualization
    }

    // Normalize and scale normals to [0, 1]^3 for visualization
    std::vector<float> normals_vis(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        // Normalize the normal vector
        float nx = normals[i * 3 + 0];
        float ny = normals[i * 3 + 1];
        float nz = normals[i * 3 + 2];
        float norm = std::sqrt(nx * nx + ny * ny + nz * nz);

        if (norm > 1e-8f) {
            nx /= norm;
            ny /= norm;
            nz /= norm;
        }

        // Scale from [-1, 1] to [0, 1]
        normals_vis[i * 3 + 0] = (nx + 1.0f) * 0.5f;
        normals_vis[i * 3 + 1] = (ny + 1.0f) * 0.5f;
        normals_vis[i * 3 + 2] = (nz + 1.0f) * 0.5f;
    }

    // Write visualization images
    std::cout << "Writing visualization images..." << std::endl;
    write_ppm(output_dir + "/optix_visibility.ppm", visibility_threshold.data(), width, height, 1);
    write_ppm(output_dir + "/optix_normals.ppm", normals_vis.data(), width, height, 3);
    write_ppm(output_dir + "/optix_depth.ppm", depth.data(), width, height, 1);

    // Cleanup
    cuda_utils::free_device(d_positions);
    cuda_utils::free_device(d_directions);
    cuda_utils::free_device(d_visibility);
    cuda_utils::free_device(d_normals);
    cuda_utils::free_device(d_depth);

    std::cout << "Grid test complete!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== OptiX Neural Network Test ===" << std::endl;

    // Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> [predictions.bin] [output_dir]" << std::endl;
        std::cerr << "\nweights.bin: Binary weight file (convert from .pth using scripts/convert_checkpoint.py)" << std::endl;
        std::cerr << "predictions.bin: (optional) Ground truth predictions for validation" << std::endl;
        std::cerr << "output_dir: (optional) Output directory for grid test images (default: output)" << std::endl;
        return 1;
    }

    std::string weights_file = argv[1];
    std::string predictions_file = (argc >= 3) ? argv[2] : "";
    std::string output_dir = (argc >= 4) ? argv[3] : "output";

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
        if (!predictions_file.empty()) {
            test_optix_with_predictions(neural_params, predictions_file);
        }

        // Run grid visualization test
        test_simple_grid(neural_params, output_dir);

        std::cout << "\n=== Test Complete ===" << std::endl;
        if (!predictions_file.empty()) {
            std::cout << "\nNote: Compare these metrics with the output from test_network" << std::endl;
            std::cout << "to verify the OptiX implementation matches the tiny-cuda-nn version." << std::endl;
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
