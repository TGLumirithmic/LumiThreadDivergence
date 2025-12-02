/**
 * Phase 1 Test: Neural Network Weight Loading and Inference
 *
 * This test program:
 * 1. Loads PyTorch weights into tiny-cuda-nn
 * 2. Runs test queries at known 3D positions
 * 3. Compares outputs with ground truth predictions (if available)
 * 4. Outputs a simple visualization
 */

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "../src/neural/network.h"
#include "../src/neural/weight_loader.h"
#include "../src/neural/config.h"
#include "../src/utils/cuda_utils.h"
#include "../src/utils/error.h"
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

// Generate test positions in a 2D grid (for visualization)
void generate_test_grid(std::vector<float>& positions,
                       int width, int height,
                       float z_plane = 0.0f) {
    positions.resize(width * height * 3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            // Map pixel coordinates to [-1, 1] range
            positions[idx + 0] = x / float(width);   // x
            positions[idx + 1] = y / float(height);  // y
            positions[idx + 2] = z_plane;                             // z
        }
    }
}

// Test with ground truth predictions if available (batched)
void test_with_predictions(neural::NeuralNetwork& network,
                          const std::string& prediction_file,
                          int batch_size = 256) {
    std::cout << "\n=== Testing with Ground Truth Predictions (Batched) ===" << std::endl;

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
                h_positions[i * 3 + 0] = (sample.origin[0] + 1.0) / 2;
                h_positions[i * 3 + 1] = (sample.origin[1] + 1.0) / 2;
                h_positions[i * 3 + 2] = (sample.origin[2] + 1.0) / 2;

                h_directions[i * 3 + 0] = sample.direction[0];
                h_directions[i * 3 + 1] = sample.direction[1];
                h_directions[i * 3 + 2] = sample.direction[2];
            }

            // Copy to device
            cuda_utils::copy_to_device(d_positions, h_positions.data(), batch_size * 3);
            cuda_utils::copy_to_device(d_directions, h_directions.data(), batch_size * 3);

            // Run batched inference
            network.inference(d_positions, d_directions, d_visibility, d_normals, d_depth, batch_size);
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
        std::cout << "\n=== Aggregated Metrics ===" << std::endl;

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
void test_simple_grid(neural::NeuralNetwork& network,
                     const std::string& output_dir) {
    std::cout << "\n=== Simple Grid Test ===" << std::endl;

    const int width = 512;
    const int height = 512;

    // Generate test positions
    std::vector<float> positions;
    generate_test_grid(positions, width, height, 0.0f);

    std::cout << "Generated " << (width * height) << " test positions" << std::endl;

    // Generate default directions (all pointing forward in +Z)
    std::vector<float> directions(width * height * 3);
    for (int i = 0; i < width * height; ++i) {
        directions[i * 3 + 0] = 0.0f;  // dx
        directions[i * 3 + 1] = 0.0f;  // dy
        directions[i * 3 + 2] = 1.0f;  // dz
    }

    // Allocate device memory
    float* d_positions = cuda_utils::allocate_device<float>(positions.size());
    float* d_directions = cuda_utils::allocate_device<float>(directions.size());
    float* d_visibility = cuda_utils::allocate_device<float>(width * height);
    float* d_normals = cuda_utils::allocate_device<float>(width * height * 3);
    float* d_depth = cuda_utils::allocate_device<float>(width * height);

    // Copy inputs to device
    cuda_utils::copy_to_device(d_positions, positions.data(), positions.size());
    cuda_utils::copy_to_device(d_directions, directions.data(), directions.size());

    // Run inference
    std::cout << "Running inference..." << std::endl;
    network.inference(d_positions, d_directions, d_visibility, d_normals, d_depth,
                     width * height);
    CUDA_SYNC_CHECK();

    // Copy results back
    std::vector<float> visibility(width * height);
    std::vector<float> normals(width * height * 3);
    std::vector<float> depth(width * height);

    cuda_utils::copy_to_host(visibility.data(), d_visibility, width * height);
    cuda_utils::copy_to_host(normals.data(), d_normals, width * height * 3);
    cuda_utils::copy_to_host(depth.data(), d_depth, width * height);

    std::vector<float> visibility_threshold;
    for (auto it = visibility.begin(); it != visibility.end(); it++)
    {
        visibility_threshold.push_back((*it >= 0.5) ? 1.0 : 0.0);
    }

    // Normalize, scale and shift normals to [0, 1]^3 for visualization
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
    write_ppm(output_dir + "/visibility.ppm", visibility_threshold.data(), width, height, 1);
    write_ppm(output_dir + "/normals.ppm", normals_vis.data(), width, height, 3);
    write_ppm(output_dir + "/depth.ppm", depth.data(), width, height, 1);

    // Cleanup
    cuda_utils::free_device(d_positions);
    cuda_utils::free_device(d_directions);
    cuda_utils::free_device(d_visibility);
    cuda_utils::free_device(d_normals);
    cuda_utils::free_device(d_depth);

    std::cout << "Grid test complete!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Phase 1: Neural Network Test ===" << std::endl;

    // Parse arguments
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> [predictions.bin] [output_dir]" << std::endl;
        std::cerr << "\nweights.bin: Binary weight file (convert from .pth using scripts/convert_checkpoint.py)" << std::endl;
        std::cerr << "predictions.bin: Optional ground truth predictions for validation" << std::endl;
        std::cerr << "output_dir: Optional output directory for visualizations (default: output/)" << std::endl;
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

        // Initialize network
        std::cout << "\n=== Initializing Network ===" << std::endl;
        neural::NeuralNetwork network(config);
        if (!network.initialize_from_weights(loader)) {
            std::cerr << "Failed to initialize network" << std::endl;
            return 1;
        }

        // Run tests
        if (!predictions_file.empty()) {
            test_with_predictions(network, predictions_file);
        }

        test_simple_grid(network, output_dir);

        std::cout << "\n=== Test Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
