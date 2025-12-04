/**
 * Phase 1 Test: Neural Network Weight Loading and Inference
 *
 * This test program:
 * 1. Loads PyTorch weights into tiny-cuda-nn
 * 2. Loads test positions and directions from ppm files
 * 3. Renders using the neural network
 * 4. Outputs a simple visualization
 */
#include <iomanip>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>

#include "../src/neural/network.h"
#include "../src/neural/weight_loader.h"
#include "../src/neural/config.h"
#include "../src/utils/cuda_utils.h"
#include "../src/utils/error.h"
#include "../src/utils/debug_utils.h"
#include "neural_proxy_predictions.h"

// Simple PPM image reader
std::vector<float> read_ppm(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        std::cerr << "Failed to open file for reading: " << filename << std::endl;
        return {};
    }

    // Read PPM header
    char magic[3];
    int width, height, max_val;
    if (fscanf(fp, "%2s\n%d %d\n%d\n", magic, &width, &height, &max_val) != 4) {
        std::cerr << "Failed to parse PPM header: " << filename << std::endl;
        fclose(fp);
        return {};
    }

    if (std::string(magic) != "P6") {
        std::cerr << "Only P6 (binary RGB) PPM format supported" << std::endl;
        fclose(fp);
        return {};
    }

    // Read pixel data
    std::vector<unsigned char> pixels(width * height * 3);
    size_t bytes_read = fread(pixels.data(), 1, pixels.size(), fp);
    fclose(fp);

    if (bytes_read != pixels.size()) {
        std::cerr << "Failed to read all pixel data from: " << filename << std::endl;
        return {};
    }

    // Convert to float
    std::vector<float> data(width * height * 3);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(pixels[i]);
    }

    std::cout << "Read PPM: " << filename << " (" << width << "x" << height << ")" << std::endl;
    return data;
}

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

// Simple grid test (no ground truth)
void test_ppm_input(neural::NeuralNetwork& network,
                    const std::string& pos_filename,
                    const std::string& dir_filename,
                     const std::string& output_dir) {
    std::cout << "\n=== PPM input Test ===" << std::endl;

    // Read PPM files (returns values in [0, 255])
    std::vector<float> positions = read_ppm(pos_filename);
    std::vector<float> directions = read_ppm(dir_filename);

    if (positions.empty() || directions.empty()) {
        std::cerr << "Failed to read PPM files" << std::endl;
        return;
    }

    // Normalize positions from [0, 255] to [0, 1]
    for (auto pos_it = positions.begin(); pos_it != positions.end(); pos_it++)
    {
        *pos_it = (*pos_it / 255.0f);
    }

    // Normalize directions from [0, 255] to [-1, 1]
    for (auto dir_it = directions.begin(); dir_it != directions.end(); dir_it++)
    {
        *dir_it = (*dir_it / 255.0f) * 2.0f - 1.0f;
    }

    // Calculate dimensions (positions has 3 channels, so divide by 3)
    size_t num_pixels = positions.size() / 3;
    size_t width = static_cast<size_t>(std::sqrt(static_cast<double>(num_pixels)));
    size_t height = num_pixels / width;

    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Read " << num_pixels << " test positions" << std::endl;

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
    write_ppm(output_dir + "/ppm_visibility.ppm", visibility.data(), width, height, 1);
    write_ppm(output_dir + "/ppm_normals.ppm", normals_vis.data(), width, height, 3);
    write_ppm(output_dir + "/ppm_depth.ppm", depth.data(), width, height, 1);

    // Cleanup
    cuda_utils::free_device(d_positions);
    cuda_utils::free_device(d_directions);
    cuda_utils::free_device(d_visibility);
    cuda_utils::free_device(d_normals);
    cuda_utils::free_device(d_depth);

    std::cout << "Grid test complete!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Phase 1: Neural Network Test - PPM input ===" << std::endl;

    // Parse arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> <pos.ppm> <dir.ppm> [output_dir]" << std::endl;
        std::cerr << "\nweights.bin: Binary weight file (convert from .pth using scripts/convert_checkpoint.py)" << std::endl;
        std::cerr << "pos.ppm: PPM file with test positions" << std::endl;
        std::cerr << "dir.ppm: PPM file with test directions" << std::endl;
        std::cerr << "output_dir: Optional output directory for visualizations (default: output/)" << std::endl;
        return 1;
    }

    std::string weights_file = argv[1];
    std::string pos_filename = argv[2];
    std::string dir_filename = argv[3];
    std::string output_dir = (argc >= 5) ? argv[4] : "output";

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

        test_ppm_input(network, pos_filename, dir_filename, output_dir);

        std::cout << "\n=== Test Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
