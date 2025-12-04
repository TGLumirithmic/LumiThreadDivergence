/**
 * OptiX PPM Input Test: Run OptiX neural inference on PPM inputs
 *
 * This test program:
 * 1. Loads PyTorch weights into OptiX neural network format
 * 2. Reads test positions and directions from PPM files
 * 3. Runs neural inference using OptiX kernels
 * 4. Outputs visualization PPMs for visibility, normals, and depth
 */

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <cstdio>

#include "../src/neural/weight_loader.h"
#include "../src/neural/config.h"
#include "../src/optix/neural_params.h"
#include "../src/utils/cuda_utils.h"
#include "../src/utils/error.h"
#include "../programs/neural_inference.cuh"

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

// Test with PPM input files
void test_ppm_input(
    const optix::NeuralNetworkParamsHost& neural_params,
    const std::string& pos_filename,
    const std::string& dir_filename,
    const std::string& output_dir
) {
    std::cout << "\n=== PPM Input Test ===" << std::endl;

    // Read PPM files (returns values in [0, 255])
    std::vector<float> positions = read_ppm(pos_filename);
    std::vector<float> directions = read_ppm(dir_filename);

    if (positions.empty() || directions.empty()) {
        std::cerr << "Failed to read PPM files" << std::endl;
        return;
    }

    // Normalize positions from [0, 255] to [0, 1]
    for (auto it = positions.begin(); it != positions.end(); it++)
    {
        *it = (*it / 255.0f);
    }

    // Normalize directions from [0, 255] to [-1, 1]
    for (auto it = directions.begin(); it != directions.end(); it++)
    {
        *it = (*it / 255.0f) * 2.0f - 1.0f;
    }

    // Calculate dimensions (positions has 3 channels, so divide by 3)
    size_t num_pixels = positions.size() / 3;
    size_t width = static_cast<size_t>(std::sqrt(static_cast<double>(num_pixels)));
    size_t height = num_pixels / width;

    std::cout << "Image dimensions: " << width << "x" << height << std::endl;
    std::cout << "Processing " << num_pixels << " pixels" << std::endl;

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
    std::cout << "Running OptiX neural inference..." << std::endl;
    int threads_per_block = 256;
    int num_blocks = (width * height + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching kernel with " << num_blocks << " blocks, "
              << threads_per_block << " threads per block" << std::endl;

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

    // Threshold visibility for visualization
    std::vector<float> visibility_threshold(width * height);
    for (size_t i = 0; i < width * height; ++i) {
        visibility_threshold[i] = (visibility[i] >= 0.5f) ? 1.0f : 0.0f;
    }

    // Normalize and scale normals to [0, 1]^3 for visualization
    std::vector<float> normals_vis(width * height * 3);
    for (size_t i = 0; i < width * height; ++i) {
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
    write_ppm(output_dir + "/optix_ppm_visibility.ppm", visibility_threshold.data(), width, height, 1);
    write_ppm(output_dir + "/optix_ppm_normals.ppm", normals_vis.data(), width, height, 3);
    write_ppm(output_dir + "/optix_ppm_depth.ppm", depth.data(), width, height, 1);

    // Cleanup
    cuda_utils::free_device(d_positions);
    cuda_utils::free_device(d_directions);
    cuda_utils::free_device(d_visibility);
    cuda_utils::free_device(d_normals);
    cuda_utils::free_device(d_depth);

    std::cout << "PPM input test complete!" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== OptiX Neural Network Test - PPM Input ===" << std::endl;

    // Parse arguments
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <weights.bin> <pos.ppm> <dir.ppm> [output_dir]" << std::endl;
        std::cerr << "\nweights.bin: Binary weight file (convert from .pth using scripts/convert_checkpoint.py)" << std::endl;
        std::cerr << "pos.ppm: PPM file with test positions (from OptiX render)" << std::endl;
        std::cerr << "dir.ppm: PPM file with test directions (from OptiX render)" << std::endl;
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

        // Convert to OptiX format
        std::cout << "\n=== Converting to OptiX Format ===" << std::endl;
        optix::NeuralNetworkParamsHost neural_params(config);
        if (!neural_params.load_from_weights(loader)) {
            std::cerr << "Failed to convert weights to OptiX format" << std::endl;
            return 1;
        }
        std::cout << "Successfully converted weights to OptiX format" << std::endl;

        // Run PPM input test
        test_ppm_input(neural_params, pos_filename, dir_filename, output_dir);

        std::cout << "\n=== Test Complete ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
