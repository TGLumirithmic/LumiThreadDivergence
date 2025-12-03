#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <optix.h>
#include <cuda_runtime.h>

#include "optix/context.h"
#include "optix/pipeline.h"
#include "optix/geometry.h"
#include "optix/sbt.h"
#include "optix/neural_params.h"
#include "neural/weight_loader.h"
#include "neural/config.h"
#include "utils/error.h"

// Simple 3D vector structure (matches common.h)
struct float3_aligned {
    float x, y, z;
};

// Camera parameters
struct Camera {
    float3_aligned position;
    float3_aligned u, v, w;  // Camera basis vectors
    float fov;
};

// Neural asset bounds
struct NeuralAssetBounds {
    float3_aligned min;
    float3_aligned max;
};

// Launch parameters - must exactly match programs/common.h
struct LaunchParams {
    uchar4* frame_buffer;
    uint32_t width;
    uint32_t height;
    Camera camera;
    OptixTraversableHandle traversable;
    NeuralNetworkParams neural_network;
    NeuralAssetBounds neural_bounds;
    float3_aligned background_color;
};

// Simple PPM image writer
void write_ppm(const std::string& filename, const uchar4* pixels, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.put(pixels[i].x);
        file.put(pixels[i].y);
        file.put(pixels[i].z);
    }

    std::cout << "Wrote image to: " << filename << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== OptiX Neural Renderer - Phase 2 ===" << std::endl;

    // Parse command line arguments
    std::string weight_file = "data/models/weights.bin";
    std::string output_file = "output/neural_render.ppm";
    int width = 512;
    int height = 512;

    if (argc > 1) weight_file = argv[1];
    if (argc > 2) output_file = argv[2];
    if (argc > 3) width = std::atoi(argv[3]);
    if (argc > 4) height = std::atoi(argv[4]);

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Weight file: " << weight_file << std::endl;
    std::cout << "  Output: " << output_file << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;

    // Initialize OptiX context
    optix::Context context;
    if (!context.initialize()) {
        std::cerr << "Failed to initialize OptiX context" << std::endl;
        return 1;
    }

    // Configure neural network
    neural::NetworkConfig network_config;
    network_config.n_levels = 16;
    network_config.n_features_per_level = 2;
    network_config.log2_hashmap_size = 14;
    network_config.base_resolution = 16.0f;
    network_config.max_resolution = 1024.0f;
    network_config.n_neurons = 32;
    network_config.direction_hidden_dim = 16;
    network_config.direction_n_hidden_layers = 1;
    network_config.visibility_decoder.n_decoder_layers = 4;
    network_config.normal_decoder.n_decoder_layers = 4;
    network_config.depth_decoder.n_decoder_layers = 4;

    // Load weights and convert to OptiX format
    neural::WeightLoader weight_loader;
    optix::NeuralNetworkParamsHost neural_params(network_config);

    bool weights_loaded = false;
    if (weight_loader.load_from_file(weight_file)) {
        std::cout << "Loaded weights from: " << weight_file << std::endl;
        if (neural_params.load_from_weights(weight_loader)) {
            std::cout << "Neural network parameters converted for OptiX" << std::endl;
            weights_loaded = true;
        } else {
            std::cerr << "Warning: Failed to convert weights for OptiX" << std::endl;
        }
    } else {
        std::cerr << "Warning: Could not load weights from " << weight_file << std::endl;
        std::cerr << "You need to provide a valid weight file to render neural assets" << std::endl;
        return 1;
    }

    // Build OptiX pipeline
    optix::Pipeline pipeline(context);
    if (!pipeline.build("build/lib")) {
        std::cerr << "Failed to build OptiX pipeline" << std::endl;
        std::cerr << "Make sure PTX files are in build/lib/ directory" << std::endl;
        return 1;
    }

    // Define neural asset bounds (simple cube centered at origin)
    float3 neural_min = make_float3(-1.0f, -1.0f, -1.0f);
    float3 neural_max = make_float3(1.0f, 1.0f, 1.0f);

    // Build geometry (BLAS for neural asset)
    optix::GeometryBuilder geom_builder(context);
    OptixTraversableHandle traversable = geom_builder.build_neural_asset_blas(
        neural_min, neural_max);

    // Build shader binding table
    optix::ShaderBindingTable sbt(context, pipeline);
    if (!sbt.build()) {
        std::cerr << "Failed to build shader binding table" << std::endl;
        return 1;
    }

    // Allocate frame buffer
    uchar4* d_frame_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frame_buffer, width * height * sizeof(uchar4)));

    // Set up launch parameters
    LaunchParams launch_params = {};
    launch_params.frame_buffer = d_frame_buffer;
    launch_params.width = width;
    launch_params.height = height;
    launch_params.traversable = traversable;

    // Copy neural network parameters
    launch_params.neural_network = neural_params.get_device_params();

    // Set up camera (looking at the cube from a distance)
    float camera_distance = 3.0f;
    launch_params.camera.position = {0.0f, 0.0f, camera_distance};
    launch_params.camera.fov = 45.0f;

    // Camera basis vectors (standard OpenGL-style camera)
    float aspect = (float)width / (float)height;
    float vfov = launch_params.camera.fov * M_PI / 180.0f;
    float vfov_size = std::tan(vfov / 2.0f);

    launch_params.camera.u = {aspect * vfov_size, 0.0f, 0.0f};
    launch_params.camera.v = {0.0f, vfov_size, 0.0f};
    launch_params.camera.w = {0.0f, 0.0f, -1.0f};

    // Neural bounds
    launch_params.neural_bounds.min = {neural_min.x, neural_min.y, neural_min.z};
    launch_params.neural_bounds.max = {neural_max.x, neural_max.y, neural_max.z};

    // Background color (dark gray)
    launch_params.background_color = {0.1f, 0.1f, 0.15f};

    // Upload launch parameters to device
    LaunchParams* d_launch_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_launch_params, sizeof(LaunchParams)));
    CUDA_CHECK(cudaMemcpy(
        d_launch_params,
        &launch_params,
        sizeof(LaunchParams),
        cudaMemcpyHostToDevice));

    // Launch rendering
    std::cout << "\nLaunching OptiX rendering..." << std::endl;

    OptixResult result = optixLaunch(
        pipeline.get(),
        context.get_stream(),
        (CUdeviceptr)d_launch_params,
        sizeof(LaunchParams),
        &sbt.get(),
        width,
        height,
        1  // depth
    );

    if (result != OPTIX_SUCCESS) {
        std::cerr << "optixLaunch failed: " << optixGetErrorName(result) << std::endl;
        std::cerr << "  " << optixGetErrorString(result) << std::endl;
        return 1;
    }

    CUDA_CHECK(cudaStreamSynchronize(context.get_stream()));
    std::cout << "Rendering complete!" << std::endl;

    // Download frame buffer
    std::vector<uchar4> h_frame_buffer(width * height);
    CUDA_CHECK(cudaMemcpy(
        h_frame_buffer.data(),
        d_frame_buffer,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost));

    // Write output image
    write_ppm(output_file, h_frame_buffer.data(), width, height);

    // Cleanup
    cudaFree(d_frame_buffer);
    cudaFree(d_launch_params);

    std::cout << "\n=== Rendering Complete ===" << std::endl;
    std::cout << "Output saved to: " << output_file << std::endl;
    return 0;
}
