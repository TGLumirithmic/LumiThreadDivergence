#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#include <optix.h>
#include <cuda_runtime.h>

#include "optix/context.h"
#include "optix/pipeline.h"
#include "optix/geometry.h"
#include "optix/triangle_geometry.h"
#include "optix/tlas_builder.h"
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

// Point light structure
struct PointLight {
    float3_aligned position;
    float3_aligned color;
    float intensity;
};

// Launch parameters - must exactly match programs/common.h
struct LaunchParams {
    uchar4* frame_buffer;
    uchar4* position_buffer;
    uchar4* direction_buffer;
    float3_aligned* unnormalized_position_buffer;
    uint32_t width;
    uint32_t height;
    Camera camera;
    OptixTraversableHandle traversable;
    NeuralNetworkParams neural_network;
    NeuralAssetBounds neural_bounds;
    PointLight light;
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
    std::cout << "=== OptiX Neural Renderer - Phase 3 ===" << std::endl;

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

    // Build triangle geometry BLASes
    std::cout << "\nBuilding triangle geometry..." << std::endl;
    optix::TriangleGeometry triangle_geom(context);
    OptixTraversableHandle floor_blas = triangle_geom.build_floor_blas();
    OptixTraversableHandle walls_blas = triangle_geom.build_walls_blas();

    // Build neural asset BLAS
    std::cout << "\nBuilding neural asset geometry..." << std::endl;
    optix::GeometryBuilder neural_geom(context);
    OptixTraversableHandle neural_blas = neural_geom.build_neural_asset_blas(
        neural_min, neural_max);

    // Build TLAS with all instances
    std::cout << "\nBuilding TLAS with mixed geometry..." << std::endl;
    optix::TLASBuilder tlas_builder(context);

    // Identity transform
    float identity[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    // Add floor instance (instanceId=0, sbtOffset=0 for triangles)
    tlas_builder.add_instance(floor_blas, 0, 0, identity);

    // Add walls instance (instanceId=1, sbtOffset=0 for triangles)
    tlas_builder.add_instance(walls_blas, 1, 0, identity);

    // Add neural asset instance (instanceId=2, sbtOffset=2 for neural)
    // Position it 1 unit above the floor (floor is at Y=-1, neural BLAS is [-1,1], so translate by 1.0 to sit on floor)
    float neural_transform[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 1.0f,  // Translate Y=1.0 (places bottom of neural asset at Y=0, which is 1 unit above floor at Y=-1)
        0.0f, 0.0f, 1.0f, 0.0f
    };
    tlas_builder.add_instance(neural_blas, 2, 2, neural_transform);

    // Build the TLAS
    OptixTraversableHandle traversable = tlas_builder.build();

    // Build shader binding table
    optix::ShaderBindingTable sbt(context, pipeline);
    if (!sbt.build()) {
        std::cerr << "Failed to build shader binding table" << std::endl;
        return 1;
    }

    // Allocate frame buffers
    uchar4* d_frame_buffer = nullptr;
    uchar4* d_position_buffer = nullptr;
    uchar4* d_direction_buffer = nullptr;
    float3_aligned* d_unnormalized_position_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frame_buffer, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_position_buffer, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_direction_buffer, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_unnormalized_position_buffer, width * height * sizeof(float3_aligned)));

    // Set up launch parameters
    LaunchParams launch_params = {};
    launch_params.frame_buffer = d_frame_buffer;
    launch_params.position_buffer = d_position_buffer;
    launch_params.direction_buffer = d_direction_buffer;
    launch_params.unnormalized_position_buffer = d_unnormalized_position_buffer;
    launch_params.width = width;
    launch_params.height = height;
    launch_params.traversable = traversable;

    // Copy neural network parameters
    launch_params.neural_network = neural_params.get_device_params();

    // Set up camera (view from angle to see both floor and neural asset)
    launch_params.camera.position = {3.0f, 2.0f, 5.0f};
    launch_params.camera.fov = 90.0f;

    // Camera basis vectors (looking toward origin)
    float aspect = (float)width / (float)height;
    float vfov = launch_params.camera.fov * M_PI / 180.0f;
    float vfov_size = std::tan(vfov / 2.0f);

    launch_params.camera.u = {aspect * vfov_size, 0.0f, 0.0f};
    launch_params.camera.v = {0.0f, vfov_size, 0.0f};
    launch_params.camera.w = {0.0f, 0.0f, -1.0f};

    // Neural bounds
    launch_params.neural_bounds.min = {neural_min.x, neural_min.y, neural_min.z};
    launch_params.neural_bounds.max = {neural_max.x, neural_max.y, neural_max.z};

    // Setup point light (above the scene)
    launch_params.light.position = {0.0f, 3.0f, 0.0f};
    launch_params.light.color = {1.0f, 1.0f, 1.0f};
    launch_params.light.intensity = 100.0f;

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

    // Download frame buffers
    std::vector<uchar4> h_frame_buffer(width * height);
    std::vector<uchar4> h_position_buffer(width * height);
    std::vector<uchar4> h_direction_buffer(width * height);
    std::vector<float3_aligned> h_unnormalized_position_buffer(width * height);

    CUDA_CHECK(cudaMemcpy(
        h_frame_buffer.data(),
        d_frame_buffer,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_position_buffer.data(),
        d_position_buffer,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_direction_buffer.data(),
        d_direction_buffer,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_unnormalized_position_buffer.data(),
        d_unnormalized_position_buffer,
        width * height * sizeof(float3_aligned),
        cudaMemcpyDeviceToHost));

    // Write output images
    write_ppm(output_file, h_frame_buffer.data(), width, height);

    // Generate filenames for position and direction
    std::string pos_file = output_file;
    std::string dir_file = output_file;
    size_t dot_pos = pos_file.find_last_of('.');
    if (dot_pos != std::string::npos) {
        pos_file = pos_file.substr(0, dot_pos) + "_position.ppm";
        dir_file = dir_file.substr(0, dot_pos) + "_direction.ppm";
    } else {
        pos_file += "_position.ppm";
        dir_file += "_direction.ppm";
    }

    write_ppm(pos_file, h_position_buffer.data(), width, height);
    write_ppm(dir_file, h_direction_buffer.data(), width, height);

    // Write unnormalized positions as binary file for analysis
    std::string unnorm_file = output_file;
    if (dot_pos != std::string::npos) {
        unnorm_file = unnorm_file.substr(0, dot_pos) + "_position_unnormalized.bin";
    } else {
        unnorm_file += "_position_unnormalized.bin";
    }

    std::ofstream bin_file(unnorm_file, std::ios::binary);
    if (bin_file.is_open()) {
        bin_file.write(reinterpret_cast<const char*>(h_unnormalized_position_buffer.data()),
                       width * height * sizeof(float3_aligned));
        bin_file.close();
        std::cout << "Wrote unnormalized positions to: " << unnorm_file << std::endl;
    }

    // Cleanup
    cudaFree(d_frame_buffer);
    cudaFree(d_position_buffer);
    cudaFree(d_direction_buffer);
    cudaFree(d_unnormalized_position_buffer);
    cudaFree(d_launch_params);

    std::cout << "\n=== Rendering Complete ===" << std::endl;
    std::cout << "Output saved to: " << output_file << std::endl;
    return 0;
}
