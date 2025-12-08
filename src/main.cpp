#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>

#include <yaml-cpp/yaml.h>
#include <tiny_obj_loader.h>

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

// Divergence metric indices (must match programs/common.h)
#define DIVERGENCE_RAYGEN 0
#define DIVERGENCE_INTERSECTION 1
#define DIVERGENCE_CLOSESTHIT 2
#define DIVERGENCE_SHADOW 3
#define DIVERGENCE_HASH_ENCODING 4
#define DIVERGENCE_MLP_FORWARD 5
#define DIVERGENCE_EARLY_REJECT 6
#define DIVERGENCE_HIT_MISS 7
#define DIVERGENCE_INSTANCE_ENTROPY 8
#define NUM_DIVERGENCE_METRICS 9

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
    float3_aligned* hit_position_buffer;  // World-space hit position (full precision)
    int32_t* instance_id_buffer;          // Instance ID per pixel (-1 for miss)
    uint32_t width;
    uint32_t height;
    Camera camera;
    OptixTraversableHandle traversable;
    // Support multiple neural assets: pointer to an array of NeuralNetworkParams
    uint32_t num_neural_assets;
    NeuralNetworkParams* neural_networks; // device pointer to array
    NeuralAssetBounds* neural_bounds_array; // device pointer to array
    
    int* instance_to_neural_map;
    
    NeuralAssetBounds neural_bounds;
    NeuralNetworkParams neural_network; // device pointer to array
    
    PointLight light;
    float3_aligned background_color;

    // Warp divergence profiling output
    uint32_t* divergence_buffer;
};

// Simple scene representation
struct SceneObject {
    // Identity transform for convenience
    float identity[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    enum Type { MESH, NEURAL } type;
    std::string file;       // for meshes
    std::string weights;    // for neural assets
    float3 bounds_min;
    float3 bounds_max;
    bool has_bounds = false;
    float transform[12]; // 3x4 row-major
    SceneObject() { for (int i=0;i<12;i++) transform[i]=identity[i]; }
};

float3 readFloat3(const YAML::Node& node) {
    return make_float3(
        node[0].as<float>(),
        node[1].as<float>(),
        node[2].as<float>()
    );
}

float3_aligned readFloat3Aligned(const YAML::Node& node) {
    float3 vec = make_float3(
        node[0].as<float>(),
        node[1].as<float>(),
        node[2].as<float>()
    );

    return { vec.x, vec.y, vec.z };
}

// Build a very basic transform: T = translate * scale (no rotation for simplicity)
void buildTransform(const YAML::Node& tnode, float out[12]) {
    // Start as identity
    float T[12] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0
    };

    // Scale
    if (tnode["scale"]) {
        float3 s = readFloat3(tnode["scale"]);
        T[0] *= s.x;  T[1] *= s.x;  T[2]  *= s.x;
        T[4] *= s.y;  T[5] *= s.y;  T[6]  *= s.y;
        T[8] *= s.z;  T[9] *= s.z;  T[10] *= s.z;
    }

    // Position (translation)
    if (tnode["position"]) {
        float3 p = readFloat3(tnode["position"]);
        T[3]  = p.x;
        T[7]  = p.y;
        T[11] = p.z;
    }

    // Rotation ignored for minimal example, but can be added easily.

    // Copy out
    for (int i = 0; i < 12; i++) out[i] = T[i];
}

std::vector<SceneObject> load_scene_objects(const YAML::Node& root) {
    std::vector<SceneObject> objects;

    auto obj_list = root["scene"]["objects"];
    if (!obj_list) return objects;

    for (const auto& node : obj_list) {
        SceneObject obj;

        // --- TYPE ---
        std::string t = node["type"].as<std::string>();
        if (t == "mesh")
            obj.type = SceneObject::MESH;
        else if (t == "neural_asset")
            obj.type = SceneObject::NEURAL;
        else
            continue; // unknown type

        // --- FILE / WEIGHTS ---
        if (node["file"])
            obj.file = node["file"].as<std::string>();

        if (node["weights"])
            obj.weights = node["weights"].as<std::string>();

        // --- BOUNDS ---
        if (node["bounds"]) {
            auto b = node["bounds"];
            obj.bounds_min = readFloat3(b["min"]);
            obj.bounds_max = readFloat3(b["max"]);
            obj.has_bounds = true;
        }

        // --- TRANSFORM ---
        if (node["transform"])
            buildTransform(node["transform"], obj.transform);

        objects.push_back(obj);
    }

    return objects;
}

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
    std::cout << "=== OptiX Neural Renderer - Phase 3/4 ===" << std::endl;

    // Parse command line arguments
    // Backwards-compatible defaults
    std::string weight_file = "data/models/weights.bin";
    std::string output_file = "output/neural_render.ppm";
    int width = 512;
    int height = 512;
    std::string scene_file; // optional YAML scene file

    // Simple CLI: <program> [weight_or_scene] [output] [width] [height]
    if (argc > 1) {
        std::string a1 = argv[1];
        // treat as scene file if it ends with .yaml or .yml
        if (a1.size() > 5 && (a1.substr(a1.size()-5) == ".yaml" || a1.substr(a1.size()-4) == ".yml")) {
            scene_file = a1;
        } else {
            weight_file = a1;
        }
    }
    if (argc > 2) output_file = argv[2];
    if (argc > 3) width = std::atoi(argv[3]);
    if (argc > 4) height = std::atoi(argv[4]);

    std::cout << "Configuration:" << std::endl;
    if (!scene_file.empty()) std::cout << "  Scene file: " << scene_file << std::endl;
    else std::cout << "  Weight file: " << weight_file << std::endl;
    std::cout << "  Output: " << output_file << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    YAML::Node root;

    try {
        root = YAML::LoadFile(scene_file);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load YAML: " << e.what() << std::endl;
        return 1;
    }

    // --- CAMERA ---
    const auto camera = root["scene"]["camera"];
    float3 cam_pos    = readFloat3(camera["position"]);
    float3 cam_look   = readFloat3(camera["look_at"]);
    float fov       = camera["fov"].as<float>();

    std::cout << "Camera:\n";
    std::cout << "  Position: " << cam_pos.x << ", " << cam_pos.y << ", " << cam_pos.z << "\n";
    std::cout << "  Look At:  " << cam_look.x << ", " << cam_look.y << ", " << cam_look.z << "\n";
    std::cout << "  FOV:      " << fov << "\n\n";

    // --- LIGHTS ---
    // const auto lights = root["scene"]["lights"];
    // std::cout << "Lights:\n";
    // for (const auto& light : lights) {
    //     std::string type = light["type"].as<std::string>();
    //     auto pos = readFloat3(light["position"]);
    //     auto intensity = readFloat3(light["intensity"]);

    //     std::cout << "  Light (" << type << ")\n";
    //     std::cout << "    Position:  " 
    //               << pos.x << ", " << pos.y << ", " << pos.z << "\n";
    //     std::cout << "    Intensity: " 
    //               << intensity.x << ", " << intensity.y << ", " << intensity.z << "\n";
    // }
    // std::cout << "\n";

    
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
    // neural::WeightLoader weight_loader;
    // optix::NeuralNetworkParamsHost neural_params(network_config);

    // bool weights_loaded = false;
    // if (!scene_file.empty()) {
    //     // If a scene file is provided, weight loading will be handled per-object later
    //     std::cout << "Scene mode: deferring weight loading to per-object handling" << std::endl;
    // } else {
    //     if (weight_loader.load_from_file(weight_file)) {
    //         std::cout << "Loaded weights from: " << weight_file << std::endl;
    //         if (neural_params.load_from_weights(weight_loader)) {
    //             std::cout << "Neural network parameters converted for OptiX" << std::endl;
    //             weights_loaded = true;
    //         } else {
    //             std::cerr << "Warning: Failed to convert weights for OptiX" << std::endl;
    //         }
    //     } else {
    //         std::cerr << "Warning: Could not load weights from " << weight_file << std::endl;
    //         std::cerr << "You need to provide a valid weight file to render neural assets" << std::endl;
    //         return 1;
    //     }
    // }

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

    // Build TLAS with mixed geometry driven by an optional scene file
    std::cout << "\nBuilding geometry and TLAS..." << std::endl;
    optix::TriangleGeometry triangle_geom(context);
    optix::GeometryBuilder neural_geom(context);
    optix::TLASBuilder tlas_builder(context);
    
    auto trim = [](std::string s) {
        while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
        while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
        return s;
    };

    auto parse_vec3 = [&](const std::string& token)->float3 {
        float3 v = make_float3(0.0f,0.0f,0.0f);
        auto l = token.find('[');
        auto r = token.find(']');
        if (l!=std::string::npos && r!=std::string::npos && r>l) {
            std::string inside = token.substr(l+1, r-l-1);
            std::replace(inside.begin(), inside.end(), ',', ' ');
            std::istringstream iss(inside);
            iss >> v.x >> v.y >> v.z;
        }
        return v;
    };

    // std::vector<SceneObject> scene_objects;
    // if (!scene_file.empty()) {
    std::vector<SceneObject> scene_objects = load_scene_objects(root);

    std::cout << "Parsed " << scene_objects.size() << " objects from scene file." << std::endl;
    // } else {
    //     // Default simple scene (floor + walls + one neural asset)
    //     SceneObject floor_obj;
    //     floor_obj.type = SceneObject::MESH;
    //     floor_obj.file = "floor";
    //     scene_objects.push_back(floor_obj);

    //     SceneObject walls_obj;
    //     walls_obj.type = SceneObject::MESH;
    //     walls_obj.file = "walls";
    //     scene_objects.push_back(walls_obj);

    //     SceneObject neural_obj;
    //     neural_obj.type = SceneObject::NEURAL;
    //     neural_obj.has_bounds = true;
    //     neural_obj.bounds_min = make_float3(-1.0f,-1.0f,-1.0f);
    //     neural_obj.bounds_max = make_float3(1.0f,1.0f,1.0f);
    //     // translate neural asset 1 unit up like before
    //     neural_obj.transform[7] = 1.0f; // Y translation (index 7)
    //     scene_objects.push_back(neural_obj);
    // }

    // Allocate frame buffers
    uchar4* d_frame_buffer = nullptr;
    float3_aligned* d_hit_position_buffer = nullptr;
    int32_t* d_instance_id_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_frame_buffer, width * height * sizeof(uchar4)));
    CUDA_CHECK(cudaMalloc(&d_hit_position_buffer, width * height * sizeof(float3_aligned)));
    CUDA_CHECK(cudaMalloc(&d_instance_id_buffer, width * height * sizeof(int32_t)));

    // Allocate divergence profiling buffer
    uint32_t* d_divergence_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_divergence_buffer, width * height * NUM_DIVERGENCE_METRICS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_divergence_buffer, 0, width * height * NUM_DIVERGENCE_METRICS * sizeof(uint32_t)));

    // Set up launch parameters
    LaunchParams launch_params = {};
    launch_params.frame_buffer = d_frame_buffer;
    launch_params.hit_position_buffer = d_hit_position_buffer;
    launch_params.instance_id_buffer = d_instance_id_buffer;
    launch_params.width = width;
    launch_params.height = height;
    launch_params.divergence_buffer = d_divergence_buffer;

    // Build BLASes and add instances
    const int TRIANGLE_SBT_OFFSET = 0; // must match SBT hitgroup ordering
    const int NEURAL_SBT_OFFSET = 2;   // must match SBT hitgroup ordering used in programs

    int instance_id = 0;

    // Storage for per-neural-asset parameters (keep hosts alive)
    std::vector<std::unique_ptr<optix::NeuralNetworkParamsHost>> neural_hosts;
    std::vector<NeuralNetworkParams> host_nn_params_values;
    std::vector<NeuralAssetBounds> host_neural_bounds;

    // Mapping from instance ID to neural asset index (-1 for non-neural instances)
    std::vector<int> instance_to_neural_vec;

    for (auto &obj : scene_objects) {
        if (obj.type == SceneObject::MESH) {
            instance_to_neural_vec.push_back(-1);  // Not a neural asset
            OptixTraversableHandle mesh_blas = 0;
            void* d_vertex_buffer = nullptr;
            void* d_index_buffer = nullptr;

            if (obj.file == "floor") {
                mesh_blas = triangle_geom.build_floor_blas();
                d_vertex_buffer = triangle_geom.get_floor_vertex_buffer();
                d_index_buffer = triangle_geom.get_floor_index_buffer();
            } else if (obj.file == "walls") {
                mesh_blas = triangle_geom.build_walls_blas();
                d_vertex_buffer = triangle_geom.get_walls_vertex_buffer();
                d_index_buffer = triangle_geom.get_walls_index_buffer();
            } else if (obj.file.size() > 4 && obj.file.substr(obj.file.size()-4) == ".obj") {
                // Load mesh from OBJ file using tinyobjloader
                std::vector<Vertex> vertices;
                std::vector<uint3> indices;
                tinyobj::ObjReaderConfig config;
                config.vertex_color = false;
                tinyobj::ObjReader reader;
                if (!reader.ParseFromFile(obj.file, config)) {
                    std::cerr << "Failed to load OBJ file: " << obj.file << "\n" << reader.Error() << std::endl;
                } else {
                    const auto& attrib = reader.GetAttrib();
                    const auto& shapes = reader.GetShapes();
                    // Build vertices
                    for (size_t v = 0; v < attrib.vertices.size() / 3; ++v) {
                        Vertex vert;
                        vert.position.x = attrib.vertices[3*v+0];
                        vert.position.y = attrib.vertices[3*v+1];
                        vert.position.z = attrib.vertices[3*v+2];
                        // If normals present, use them
                        if (attrib.normals.size() >= 3*(v+1)) {
                            vert.normal.x = attrib.normals[3*v+0];
                            vert.normal.y = attrib.normals[3*v+1];
                            vert.normal.z = attrib.normals[3*v+2];
                        } else {
                            vert.normal = {0.0f, 1.0f, 0.0f};
                        }
                        vertices.push_back(vert);
                    }
                    // Build indices
                    for (const auto& shape : shapes) {
                        size_t index_offset = 0;
                        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
                            int fv = shape.mesh.num_face_vertices[f];
                            if (fv == 3) {
                                uint3 tri;
                                tri.x = shape.mesh.indices[index_offset+0].vertex_index;
                                tri.y = shape.mesh.indices[index_offset+1].vertex_index;
                                tri.z = shape.mesh.indices[index_offset+2].vertex_index;
                                indices.push_back(tri);
                            }
                            index_offset += fv;
                        }
                    }
                    void* d_blas_buffer = nullptr;
                    size_t blas_size = 0;
                    mesh_blas = triangle_geom.build_mesh_blas(vertices, indices, &d_vertex_buffer, &d_index_buffer, &d_blas_buffer, &blas_size);
                }
            } else {
                // Unknown mesh file -- fallback to floor for now
                std::cerr << "Warning: mesh file '" << obj.file << "' not supported; using floor placeholder." << std::endl;
                mesh_blas = triangle_geom.build_floor_blas();
                d_vertex_buffer = triangle_geom.get_floor_vertex_buffer();
                d_index_buffer = triangle_geom.get_floor_index_buffer();
            }

            // Create metadata with buffer pointers
            optix::InstanceMetadata metadata = {};
            metadata.type = optix::GeometryType::TRIANGLE_MESH;
            metadata.neural_params_device = nullptr;
            metadata.albedo = {0.8f, 0.8f, 0.8f};
            metadata.roughness = 0.5f;
            metadata.vertex_buffer = d_vertex_buffer;
            metadata.index_buffer = d_index_buffer;

            tlas_builder.add_instance_with_metadata(mesh_blas, metadata, obj.transform);
        } else { // NEURAL
            
            // bool weights_loaded = false;
            // If the scene specified weights for this neural object, attempt to load and convert
            if (!obj.weights.empty()) {
                auto host = std::make_unique<optix::NeuralNetworkParamsHost>(network_config);
                neural::WeightLoader local_loader;
                if (local_loader.load_from_file(obj.weights)) {
                    if (host->load_from_weights(local_loader)) {
                        std::cout << "Loaded weights for neural object from: " << obj.weights << std::endl;
                        
                        // Record mapping from this instance ID to neural asset index
                        instance_to_neural_vec.push_back(launch_params.num_neural_assets);
                        
                        float3 bmin = neural_min;
                        float3 bmax = neural_max;
                        if (obj.has_bounds) { bmin = obj.bounds_min; bmax = obj.bounds_max; }
                        
                        OptixTraversableHandle n_blas = neural_geom.build_neural_asset_blas(bmin, bmax);
                        tlas_builder.add_instance(n_blas, instance_id, optix::GeometryType::NEURAL_ASSET, obj.transform);
                        std::cout << "Converted neural params for this object." << std::endl;
                    
                        // Keep host alive by moving into vector
                        host_nn_params_values.push_back(host->get_device_params());
                        host_neural_bounds.push_back({{obj.bounds_min.x, obj.bounds_min.y, obj.bounds_min.z}, {obj.bounds_max.x, obj.bounds_max.y, obj.bounds_max.z}});
                        neural_hosts.push_back(std::move(host));
                        
                        launch_params.num_neural_assets++;
                        instance_id++;
                    } else {
                        std::cerr << "Warning: failed to convert weights for object: " << obj.weights << std::endl;
                    }
                } else {
                    std::cerr << "Warning: could not load weights file: " << obj.weights << std::endl;
                }
            // } else {
            //     // No per-object weights specified: if a global weights set was loaded earlier, reuse it
            //     if (weights_loaded) {
            //         host_nn_params_values.push_back(neural_params.get_device_params());
            //         host_neural_bounds.push_back({{bmin.x, bmin.y, bmin.z}, {bmax.x, bmax.y, bmax.z}});
            //     } else {
            //         // no weights available for this neural object
            //         std::cerr << "Warning: neural object has no weights and no global weights provided; it will use default (possibly empty) params." << std::endl;
            //         host_nn_params_values.push_back(neural_params.get_device_params());
            //         host_neural_bounds.push_back({{bmin.x, bmin.y, bmin.z}, {bmax.x, bmax.y, bmax.z}});
            //     }
            }
        }
    }

    // Build the TLAS
    launch_params.traversable = tlas_builder.build();

    // Build shader binding table with instance metadata
    optix::ShaderBindingTable sbt(context, pipeline);
    if (!sbt.build(tlas_builder.get_instance_metadata())) {
        std::cerr << "Failed to build shader binding table" << std::endl;
        return 1;
    }

    // Copy neural network parameters (support multiple neural assets)
    NeuralNetworkParams* d_neural_array = nullptr;
    NeuralAssetBounds* d_neural_bounds_array = nullptr;
    int* d_instance_to_neural_map = nullptr;

    // Upload instance-to-neural mapping
    if (!instance_to_neural_vec.empty()) {
        CUDA_CHECK(cudaMalloc(&d_instance_to_neural_map, instance_to_neural_vec.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_instance_to_neural_map, instance_to_neural_vec.data(),
                   instance_to_neural_vec.size() * sizeof(int), cudaMemcpyHostToDevice));
        launch_params.instance_to_neural_map = d_instance_to_neural_map;
    } else {
        launch_params.instance_to_neural_map = nullptr;
    }

    std::cout << "launch_params.num_neural_assets: " << launch_params.num_neural_assets;
    std::cout << ", len host_nn_params_values: " << host_nn_params_values.size() << std::endl;
    if (!host_nn_params_values.empty()) {
        size_t n = host_nn_params_values.size();
        CUDA_CHECK(cudaMalloc(&d_neural_array, n * sizeof(NeuralNetworkParams)));
        CUDA_CHECK(cudaMemcpy(d_neural_array, host_nn_params_values.data(), n * sizeof(NeuralNetworkParams), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&d_neural_bounds_array, n * sizeof(NeuralAssetBounds)));
        CUDA_CHECK(cudaMemcpy(d_neural_bounds_array, host_neural_bounds.data(), n * sizeof(NeuralAssetBounds), cudaMemcpyHostToDevice));

        // launch_params.num_neural_assets = static_cast<uint32_t>(n);
        launch_params.neural_networks = d_neural_array;
        launch_params.neural_bounds_array = d_neural_bounds_array;
        // Also keep the first bounds for legacy single-network code paths
        launch_params.neural_network = host_nn_params_values[0];
        launch_params.neural_bounds = host_neural_bounds[0];
    // } else {
    //     // No per-object neural assets collected; fall back to single global params if available
    //     launch_params.num_neural_assets = weights_loaded ? 1u : 0u;
    //     if (weights_loaded) {
    //         // copy single NeuralNetworkParams struct into device memory so the shader can reference it
    //         CUDA_CHECK(cudaMalloc(&d_neural_array, sizeof(NeuralNetworkParams)));
    //         NeuralNetworkParams tmp = neural_params.get_device_params();
    //         CUDA_CHECK(cudaMemcpy(d_neural_array, &tmp, sizeof(NeuralNetworkParams), cudaMemcpyHostToDevice));
    //         CUDA_CHECK(cudaMalloc(&d_neural_bounds_array, sizeof(NeuralAssetBounds)));
    //         NeuralAssetBounds b = {{neural_min.x, neural_min.y, neural_min.z}, {neural_max.x, neural_max.y, neural_max.z}};
    //         CUDA_CHECK(cudaMemcpy(d_neural_bounds_array, &b, sizeof(NeuralAssetBounds), cudaMemcpyHostToDevice));

    //         launch_params.neural_networks = d_neural_array;
    //         launch_params.neural_bounds_array = d_neural_bounds_array;
    //         launch_params.neural_bounds = b;
    //     } else {
    //         launch_params.neural_networks = nullptr;
    //         launch_params.neural_bounds_array = nullptr;
    //         launch_params.neural_bounds = { {neural_min.x, neural_min.y, neural_min.z}, {neural_max.x, neural_max.y, neural_max.z} };
    //     }
    }

    // Set up camera (view from angle to see both floor and neural asset)
    launch_params.camera.position = { cam_pos.x, cam_pos.y, cam_pos.z };
    launch_params.camera.fov = fov;

    // Camera basis vectors computed from look_at
    float aspect = (float)width / (float)height;
    float vfov = launch_params.camera.fov * M_PI / 180.0f;
    float vfov_size = std::tan(vfov / 2.0f);

    // Compute camera coordinate system
    float3 up = make_float3(0.0f, 1.0f, 0.0f);  // World up
    float3 forward = make_float3(cam_look.x - cam_pos.x, cam_look.y - cam_pos.y, cam_look.z - cam_pos.z);

    // Normalize forward (w points opposite to view direction)
    float forward_len = std::sqrt(forward.x * forward.x + forward.y * forward.y + forward.z * forward.z);
    float3 w = make_float3(forward.x / forward_len, forward.y / forward_len, forward.z / forward_len);

    // Right vector (u) = up x w, then normalize
    float3 u_raw = make_float3(up.y * w.z - up.z * w.y, up.z * w.x - up.x * w.z, up.x * w.y - up.y * w.x);
    float u_len = std::sqrt(u_raw.x * u_raw.x + u_raw.y * u_raw.y + u_raw.z * u_raw.z);
    float3 u_norm = make_float3(u_raw.x / u_len, u_raw.y / u_len, u_raw.z / u_len);

    // Up vector (v) = w x u
    float3 v_norm = make_float3(w.y * u_norm.z - w.z * u_norm.y, w.z * u_norm.x - w.x * u_norm.z, w.x * u_norm.y - w.y * u_norm.x);

    // Scale by FOV and aspect ratio
    launch_params.camera.u = {u_norm.x * aspect * vfov_size, u_norm.y * aspect * vfov_size, u_norm.z * aspect * vfov_size};
    launch_params.camera.v = {v_norm.x * vfov_size, v_norm.y * vfov_size, v_norm.z * vfov_size};
    launch_params.camera.w = {w.x, w.y, w.z};

    std::cout << "Camera parameters:" << std::endl;
    std::cout << "Camera u: " << "[ " << launch_params.camera.u.x << ", " << launch_params.camera.u.y << ", " << launch_params.camera.u.z << " ]" << std::endl;
    std::cout << "Camera v: " << "[ " << launch_params.camera.v.x << ", " << launch_params.camera.v.y << ", " << launch_params.camera.v.z << " ]" << std::endl;
    std::cout << "Camera w: " << "[ " << launch_params.camera.w.x << ", " << launch_params.camera.w.y << ", " << launch_params.camera.w.z << " ]" << std::endl;

    // Neural bounds
    launch_params.neural_bounds.min = {neural_min.x, neural_min.y, neural_min.z};
    launch_params.neural_bounds.max = {neural_max.x, neural_max.y, neural_max.z};

    // Setup point light (above the scene)
    launch_params.light.position = readFloat3Aligned(root["scene"]["light"]["position"]);
    launch_params.light.color = readFloat3Aligned(root["scene"]["light"]["color"]);
    launch_params.light.intensity = root["scene"]["light"]["intensity"].as<float>();

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
    std::vector<float3_aligned> h_hit_position_buffer(width * height);
    std::vector<int32_t> h_instance_id_buffer(width * height);

    CUDA_CHECK(cudaMemcpy(
        h_frame_buffer.data(),
        d_frame_buffer,
        width * height * sizeof(uchar4),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_hit_position_buffer.data(),
        d_hit_position_buffer,
        width * height * sizeof(float3_aligned),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_instance_id_buffer.data(),
        d_instance_id_buffer,
        width * height * sizeof(int32_t),
        cudaMemcpyDeviceToHost));

    // Download divergence profiling data
    std::vector<uint32_t> h_divergence_buffer(width * height * NUM_DIVERGENCE_METRICS);
    CUDA_CHECK(cudaMemcpy(
        h_divergence_buffer.data(),
        d_divergence_buffer,
        width * height * NUM_DIVERGENCE_METRICS * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    // Write output images
    write_ppm(output_file, h_frame_buffer.data(), width, height);

    // Write hit positions as binary file for analysis
    std::string hit_pos_file = output_file;
    size_t dot_pos = hit_pos_file.find_last_of('.');
    if (dot_pos != std::string::npos) {
        hit_pos_file = hit_pos_file.substr(0, dot_pos) + "_hit_position.bin";
    } else {
        hit_pos_file += "_hit_position.bin";
    }

    std::ofstream bin_file(hit_pos_file, std::ios::binary);
    if (bin_file.is_open()) {
        bin_file.write(reinterpret_cast<const char*>(h_hit_position_buffer.data()),
                       width * height * sizeof(float3_aligned));
        bin_file.close();
        std::cout << "Wrote hit positions to: " << hit_pos_file << std::endl;
    }

    // Write instance IDs as binary file for analysis
    std::string instance_id_file = output_file;
    if (dot_pos != std::string::npos) {
        instance_id_file = instance_id_file.substr(0, dot_pos) + "_instance_id.bin";
    } else {
        instance_id_file += "_instance_id.bin";
    }

    std::ofstream instance_bin_file(instance_id_file, std::ios::binary);
    if (instance_bin_file.is_open()) {
        instance_bin_file.write(reinterpret_cast<const char*>(h_instance_id_buffer.data()),
                                width * height * sizeof(int32_t));
        instance_bin_file.close();
        std::cout << "Wrote instance IDs to: " << instance_id_file << std::endl;
    }

    // Write divergence profiling metrics to binary file
    std::string div_file = output_file;
    if (dot_pos != std::string::npos) {
        div_file = div_file.substr(0, dot_pos) + "_divergence.bin";
    } else {
        div_file += "_divergence.bin";
    }

    std::ofstream div_bin_file(div_file, std::ios::binary);
    if (div_bin_file.is_open()) {
        // Write header: [width, height, num_metrics]
        uint32_t header[3] = {(uint32_t)width, (uint32_t)height, NUM_DIVERGENCE_METRICS};
        div_bin_file.write(reinterpret_cast<const char*>(header), 3 * sizeof(uint32_t));

        // Write divergence data
        div_bin_file.write(reinterpret_cast<const char*>(h_divergence_buffer.data()),
                           width * height * NUM_DIVERGENCE_METRICS * sizeof(uint32_t));
        div_bin_file.close();
        std::cout << "Wrote divergence metrics to: " << div_file << std::endl;
    }

    // Cleanup
    cudaFree(d_frame_buffer);
    cudaFree(d_hit_position_buffer);
    cudaFree(d_instance_id_buffer);
    cudaFree(d_divergence_buffer);
    cudaFree(d_launch_params);
    if (d_neural_array) cudaFree(d_neural_array);
    if (d_neural_bounds_array) cudaFree(d_neural_bounds_array);
    if (d_instance_to_neural_map) cudaFree(d_instance_to_neural_map);

    std::cout << "\n=== Rendering Complete ===" << std::endl;
    std::cout << "Output saved to: " << output_file << std::endl;
    return 0;
}
