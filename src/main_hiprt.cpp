// =============================================================================
// HIPRT Neural Renderer with Divergence Profiling
// =============================================================================
// This is the HIPRT-based replacement for the OptiX renderer.
// It uses HIPRT's software-based ray traversal on NVIDIA GPUs via Orochi,
// which provides full visibility into the traversal process for profiling.
// =============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <memory>
#include <cstring>
#include <limits>

#include <yaml-cpp/yaml.h>
#include <tiny_obj_loader.h>

// HIPRT host API
#include "hiprt/hiprt_context.h"
#include "hiprt/geometry_builder.h"
#include "hiprt/scene_builder.h"
#include "hiprt/kernel_compiler.h"
#include "hiprt/kernel_source.h"

// Neural network utilities
#include "neural/weight_loader.h"
#include "neural/config.h"

// Divergence metric count for output (11 uint32 fields + instance_entropy as fixed-point = 12 total)
#define NUM_DIVERGENCE_METRICS 12

// =============================================================================
// Neural Network Types (must match kernel_source.h exactly)
// =============================================================================

// Forward declarations for kernel-side types (used in NeuralAssetData)
struct HashGridParams {
    float* hash_table;
    uint32_t* offset_table;
    uint32_t n_levels;
    uint32_t n_features_per_level;
    uint32_t log2_hashmap_size;
    float base_resolution;
    float per_level_scale;
};

struct MLPLayer {
    float* weights;
    float* biases;
    uint32_t in_dim;
    uint32_t out_dim;
};

struct MLPParams {
    MLPLayer* layers;
    uint32_t n_layers;
    const char* output_activation;
};

struct NeuralNetworkParams {
    HashGridParams hash_encoding;
    MLPParams direction_encoder;
    MLPParams visibility_decoder;
    MLPParams normal_decoder;
    MLPParams depth_decoder;
    float* scratch_buffer;
    uint32_t scratch_buffer_size;
};

// Forward declare float3 equivalent for NeuralAssetData
struct float3_kernel {
    float x, y, z;
};

// This must match TraversalMetrics in kernel_source.h
struct TraversalMetrics {
    uint32_t traversal_steps;
    uint32_t node_divergence;
    uint32_t triangle_tests;
    uint32_t triangle_divergence;
    uint32_t neural_tests;
    uint32_t neural_divergence;
    uint32_t early_reject_divergence;
    uint32_t hash_divergence;
    uint32_t mlp_divergence;
    uint32_t shadow_tests;
    uint32_t shadow_divergence;
    float instance_entropy;
};

struct NeuralAssetData {
    float3_kernel* aabb_min;
    float3_kernel* aabb_max;
    NeuralNetworkParams* neural_params;
    uint32_t num_assets;
    int32_t* instance_to_neural_idx;
    uint32_t max_instance_id;
    TraversalMetrics* metrics;
};

// =============================================================================
// Orochi-based Neural Weight Loader (adapted from optix::NeuralNetworkParamsHost)
// =============================================================================

static uint32_t next_multiple(uint32_t val, uint32_t divisor) {
    return ((val + divisor - 1) / divisor) * divisor;
}

static uint32_t powi(uint32_t base, uint32_t exp) {
    uint32_t result = 1;
    for (uint32_t i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
}

static float grid_scale(uint32_t level, float log2_per_level_scale, float base_resolution) {
    return std::exp2(level * log2_per_level_scale) * base_resolution - 1.0f;
}

static uint32_t grid_resolution(float scale) {
    return (uint32_t)std::ceil(scale) + 1;
}

// Host-side container for neural network parameters using Orochi API
class NeuralNetworkParamsOrochi {
public:
    NeuralNetworkParamsOrochi(const neural::NetworkConfig& config) : config_(config) {
        std::memset(&d_params_, 0, sizeof(NeuralNetworkParams));
        // Allocate activation strings on device
        d_relu_str_ = allocate_string("ReLU");
        d_sigmoid_str_ = allocate_string("Sigmoid");
        d_none_str_ = allocate_string("None");
    }

    ~NeuralNetworkParamsOrochi() {
        free_device_memory();
    }

    bool load_from_weights(const neural::WeightLoader& loader);
    const NeuralNetworkParams& get_device_params() const { return d_params_; }
    bool is_loaded() const { return loaded_; }

private:
    neural::NetworkConfig config_;
    bool loaded_ = false;
    NeuralNetworkParams d_params_;

    // Device memory tracking
    float* d_hash_table_ = nullptr;
    uint32_t* d_hash_offsets_ = nullptr;
    size_t hash_table_size_ = 0;

    MLPLayer* d_dir_encoder_layers_ = nullptr;
    uint32_t dir_encoder_n_layers_ = 0;
    MLPLayer* d_vis_decoder_layers_ = nullptr;
    uint32_t vis_decoder_n_layers_ = 0;
    MLPLayer* d_norm_decoder_layers_ = nullptr;
    uint32_t norm_decoder_n_layers_ = 0;
    MLPLayer* d_depth_decoder_layers_ = nullptr;
    uint32_t depth_decoder_n_layers_ = 0;

    char* d_relu_str_ = nullptr;
    char* d_sigmoid_str_ = nullptr;
    char* d_none_str_ = nullptr;

    // Store host-side layer info for cleanup
    std::vector<MLPLayer> h_dir_layers_;
    std::vector<MLPLayer> h_vis_layers_;
    std::vector<MLPLayer> h_norm_layers_;
    std::vector<MLPLayer> h_depth_layers_;

    char* allocate_string(const std::string& str) {
        char* d_str = nullptr;
        size_t len = str.length() + 1;
        ORO_CHECK(oroMalloc((void**)&d_str, len));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_str, const_cast<char*>(str.c_str()), len));
        return d_str;
    }

    void free_device_memory() {
        if (d_hash_table_) oroFree((oroDeviceptr)d_hash_table_);
        if (d_hash_offsets_) oroFree((oroDeviceptr)d_hash_offsets_);

        auto free_mlp_layers = [](const std::vector<MLPLayer>& layers) {
            for (const auto& layer : layers) {
                if (layer.weights) oroFree((oroDeviceptr)layer.weights);
                if (layer.biases) oroFree((oroDeviceptr)layer.biases);
            }
        };

        free_mlp_layers(h_dir_layers_);
        free_mlp_layers(h_vis_layers_);
        free_mlp_layers(h_norm_layers_);
        free_mlp_layers(h_depth_layers_);

        if (d_dir_encoder_layers_) oroFree((oroDeviceptr)d_dir_encoder_layers_);
        if (d_vis_decoder_layers_) oroFree((oroDeviceptr)d_vis_decoder_layers_);
        if (d_norm_decoder_layers_) oroFree((oroDeviceptr)d_norm_decoder_layers_);
        if (d_depth_decoder_layers_) oroFree((oroDeviceptr)d_depth_decoder_layers_);

        if (d_relu_str_) oroFree((oroDeviceptr)d_relu_str_);
        if (d_sigmoid_str_) oroFree((oroDeviceptr)d_sigmoid_str_);
        if (d_none_str_) oroFree((oroDeviceptr)d_none_str_);

        loaded_ = false;
    }

    bool load_hash_encoding(const neural::WeightLoader& loader);
    bool load_mlp(
        const neural::WeightLoader& loader,
        const std::string& prefix,
        uint32_t n_layers,
        uint32_t hidden_dim,
        uint32_t input_dim,
        uint32_t output_dim,
        const std::string& output_activation,
        MLPLayer*& d_layers_out,
        uint32_t& n_layers_out,
        std::vector<MLPLayer>& h_layers_storage
    );
};

bool NeuralNetworkParamsOrochi::load_hash_encoding(const neural::WeightLoader& loader) {
    std::cout << "  Loading hash encoding weights..." << std::endl;

    const neural::Tensor* hash_tensor = loader.get_tensor("position_encoder.params");
    if (!hash_tensor) {
        std::cerr << "    Error: Could not find position_encoder.params" << std::endl;
        return false;
    }

    constexpr uint32_t N_POS_DIMS = 3;
    constexpr uint32_t N_FEATURES_PER_LEVEL = 2;

    std::vector<uint32_t> h_offsets(config_.n_levels + 1);
    uint32_t offset = 0;
    float log2_per_level_scale = std::log2(config_.compute_per_level_scale());

    for (uint32_t i = 0; i < config_.n_levels; ++i) {
        float scale = grid_scale(i, log2_per_level_scale, config_.base_resolution);
        uint32_t resolution = grid_resolution(scale);
        uint32_t max_params = std::numeric_limits<uint32_t>::max() / 2;
        float dense_params = std::pow((float)resolution, (float)N_POS_DIMS);
        uint32_t params_in_level = (dense_params > (float)max_params) ? max_params : powi(resolution, N_POS_DIMS);
        params_in_level = next_multiple(params_in_level, 8u);
        uint32_t hashmap_size = 1u << config_.log2_hashmap_size;
        params_in_level = std::min(params_in_level, hashmap_size);
        h_offsets[i] = offset;
        offset += params_in_level;
    }

    h_offsets[config_.n_levels] = offset;
    uint32_t total_params = offset * N_FEATURES_PER_LEVEL;

    std::cout << "    Hash table: " << total_params << " parameters (loaded: " << hash_tensor->data.size() << ")" << std::endl;

    // Allocate and upload hash table
    hash_table_size_ = hash_tensor->data.size();
    ORO_CHECK(oroMalloc((void**)&d_hash_table_, hash_table_size_ * sizeof(float)));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_hash_table_,
                            const_cast<float*>(hash_tensor->data.data()),
                            hash_table_size_ * sizeof(float)));

    // Allocate and upload offset table
    ORO_CHECK(oroMalloc((void**)&d_hash_offsets_, (config_.n_levels + 1) * sizeof(uint32_t)));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_hash_offsets_,
                            h_offsets.data(),
                            (config_.n_levels + 1) * sizeof(uint32_t)));

    // Setup HashGridParams
    d_params_.hash_encoding.hash_table = d_hash_table_;
    d_params_.hash_encoding.offset_table = d_hash_offsets_;
    d_params_.hash_encoding.n_levels = config_.n_levels;
    d_params_.hash_encoding.n_features_per_level = config_.n_features_per_level;
    d_params_.hash_encoding.log2_hashmap_size = config_.log2_hashmap_size;
    d_params_.hash_encoding.base_resolution = config_.base_resolution;
    d_params_.hash_encoding.per_level_scale = config_.compute_per_level_scale();

    std::cout << "    Hash encoding: " << config_.n_levels << " levels, "
              << config_.n_features_per_level << " features/level" << std::endl;
    return true;
}

bool NeuralNetworkParamsOrochi::load_mlp(
    const neural::WeightLoader& loader,
    const std::string& prefix,
    uint32_t n_layers,
    uint32_t hidden_dim,
    uint32_t input_dim,
    uint32_t output_dim,
    const std::string& output_activation,
    MLPLayer*& d_layers_out,
    uint32_t& n_layers_out,
    std::vector<MLPLayer>& h_layers_storage
) {
    std::cout << "  Loading " << prefix << " MLP weights..." << std::endl;

    const neural::Tensor* params_tensor = loader.get_tensor(prefix + ".params");
    if (!params_tensor) {
        std::cerr << "    Error: Could not find " << prefix << ".params" << std::endl;
        return false;
    }

    h_layers_storage.resize(n_layers + 1);
    size_t offset = 0;
    uint32_t padded_output_dim = ((output_dim + 15) / 16) * 16;

    for (uint32_t l = 0; l < n_layers + 1; ++l) {
        uint32_t layer_in_dim = (l == 0) ? input_dim : hidden_dim;
        uint32_t layer_out_dim = (l == n_layers) ? padded_output_dim : hidden_dim;

        h_layers_storage[l].in_dim = layer_in_dim;
        h_layers_storage[l].out_dim = layer_out_dim;
        h_layers_storage[l].biases = nullptr;

        size_t weight_size = layer_out_dim * layer_in_dim;

        if (offset + weight_size > params_tensor->data.size()) {
            std::cerr << "    Error: Parameter size mismatch at layer " << l << std::endl;
            return false;
        }

        // Allocate and copy weights
        ORO_CHECK(oroMalloc((void**)&h_layers_storage[l].weights, weight_size * sizeof(float)));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)h_layers_storage[l].weights,
                                const_cast<float*>(params_tensor->data.data() + offset),
                                weight_size * sizeof(float)));
        offset += weight_size;
    }

    std::cout << "    " << prefix << ": " << n_layers + 1 << " layers, "
              << offset << "/" << params_tensor->data.size() << " params used" << std::endl;

    // Upload layer array to device
    ORO_CHECK(oroMalloc((void**)&d_layers_out, (n_layers + 1) * sizeof(MLPLayer)));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_layers_out,
                            h_layers_storage.data(),
                            (n_layers + 1) * sizeof(MLPLayer)));

    n_layers_out = n_layers + 1;
    return true;
}

bool NeuralNetworkParamsOrochi::load_from_weights(const neural::WeightLoader& loader) {
    if (!loader.is_loaded()) {
        std::cerr << "WeightLoader has no weights loaded" << std::endl;
        return false;
    }

    std::cout << "Converting weights for HIPRT kernel..." << std::endl;

    if (!load_hash_encoding(loader)) return false;

    uint32_t pos_encoding_dim = config_.encoding_n_output_dims();
    uint32_t dir_encoding_dim = config_.direction_encoder_n_output_dims();
    uint32_t decoder_input_dim = pos_encoding_dim + dir_encoding_dim;

    // Direction encoder
    if (!load_mlp(loader, "direction_encoder",
                  config_.direction_n_hidden_layers, config_.direction_hidden_dim,
                  16, config_.direction_hidden_dim, "None",
                  d_dir_encoder_layers_, dir_encoder_n_layers_, h_dir_layers_)) return false;

    // Visibility decoder
    if (!load_mlp(loader, "visibility_decoder",
                  config_.visibility_decoder.n_decoder_layers, config_.n_neurons,
                  decoder_input_dim, config_.visibility_decoder.n_output_dims,
                  config_.visibility_decoder.output_activation,
                  d_vis_decoder_layers_, vis_decoder_n_layers_, h_vis_layers_)) return false;

    // Normal decoder
    if (!load_mlp(loader, "normal_decoder",
                  config_.normal_decoder.n_decoder_layers, config_.n_neurons,
                  decoder_input_dim, config_.normal_decoder.n_output_dims,
                  config_.normal_decoder.output_activation,
                  d_norm_decoder_layers_, norm_decoder_n_layers_, h_norm_layers_)) return false;

    // Depth decoder
    if (!load_mlp(loader, "depth_decoder",
                  config_.depth_decoder.n_decoder_layers, config_.n_neurons,
                  decoder_input_dim, config_.depth_decoder.n_output_dims,
                  config_.depth_decoder.output_activation,
                  d_depth_decoder_layers_, depth_decoder_n_layers_, h_depth_layers_)) return false;

    // Setup MLPParams structures
    d_params_.direction_encoder.layers = d_dir_encoder_layers_;
    d_params_.direction_encoder.n_layers = dir_encoder_n_layers_;
    d_params_.direction_encoder.output_activation = d_none_str_;

    d_params_.visibility_decoder.layers = d_vis_decoder_layers_;
    d_params_.visibility_decoder.n_layers = vis_decoder_n_layers_;
    if (config_.visibility_decoder.output_activation[0] == 'S' ||
        config_.visibility_decoder.output_activation[0] == 's')
        d_params_.visibility_decoder.output_activation = d_sigmoid_str_;
    else if (config_.visibility_decoder.output_activation[0] == 'R' ||
             config_.visibility_decoder.output_activation[0] == 'r')
        d_params_.visibility_decoder.output_activation = d_relu_str_;
    else
        d_params_.visibility_decoder.output_activation = d_none_str_;

    d_params_.normal_decoder.layers = d_norm_decoder_layers_;
    d_params_.normal_decoder.n_layers = norm_decoder_n_layers_;
    d_params_.normal_decoder.output_activation = d_none_str_;

    d_params_.depth_decoder.layers = d_depth_decoder_layers_;
    d_params_.depth_decoder.n_layers = depth_decoder_n_layers_;
    d_params_.depth_decoder.output_activation = d_none_str_;

    loaded_ = true;
    std::cout << "Neural weights loaded successfully" << std::endl;
    return true;
}

// =============================================================================
// Data Structures
// =============================================================================

// Simple 3D vector
struct float3_simple {
    float x, y, z;
};

// Mesh data for smooth shading (vertex normals + triangle indices)
struct MeshNormalData {
    std::vector<float3_simple> vertex_normals;  // Per-vertex normals
    std::vector<uint32_t> triangle_indices;      // 3 indices per triangle (v0, v1, v2)
    uint32_t instance_id;
};

// GPU-side mesh normal data (pointers to device memory)
struct GPUMeshData {
    float* d_vertex_normals;    // [num_vertices * 3] float array
    uint32_t* d_triangle_indices; // [num_triangles * 3] indices
    uint32_t num_vertices;
    uint32_t num_triangles;
};

// Camera parameters
struct Camera {
    float3_simple position;
    float3_simple direction;
    float3_simple up;
    float3_simple right;
    float fov;
    float aspect;
};

// Point light
struct PointLight {
    float3_simple position;
    float3_simple color;
    float intensity;
};

// Scene object from YAML
struct SceneObject {
    enum Type { MESH, NEURAL } type;
    std::string file;       // for meshes (OBJ path)
    std::string weights;    // for neural assets
    float3_simple bounds_min;
    float3_simple bounds_max;
    bool has_bounds = false;
    float transform[12] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
};

// =============================================================================
// YAML Parsing Utilities
// =============================================================================

float3_simple readFloat3(const YAML::Node& node) {
    return {
        node[0].as<float>(),
        node[1].as<float>(),
        node[2].as<float>()
    };
}

void buildTransform(const YAML::Node& tnode, float out[12]) {
    // Start as identity
    float T[12] = {
        1,0,0,0,
        0,1,0,0,
        0,0,1,0
    };

    // Scale
    if (tnode["scale"]) {
        float3_simple s = readFloat3(tnode["scale"]);
        T[0] *= s.x;  T[1] *= s.x;  T[2]  *= s.x;
        T[4] *= s.y;  T[5] *= s.y;  T[6]  *= s.y;
        T[8] *= s.z;  T[9] *= s.z;  T[10] *= s.z;
    }

    // Position (translation)
    if (tnode["position"]) {
        float3_simple p = readFloat3(tnode["position"]);
        T[3]  = p.x;
        T[7]  = p.y;
        T[11] = p.z;
    }

    for (int i = 0; i < 12; i++) out[i] = T[i];
}

std::vector<SceneObject> load_scene_objects(const YAML::Node& root) {
    std::vector<SceneObject> objects;

    auto obj_list = root["scene"]["objects"];
    if (!obj_list) return objects;

    for (const auto& node : obj_list) {
        SceneObject obj;

        std::string t = node["type"].as<std::string>();
        if (t == "mesh")
            obj.type = SceneObject::MESH;
        else if (t == "neural_asset")
            obj.type = SceneObject::NEURAL;
        else
            continue;

        if (node["file"])
            obj.file = node["file"].as<std::string>();

        if (node["weights"])
            obj.weights = node["weights"].as<std::string>();

        if (node["bounds"]) {
            auto b = node["bounds"];
            obj.bounds_min = readFloat3(b["min"]);
            obj.bounds_max = readFloat3(b["max"]);
            obj.has_bounds = true;
        }

        if (node["transform"])
            buildTransform(node["transform"], obj.transform);

        objects.push_back(obj);
    }

    return objects;
}

// =============================================================================
// PPM Image Writer
// =============================================================================

void write_ppm(const std::string& filename, const unsigned char* pixels, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }

    file << "P6\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        file.put(pixels[i * 4 + 0]);  // R
        file.put(pixels[i * 4 + 1]);  // G
        file.put(pixels[i * 4 + 2]);  // B
    }

    std::cout << "Wrote image to: " << filename << std::endl;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    std::cout << "=== HIPRT Neural Renderer ===" << std::endl;

    // Parse command line arguments
    std::string scene_file;
    std::string output_file = "output/hiprt_render.ppm";
    int width = 512;
    int height = 512;
    bool enable_divergence = true;

    // Parse positional and flag arguments
    int positional_idx = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-divergence" || arg == "-D") {
            enable_divergence = false;
        } else {
            // Positional arguments: scene, output, width, height
            switch (positional_idx) {
                case 0: scene_file = arg; break;
                case 1: output_file = arg; break;
                case 2: width = std::atoi(arg.c_str()); break;
                case 3: height = std::atoi(arg.c_str()); break;
            }
            positional_idx++;
        }
    }

    if (scene_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " <scene.yaml> [output.ppm] [width] [height] [--no-divergence|-D]" << std::endl;
        std::cerr << "  --no-divergence, -D  Disable divergence measurement output" << std::endl;
        return 1;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Scene file: " << scene_file << std::endl;
    std::cout << "  Output: " << output_file << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    std::cout << "  Divergence measurement: " << (enable_divergence ? "enabled" : "disabled") << std::endl;

    // Load YAML scene
    YAML::Node root;
    try {
        root = YAML::LoadFile(scene_file);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load YAML: " << e.what() << std::endl;
        return 1;
    }

    // Parse camera
    const auto camera_node = root["scene"]["camera"];
    float3_simple cam_pos = readFloat3(camera_node["position"]);
    float3_simple cam_look = readFloat3(camera_node["look_at"]);
    float fov = camera_node["fov"].as<float>();

    std::cout << "\nCamera:" << std::endl;
    std::cout << "  Position: " << cam_pos.x << ", " << cam_pos.y << ", " << cam_pos.z << std::endl;
    std::cout << "  Look At: " << cam_look.x << ", " << cam_look.y << ", " << cam_look.z << std::endl;
    std::cout << "  FOV: " << fov << std::endl;

    // Parse light
    float3_simple light_pos = readFloat3(root["scene"]["light"]["position"]);
    float3_simple light_color = readFloat3(root["scene"]["light"]["color"]);
    float light_intensity = root["scene"]["light"]["intensity"].as<float>();

    // Initialize HIPRT context
    std::cout << "\nInitializing HIPRT..." << std::endl;
    hiprt::HIPRTContext context;
    if (!context.initialize(0)) {
        std::cerr << "Failed to initialize HIPRT context" << std::endl;
        return 1;
    }

    // Parse scene objects
    std::vector<SceneObject> scene_objects = load_scene_objects(root);
    std::cout << "\nParsed " << scene_objects.size() << " objects from scene file" << std::endl;

    // Build geometry and scene
    hiprt::GeometryBuilder geom_builder(context);
    hiprt::SceneBuilder scene_builder(context);

    // Keep geometry handles alive until scene is built
    std::vector<hiprt::GeometryHandle> geometry_handles;

    // Store mesh normal data for smooth shading
    std::vector<MeshNormalData> mesh_normal_data;

    // Store neural asset data
    struct NeuralAssetInfo {
        float3_simple bounds_min;
        float3_simple bounds_max;
        std::string weights_path;
        uint32_t instance_id;
    };
    std::vector<NeuralAssetInfo> neural_asset_info;
    std::vector<std::unique_ptr<NeuralNetworkParamsOrochi>> neural_hosts;

    // Default network config (matches training parameters)
    neural::NetworkConfig network_config;

    uint32_t instance_id = 0;
    bool has_neural_assets = false;
    bool has_triangle_meshes = false;

    for (const auto& obj : scene_objects) {
        if (obj.type == SceneObject::MESH) {
            has_triangle_meshes = true;

            // Load OBJ mesh
            if (obj.file.size() > 4 && obj.file.substr(obj.file.size()-4) == ".obj") {
                std::cout << "Loading mesh: " << obj.file << std::endl;

                tinyobj::ObjReaderConfig config;
                config.vertex_color = false;
                tinyobj::ObjReader reader;

                if (!reader.ParseFromFile(obj.file, config)) {
                    std::cerr << "Failed to load OBJ: " << obj.file << std::endl;
                    continue;
                }

                const auto& attrib = reader.GetAttrib();
                const auto& shapes = reader.GetShapes();

                // Build vertices
                std::vector<hiprt::Vertex> vertices;
                for (size_t v = 0; v < attrib.vertices.size() / 3; ++v) {
                    hiprt::Vertex vert;
                    vert.x = attrib.vertices[3*v+0];
                    vert.y = attrib.vertices[3*v+1];
                    vert.z = attrib.vertices[3*v+2];
                    vertices.push_back(vert);
                }

                // Build vertex normals - use OBJ normals if present (same indexing as vertices)
                std::vector<float3_simple> vertex_normals;
                for (size_t v = 0; v < vertices.size(); ++v) {
                    float3_simple normal;
                    if (attrib.normals.size() >= 3*(v+1)) {
                        normal.x = attrib.normals[3*v+0];
                        normal.y = attrib.normals[3*v+1];
                        normal.z = attrib.normals[3*v+2];
                    } else {
                        normal = {0.0f, 1.0f, 0.0f};  // fallback
                    }
                    vertex_normals.push_back(normal);
                }

                // Build triangles and collect normal data
                std::vector<hiprt::Triangle> triangles;
                MeshNormalData normal_data;
                normal_data.instance_id = instance_id;

                for (const auto& shape : shapes) {
                    size_t index_offset = 0;
                    for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); ++f) {
                        int fv = shape.mesh.num_face_vertices[f];
                        if (fv == 3) {
                            hiprt::Triangle tri;
                            tri.v0 = shape.mesh.indices[index_offset+0].vertex_index;
                            tri.v1 = shape.mesh.indices[index_offset+1].vertex_index;
                            tri.v2 = shape.mesh.indices[index_offset+2].vertex_index;
                            triangles.push_back(tri);

                            // Store triangle indices for GPU lookup
                            normal_data.triangle_indices.push_back(tri.v0);
                            normal_data.triangle_indices.push_back(tri.v1);
                            normal_data.triangle_indices.push_back(tri.v2);
                        }
                        index_offset += fv;
                    }
                }

                normal_data.vertex_normals = std::move(vertex_normals);

                // Build triangle geometry (uses default geomType=0 for triangles)
                auto geom = geom_builder.build_triangle_geometry(vertices, triangles);
                if (geom.valid()) {
                    // Convert transform to HIPRT format
                    std::array<float, 12> transform;
                    for (int i = 0; i < 12; ++i) transform[i] = obj.transform[i];

                    scene_builder.add_instance(geom.get(), transform, instance_id);
                    std::cout << "  Added mesh with " << vertices.size() << " vertices, "
                              << triangles.size() << " triangles (instance " << instance_id << ")" << std::endl;

                    mesh_normal_data.push_back(std::move(normal_data));
                    instance_id++;

                    // Keep geometry alive until scene is built
                    geometry_handles.push_back(std::move(geom));
                }
            }
        } else if (obj.type == SceneObject::NEURAL) {
            has_neural_assets = true;

            // Build AABB geometry for neural asset
            float3_simple bmin = obj.has_bounds ? obj.bounds_min : float3_simple{-1,-1,-1};
            float3_simple bmax = obj.has_bounds ? obj.bounds_max : float3_simple{1,1,1};

            std::vector<hiprt::AABB> aabbs = {
                {bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z}
            };

            // Build AABB geometry with geomType=1 for neural assets (GEOM_TYPE_NEURAL)
            // This matches the custom intersection function registration in kernel_compiler
            // geomType=0 is reserved for triangles, geomType=1 for custom/neural primitives
            auto geom = geom_builder.build_aabb_geometry(aabbs, 1);
            if (geom.valid()) {
                std::array<float, 12> transform;
                for (int i = 0; i < 12; ++i) transform[i] = obj.transform[i];

                scene_builder.add_instance(geom.get(), transform, instance_id);
                std::cout << "  Added neural asset AABB: ["
                          << bmin.x << "," << bmin.y << "," << bmin.z << "] to ["
                          << bmax.x << "," << bmax.y << "," << bmax.z << "]" << std::endl;
                std::cout << "  Transform matrix:" << std::endl;
                std::cout << "    [" << transform[0] << ", " << transform[1] << ", " << transform[2] << ", " << transform[3] << "]" << std::endl;
                std::cout << "    [" << transform[4] << ", " << transform[5] << ", " << transform[6] << ", " << transform[7] << "]" << std::endl;
                std::cout << "    [" << transform[8] << ", " << transform[9] << ", " << transform[10] << ", " << transform[11] << "]" << std::endl;

                // Store neural asset info for later weight loading
                NeuralAssetInfo info;
                info.bounds_min = bmin;
                info.bounds_max = bmax;
                info.weights_path = obj.weights;
                info.instance_id = instance_id;
                neural_asset_info.push_back(info);

                // Load weights if specified
                if (!obj.weights.empty()) {
                    std::cout << "  Loading neural weights: " << obj.weights << std::endl;
                    auto host = std::make_unique<NeuralNetworkParamsOrochi>(network_config);
                    neural::WeightLoader loader;
                    if (loader.load_from_file(obj.weights)) {
                        if (host->load_from_weights(loader)) {
                            neural_hosts.push_back(std::move(host));
                        } else {
                            std::cerr << "  Warning: Failed to convert weights for neural asset" << std::endl;
                            neural_hosts.push_back(nullptr);
                        }
                    } else {
                        std::cerr << "  Warning: Failed to load weights from " << obj.weights << std::endl;
                        neural_hosts.push_back(nullptr);
                    }
                } else {
                    std::cerr << "  Warning: Neural asset has no weights specified" << std::endl;
                    neural_hosts.push_back(nullptr);
                }

                instance_id++;

                // Export and verify AABB geometry bounds
                hiprtFloat3 exportedMin, exportedMax;
                hiprtError exportErr = hiprtExportGeometryAabb(
                    context.get_context(), geom.get(), exportedMin, exportedMax);
                if (exportErr == hiprtSuccess) {
                    std::cout << "  Exported AABB bounds: min=(" << exportedMin.x << "," << exportedMin.y << "," << exportedMin.z
                              << ") max=(" << exportedMax.x << "," << exportedMax.y << "," << exportedMax.z << ")" << std::endl;
                } else {
                    std::cerr << "  Warning: Failed to export AABB bounds (error " << exportErr << ")" << std::endl;
                }

                // Keep geometry alive until scene is built
                geometry_handles.push_back(std::move(geom));
            }
        }
    }

    // Build scene (TLAS)
    std::cout << "\nBuilding scene TLAS..." << std::endl;
    auto scene = scene_builder.build();
    if (!scene.valid()) {
        std::cerr << "Failed to build scene" << std::endl;
        return 1;
    }

    // Export and verify scene AABB
    hiprtFloat3 sceneMin, sceneMax;
    hiprtError sceneExportErr = hiprtExportSceneAabb(
        context.get_context(), scene.get(), sceneMin, sceneMax);
    if (sceneExportErr == hiprtSuccess) {
        std::cout << "Scene AABB: min=(" << sceneMin.x << "," << sceneMin.y << "," << sceneMin.z
                  << ") max=(" << sceneMax.x << "," << sceneMax.y << "," << sceneMax.z << ")" << std::endl;
    } else {
        std::cerr << "Warning: Failed to export scene AABB (error " << sceneExportErr << ")" << std::endl;
    }

    // Upload mesh normal data to GPU for smooth shading
    std::cout << "\nUploading mesh normal data for smooth shading..." << std::endl;
    std::vector<GPUMeshData> gpu_mesh_data(mesh_normal_data.size());

    for (size_t i = 0; i < mesh_normal_data.size(); ++i) {
        const auto& mesh = mesh_normal_data[i];
        GPUMeshData& gpu = gpu_mesh_data[i];

        gpu.num_vertices = mesh.vertex_normals.size();
        gpu.num_triangles = mesh.triangle_indices.size() / 3;

        // Upload vertex normals (as float array: x,y,z,x,y,z,...)
        size_t normals_size = gpu.num_vertices * 3 * sizeof(float);
        ORO_CHECK(oroMalloc((void**)&gpu.d_vertex_normals, normals_size));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)gpu.d_vertex_normals,
                                const_cast<float3_simple*>(mesh.vertex_normals.data()), normals_size));

        // Upload triangle indices
        size_t indices_size = mesh.triangle_indices.size() * sizeof(uint32_t);
        ORO_CHECK(oroMalloc((void**)&gpu.d_triangle_indices, indices_size));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)gpu.d_triangle_indices,
                                const_cast<uint32_t*>(mesh.triangle_indices.data()), indices_size));

        std::cout << "  Instance " << mesh.instance_id << ": "
                  << gpu.num_vertices << " vertex normals, "
                  << gpu.num_triangles << " triangles" << std::endl;
    }

    // Create GPU-side array of mesh data pointers (indexed by instance ID)
    // For simplicity, we assume mesh instances are contiguous starting at 0
    void* d_mesh_data_array = nullptr;
    if (!gpu_mesh_data.empty()) {
        size_t mesh_data_size = gpu_mesh_data.size() * sizeof(GPUMeshData);
        ORO_CHECK(oroMalloc(&d_mesh_data_array, mesh_data_size));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_mesh_data_array,
                                gpu_mesh_data.data(), mesh_data_size));
    }
    uint32_t num_mesh_instances = gpu_mesh_data.size();

    // Compile render kernel
    std::cout << "\nCompiling render kernel..." << std::endl;
    hiprt::KernelCompiler compiler(context);

    // Determine which intersection functions to use
    const char* intersect_func = has_neural_assets ? getIntersectFuncName() : nullptr;
    int num_geom_types = has_neural_assets ? 2 : 1;

    auto compiled = compiler.compile(
        getKernelSource(),
        getRenderKernelName(),
        intersect_func,
        nullptr,  // No filter function
        num_geom_types,
        2  // 2 ray types (primary + shadow)
    );

    if (!compiled.valid()) {
        std::cerr << "Failed to compile render kernel" << std::endl;
        return 1;
    }

    // Allocate frame buffer
    void* d_frame_buffer = nullptr;
    size_t frame_buffer_size = width * height * 4;  // RGBA
    ORO_CHECK(oroMalloc(&d_frame_buffer, frame_buffer_size));

    // Allocate metrics buffer (only if divergence measurement is enabled)
    void* d_metrics_buffer = nullptr;
    if (enable_divergence) {
        // TraversalMetrics has 11 uint32_t fields + 1 float (12 * 4 = 48 bytes)
        size_t metrics_size = width * height * 48;
        ORO_CHECK(oroMalloc(&d_metrics_buffer, metrics_size));
        ORO_CHECK(oroMemset((oroDeviceptr)d_metrics_buffer, 0, metrics_size));
    }

    // Set up camera
    Camera camera;
    camera.position = cam_pos;
    camera.fov = fov * M_PI / 180.0f;
    camera.aspect = (float)width / (float)height;

    // Compute camera basis vectors
    float3_simple forward = {
        cam_look.x - cam_pos.x,
        cam_look.y - cam_pos.y,
        cam_look.z - cam_pos.z
    };
    float forward_len = std::sqrt(forward.x*forward.x + forward.y*forward.y + forward.z*forward.z);
    camera.direction = {forward.x/forward_len, forward.y/forward_len, forward.z/forward_len};

    float3_simple world_up = {0, 1, 0};
    // right = cross(direction, up)
    camera.right = {
        camera.direction.y * world_up.z - camera.direction.z * world_up.y,
        camera.direction.z * world_up.x - camera.direction.x * world_up.z,
        camera.direction.x * world_up.y - camera.direction.y * world_up.x
    };
    float right_len = std::sqrt(camera.right.x*camera.right.x + camera.right.y*camera.right.y + camera.right.z*camera.right.z);
    camera.right = {camera.right.x/right_len, camera.right.y/right_len, camera.right.z/right_len};

    // up = cross(right, direction)
    camera.up = {
        camera.right.y * camera.direction.z - camera.right.z * camera.direction.y,
        camera.right.z * camera.direction.x - camera.right.x * camera.direction.z,
        camera.right.x * camera.direction.y - camera.right.y * camera.direction.x
    };

    std::cout << "\nCamera vectors:" << std::endl;
    std::cout << "  Position:  (" << camera.position.x << ", " << camera.position.y << ", " << camera.position.z << ")" << std::endl;
    std::cout << "  Direction: (" << camera.direction.x << ", " << camera.direction.y << ", " << camera.direction.z << ")" << std::endl;
    std::cout << "  Right:     (" << camera.right.x << ", " << camera.right.y << ", " << camera.right.z << ")" << std::endl;
    std::cout << "  Up:        (" << camera.up.x << ", " << camera.up.y << ", " << camera.up.z << ")" << std::endl;
    std::cout << "  FOV:       " << (camera.fov * 180.0f / M_PI) << " degrees" << std::endl;
    std::cout << "  Aspect:    " << camera.aspect << std::endl;

    // Set up light
    PointLight light;
    light.position = light_pos;
    light.color = light_color;
    light.intensity = light_intensity;

    // Set up NeuralAssetData if we have neural assets
    void* d_neural_asset_data = nullptr;
    void* d_neural_aabb_min = nullptr;
    void* d_neural_aabb_max = nullptr;
    void* d_neural_params_array = nullptr;
    void* d_instance_to_neural_idx = nullptr;

    if (has_neural_assets && !neural_asset_info.empty()) {
        std::cout << "\nSetting up neural asset data for " << neural_asset_info.size() << " neural assets..." << std::endl;

        // Allocate AABB arrays
        size_t num_neural = neural_asset_info.size();
        ORO_CHECK(oroMalloc(&d_neural_aabb_min, num_neural * sizeof(float3_kernel)));
        ORO_CHECK(oroMalloc(&d_neural_aabb_max, num_neural * sizeof(float3_kernel)));

        // Build host-side AABB arrays
        std::vector<float3_kernel> h_aabb_min(num_neural);
        std::vector<float3_kernel> h_aabb_max(num_neural);
        for (size_t i = 0; i < num_neural; ++i) {
            h_aabb_min[i] = {neural_asset_info[i].bounds_min.x,
                            neural_asset_info[i].bounds_min.y,
                            neural_asset_info[i].bounds_min.z};
            h_aabb_max[i] = {neural_asset_info[i].bounds_max.x,
                            neural_asset_info[i].bounds_max.y,
                            neural_asset_info[i].bounds_max.z};
        }

        // Upload AABBs
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_neural_aabb_min,
                                h_aabb_min.data(), num_neural * sizeof(float3_kernel)));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_neural_aabb_max,
                                h_aabb_max.data(), num_neural * sizeof(float3_kernel)));

        // Build neural params array (device pointers to per-asset NeuralNetworkParams)
        std::vector<NeuralNetworkParams> h_neural_params(num_neural);
        for (size_t i = 0; i < num_neural; ++i) {
            if (neural_hosts[i] && neural_hosts[i]->is_loaded()) {
                h_neural_params[i] = neural_hosts[i]->get_device_params();
                std::cout << "  Asset " << i << ": weights loaded" << std::endl;
            } else {
                std::memset(&h_neural_params[i], 0, sizeof(NeuralNetworkParams));
                std::cout << "  Asset " << i << ": no weights (using zeros)" << std::endl;
            }
        }

        // Upload neural params array
        ORO_CHECK(oroMalloc(&d_neural_params_array, num_neural * sizeof(NeuralNetworkParams)));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_neural_params_array,
                                h_neural_params.data(), num_neural * sizeof(NeuralNetworkParams)));

        // Build instance ID to neural asset index mapping
        // instance_id is the total index across all scene objects (meshes + neural)
        // We need to map from instance_id to neural asset index (0, 1, 2, ...)
        std::vector<int32_t> h_instance_to_neural(instance_id, -1);  // -1 = not a neural asset
        for (size_t i = 0; i < num_neural; ++i) {
            uint32_t inst_id = neural_asset_info[i].instance_id;
            if (inst_id < h_instance_to_neural.size()) {
                h_instance_to_neural[inst_id] = (int32_t)i;
                std::cout << "  Instance " << inst_id << " -> Neural asset " << i << std::endl;
            }
        }

        // Upload instance mapping
        ORO_CHECK(oroMalloc(&d_instance_to_neural_idx, h_instance_to_neural.size() * sizeof(int32_t)));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_instance_to_neural_idx,
                                h_instance_to_neural.data(), h_instance_to_neural.size() * sizeof(int32_t)));

        // Build and upload NeuralAssetData structure
        NeuralAssetData h_neural_data;
        h_neural_data.aabb_min = (float3_kernel*)d_neural_aabb_min;
        h_neural_data.aabb_max = (float3_kernel*)d_neural_aabb_max;
        h_neural_data.neural_params = (NeuralNetworkParams*)d_neural_params_array;
        h_neural_data.num_assets = (uint32_t)num_neural;
        h_neural_data.instance_to_neural_idx = (int32_t*)d_instance_to_neural_idx;
        h_neural_data.max_instance_id = (uint32_t)h_instance_to_neural.size();
        h_neural_data.metrics = (TraversalMetrics*)d_metrics_buffer;  // Share metrics buffer

        ORO_CHECK(oroMalloc(&d_neural_asset_data, sizeof(NeuralAssetData)));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_neural_asset_data,
                                &h_neural_data, sizeof(NeuralAssetData)));

        std::cout << "Neural asset data uploaded successfully" << std::endl;
    }

    // Launch kernel
    std::cout << "\nLaunching render kernel..." << std::endl;

    hiprtScene scene_handle = scene.get();
    hiprtFuncTable func_table = compiled.get_func_table();

    std::cout << "funcTable: " << (func_table ? "valid" : "NULL") << " (" << (void*)func_table << ")" << std::endl;

    // Set up function table data for custom intersection function
    // This is CRITICAL - without this, the custom intersection function receives nullptr for data
    void* d_flat_aabb_data = nullptr;
    if (has_neural_assets && func_table && !neural_asset_info.empty()) {
        std::cout << "\nSetting up function table data for custom intersection..." << std::endl;

        // Create flat AABB data array: [min_x, min_y, min_z, max_x, max_y, max_z] per primitive
        // This format matches what intersectNeuralAABB expects
        std::vector<float> flat_aabb_data;
        for (const auto& info : neural_asset_info) {
            flat_aabb_data.push_back(info.bounds_min.x);
            flat_aabb_data.push_back(info.bounds_min.y);
            flat_aabb_data.push_back(info.bounds_min.z);
            flat_aabb_data.push_back(info.bounds_max.x);
            flat_aabb_data.push_back(info.bounds_max.y);
            flat_aabb_data.push_back(info.bounds_max.z);

            std::cout << "  AABB[" << (&info - &neural_asset_info[0]) << "]: "
                      << "min=(" << info.bounds_min.x << "," << info.bounds_min.y << "," << info.bounds_min.z << ") "
                      << "max=(" << info.bounds_max.x << "," << info.bounds_max.y << "," << info.bounds_max.z << ")" << std::endl;
        }

        // Upload flat AABB data to GPU
        size_t flat_aabb_size = flat_aabb_data.size() * sizeof(float);
        ORO_CHECK(oroMalloc(&d_flat_aabb_data, flat_aabb_size));
        ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_flat_aabb_data, flat_aabb_data.data(), flat_aabb_size));

        std::cout << "  Uploaded " << flat_aabb_data.size() << " floats (" << flat_aabb_size << " bytes) to GPU" << std::endl;
        std::cout << "  d_flat_aabb_data = " << d_flat_aabb_data << std::endl;

        // Set function table data for geomType=1 (neural/custom), all ray types
        hiprtFuncDataSet data_set;
        data_set.intersectFuncData = d_flat_aabb_data;
        data_set.filterFuncData = nullptr;

        // Set for rayType=0 (primary rays)
        hiprtError err0 = hiprtSetFuncTable(context.get_context(), func_table, 1, 0, data_set);
        std::cout << "  hiprtSetFuncTable(geomType=1, rayType=0, data=" << d_flat_aabb_data << ") -> "
                  << (err0 == hiprtSuccess ? "SUCCESS" : "FAILED") << " (err=" << err0 << ")" << std::endl;

        // Set for rayType=1 (shadow rays)
        hiprtError err1 = hiprtSetFuncTable(context.get_context(), func_table, 1, 1, data_set);
        std::cout << "  hiprtSetFuncTable(geomType=1, rayType=1, data=" << d_flat_aabb_data << ") -> "
                  << (err1 == hiprtSuccess ? "SUCCESS" : "FAILED") << " (err=" << err1 << ")" << std::endl;
    }

    uint32_t w = width;
    uint32_t h = height;

    void* kernel_args[] = {
        &scene_handle,
        &func_table,
        &d_neural_asset_data,
        &d_mesh_data_array,
        &num_mesh_instances,
        &camera,
        &light,
        &d_frame_buffer,
        &d_metrics_buffer,
        &w,
        &h
    };

    unsigned int block_x = 8, block_y = 8, block_z = 1;
    unsigned int grid_x = (width + block_x - 1) / block_x;
    unsigned int grid_y = (height + block_y - 1) / block_y;
    unsigned int grid_z = 1;

    oroFunction kernel_func = reinterpret_cast<oroFunction>(compiled.get_function());

    oroStream stream;
    ORO_CHECK(oroStreamCreate(&stream));

    oroError launch_err = oroModuleLaunchKernel(
        kernel_func,
        grid_x, grid_y, grid_z,
        block_x, block_y, block_z,
        0,
        stream,
        kernel_args,
        nullptr
    );

    if (launch_err != oroSuccess) {
        std::cerr << "Kernel launch failed: " << launch_err << std::endl;
        oroStreamDestroy(stream);
        oroFree((oroDeviceptr)d_frame_buffer);
        oroFree((oroDeviceptr)d_metrics_buffer);
        return 1;
    }

    ORO_CHECK(oroStreamSynchronize(stream));

    std::cout << "Rendering complete!" << std::endl;

    // Download and output divergence metrics (if enabled)
    if (enable_divergence && d_metrics_buffer) {
        std::vector<TraversalMetrics> h_metrics(width * height);
        ORO_CHECK(oroMemcpyDtoH(h_metrics.data(), (oroDeviceptr)d_metrics_buffer,
                               width * height * sizeof(TraversalMetrics)));

        // Write divergence metrics to binary file
        std::string div_file = output_file;
        size_t dot_pos = div_file.rfind('.');
        if (dot_pos != std::string::npos) {
            div_file = div_file.substr(0, dot_pos) + "_divergence.bin";
        } else {
            div_file += "_divergence.bin";
        }

        std::ofstream div_bin_file(div_file, std::ios::binary);
        if (div_bin_file.is_open()) {
            // Write header: [width, height, num_metrics]
            // NUM_DIVERGENCE_METRICS = 12 (11 uint32 fields + instance_entropy as fixed-point ×1000)
            uint32_t header[3] = {(uint32_t)width, (uint32_t)height, NUM_DIVERGENCE_METRICS};
            div_bin_file.write(reinterpret_cast<const char*>(header), 3 * sizeof(uint32_t));

            // Write metrics as flat uint32 array (compatible with analyze_divergence.py)
            // Convert instance_entropy float to fixed-point (×1000) as expected by the script
            std::vector<uint32_t> flat_metrics(width * height * NUM_DIVERGENCE_METRICS);
            for (int i = 0; i < width * height; ++i) {
                const auto& m = h_metrics[i];
                uint32_t* out = &flat_metrics[i * NUM_DIVERGENCE_METRICS];
                out[0]  = m.traversal_steps;
                out[1]  = m.node_divergence;
                out[2]  = m.triangle_tests;
                out[3]  = m.triangle_divergence;
                out[4]  = m.neural_tests;
                out[5]  = m.neural_divergence;
                out[6]  = m.early_reject_divergence;
                out[7]  = m.hash_divergence;
                out[8]  = m.mlp_divergence;
                out[9]  = m.shadow_tests;
                out[10] = m.shadow_divergence;
                out[11] = (uint32_t)(m.instance_entropy * 1000.0f);  // Fixed-point ×1000
            }
            div_bin_file.write(reinterpret_cast<const char*>(flat_metrics.data()),
                               flat_metrics.size() * sizeof(uint32_t));
            div_bin_file.close();
            std::cout << "Wrote divergence metrics to: " << div_file << std::endl;
        }

        // Print divergence summary statistics
        uint64_t total_traversal_steps = 0;
        uint64_t total_node_divergence = 0;
        uint64_t total_triangle_tests = 0;
        uint64_t total_triangle_divergence = 0;
        uint64_t total_neural_tests = 0;
        uint64_t total_neural_divergence = 0;
        uint64_t total_early_reject_divergence = 0;
        uint64_t total_hash_divergence = 0;
        uint64_t total_mlp_divergence = 0;
        uint64_t total_shadow_tests = 0;
        uint64_t total_shadow_divergence = 0;
        double total_instance_entropy = 0.0;

        for (const auto& m : h_metrics) {
            total_traversal_steps += m.traversal_steps;
            total_node_divergence += m.node_divergence;
            total_triangle_tests += m.triangle_tests;
            total_triangle_divergence += m.triangle_divergence;
            total_neural_tests += m.neural_tests;
            total_neural_divergence += m.neural_divergence;
            total_early_reject_divergence += m.early_reject_divergence;
            total_hash_divergence += m.hash_divergence;
            total_mlp_divergence += m.mlp_divergence;
            total_shadow_tests += m.shadow_tests;
            total_shadow_divergence += m.shadow_divergence;
            total_instance_entropy += m.instance_entropy;
        }

        std::cout << "\n=== Divergence Metrics Summary ===" << std::endl;
        std::cout << "  Traversal steps:        " << total_traversal_steps << std::endl;
        std::cout << "  Node divergence:        " << total_node_divergence << std::endl;
        std::cout << "  Triangle tests:         " << total_triangle_tests << std::endl;
        std::cout << "  Triangle divergence:    " << total_triangle_divergence << std::endl;
        std::cout << "  Neural tests:           " << total_neural_tests << std::endl;
        std::cout << "  Neural divergence:      " << total_neural_divergence << std::endl;
        std::cout << "  Early reject div:       " << total_early_reject_divergence << std::endl;
        std::cout << "  Hash divergence:        " << total_hash_divergence << std::endl;
        std::cout << "  MLP divergence:         " << total_mlp_divergence << std::endl;
        std::cout << "  Shadow tests:           " << total_shadow_tests << std::endl;
        std::cout << "  Shadow divergence:      " << total_shadow_divergence << std::endl;
        std::cout << "  Avg instance entropy:   " << (total_instance_entropy / (width * height)) << std::endl;
    }

    // Download and save image
    std::vector<unsigned char> h_frame_buffer(frame_buffer_size);
    ORO_CHECK(oroMemcpyDtoH(h_frame_buffer.data(), (oroDeviceptr)d_frame_buffer, frame_buffer_size));

    write_ppm(output_file, h_frame_buffer.data(), width, height);

    // Cleanup
    oroStreamDestroy(stream);
    oroFree((oroDeviceptr)d_frame_buffer);
    if (d_metrics_buffer) oroFree((oroDeviceptr)d_metrics_buffer);

    // Free mesh normal data GPU buffers
    for (const auto& gpu : gpu_mesh_data) {
        if (gpu.d_vertex_normals) oroFree((oroDeviceptr)gpu.d_vertex_normals);
        if (gpu.d_triangle_indices) oroFree((oroDeviceptr)gpu.d_triangle_indices);
    }
    if (d_mesh_data_array) oroFree((oroDeviceptr)d_mesh_data_array);

    // Free neural asset GPU buffers
    if (d_neural_asset_data) oroFree((oroDeviceptr)d_neural_asset_data);
    if (d_neural_aabb_min) oroFree((oroDeviceptr)d_neural_aabb_min);
    if (d_neural_aabb_max) oroFree((oroDeviceptr)d_neural_aabb_max);
    if (d_neural_params_array) oroFree((oroDeviceptr)d_neural_params_array);
    if (d_instance_to_neural_idx) oroFree((oroDeviceptr)d_instance_to_neural_idx);
    if (d_flat_aabb_data) oroFree((oroDeviceptr)d_flat_aabb_data);
    // neural_hosts will be freed automatically by unique_ptr destructors

    std::cout << "\n=== HIPRT Rendering Complete ===" << std::endl;
    context.cleanup();
    return 0;
}
