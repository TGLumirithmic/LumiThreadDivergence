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

// Divergence metric count (must match kernel_source.h)
#define NUM_DIVERGENCE_METRICS 11

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

    if (argc > 1) scene_file = argv[1];
    if (argc > 2) output_file = argv[2];
    if (argc > 3) width = std::atoi(argv[3]);
    if (argc > 4) height = std::atoi(argv[4]);

    if (scene_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " <scene.yaml> [output.ppm] [width] [height]" << std::endl;
        return 1;
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Scene file: " << scene_file << std::endl;
    std::cout << "  Output: " << output_file << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;

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

            // Build AABB geometry (geomType=1 for neural/custom)
            auto geom = geom_builder.build_aabb_geometry(aabbs, 1);
            if (geom.valid()) {
                std::array<float, 12> transform;
                for (int i = 0; i < 12; ++i) transform[i] = obj.transform[i];

                scene_builder.add_instance(geom.get(), transform, instance_id++);
                std::cout << "  Added neural asset AABB: ["
                          << bmin.x << "," << bmin.y << "," << bmin.z << "] to ["
                          << bmax.x << "," << bmax.y << "," << bmax.z << "]" << std::endl;

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

    // Allocate metrics buffer
    void* d_metrics_buffer = nullptr;
    // TraversalMetrics has 11 uint32_t fields + 1 float (12 * 4 = 48 bytes)
    size_t metrics_size = width * height * 48;
    ORO_CHECK(oroMalloc(&d_metrics_buffer, metrics_size));
    ORO_CHECK(oroMemset((oroDeviceptr)d_metrics_buffer, 0, metrics_size));

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

    // Launch kernel
    std::cout << "\nLaunching render kernel..." << std::endl;

    hiprtScene scene_handle = scene.get();
    hiprtFuncTable func_table = compiled.get_func_table();
    void* neural_data = nullptr;  // TODO: Set up neural asset data when weights are loaded

    uint32_t w = width;
    uint32_t h = height;

    void* kernel_args[] = {
        &scene_handle,
        &func_table,
        &neural_data,
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

    // Download and save image
    std::vector<unsigned char> h_frame_buffer(frame_buffer_size);
    ORO_CHECK(oroMemcpyDtoH(h_frame_buffer.data(), (oroDeviceptr)d_frame_buffer, frame_buffer_size));

    write_ppm(output_file, h_frame_buffer.data(), width, height);

    // Cleanup
    oroStreamDestroy(stream);
    oroFree((oroDeviceptr)d_frame_buffer);
    oroFree((oroDeviceptr)d_metrics_buffer);

    // Free mesh normal data GPU buffers
    for (const auto& gpu : gpu_mesh_data) {
        if (gpu.d_vertex_normals) oroFree((oroDeviceptr)gpu.d_vertex_normals);
        if (gpu.d_triangle_indices) oroFree((oroDeviceptr)gpu.d_triangle_indices);
    }
    if (d_mesh_data_array) oroFree((oroDeviceptr)d_mesh_data_array);

    std::cout << "\n=== HIPRT Rendering Complete ===" << std::endl;
    context.cleanup();
    return 0;
}
