#pragma once

// ============================================================================
// HIPRT Render Kernel Source (for runtime compilation)
// ============================================================================
//
// This file contains the CUDA/HIP kernel source as a string constant.
// HIPRT compiles this at runtime with hiprtBuildTraceKernels().
//
// Note: The kernel includes custom intersection functions for neural assets
// and divergence profiling instrumentation.
//
// ============================================================================

// Kernel source code as inline string
// This gets compiled at runtime by HIPRT
inline const char* getKernelSource() {
    return R"KERNEL_SOURCE(

// ============================================================================
// HIPRT Includes (provided by HIPRT runtime compilation)
// ============================================================================
#include <hiprt/hiprt_device.h>

// ============================================================================
// CUDA/HIP vector types
// ============================================================================

// HIPRT provides float3 via hiprt_vec.h (included by hiprt_device.h)
// NVRTC provides uchar4 via CUDA's vector_types.h automatically

// ============================================================================
// Shared Types and Constants
// ============================================================================

// Geometry type indices (must match values used in geometry_builder)
#define GEOM_TYPE_TRIANGLE 0
#define GEOM_TYPE_NEURAL   1

// Ray type indices
#define RAY_TYPE_PRIMARY 0
#define RAY_TYPE_SHADOW  1

// ============================================================================
// Divergence Profiling Utilities
// ============================================================================

// Measure divergence at a branch point using warp ballot
// Returns the number of threads that diverge (take the minority path)
__device__ __forceinline__ uint32_t measure_divergence(bool condition) {
    // Get active mask for current warp
    unsigned int active_mask = __activemask();

    // Ballot: which threads satisfy the condition?
    unsigned int ballot = __ballot_sync(active_mask, condition);

    // Count threads taking each path
    unsigned int true_count = __popc(ballot);
    unsigned int false_count = __popc(active_mask) - true_count;

    // Divergence = min(true_count, false_count)
    // This measures "wasted work" from divergence
    return min(true_count, false_count);
}

// Calculate entropy of instance IDs within a warp (spatial coherence metric)
__device__ __forceinline__ float warpInstanceEntropy(int instanceID) {
    unsigned mask = __activemask();
    int lane = threadIdx.x & 31;
    int activeCount = __popc(mask);

    if (activeCount <= 1) return 0.0f;

    float entropy = 0.0f;

    for (int leader = 0; leader < 32; ++leader) {
        if ((mask & (1u << leader)) == 0) continue;

        int leaderID = __shfl_sync(mask, instanceID, leader);
        unsigned groupMask = __ballot_sync(mask, instanceID == leaderID);
        int groupLeader = __ffs(groupMask) - 1;

        if (lane == leader && leader == groupLeader) {
            int groupSize = __popc(groupMask);
            float p = float(groupSize) / float(activeCount);
            entropy += -p * log2f(p);
        }
    }

    int firstLane = __ffs(mask) - 1;
    entropy = __shfl_sync(mask, entropy, firstLane);
    return entropy;
}

// ============================================================================
// Traversal Metrics Structure
// ============================================================================

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

// ============================================================================
// Hash Grid Encoding Parameters
// ============================================================================

struct HashGridParams {
    float* hash_table;
    uint32_t* offset_table;
    uint32_t n_levels;
    uint32_t n_features_per_level;
    uint32_t log2_hashmap_size;
    float base_resolution;
    float per_level_scale;
};

// ============================================================================
// MLP Parameters
// ============================================================================

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

// ============================================================================
// Neural Asset Data (passed to custom intersection)
// ============================================================================

struct NeuralAssetData {
    // AABB bounds for each neural primitive
    float3* aabb_min;
    float3* aabb_max;

    // Neural network parameters per asset
    NeuralNetworkParams* neural_params;
    uint32_t num_assets;

    // Mapping from scene instance ID to neural asset index (-1 for non-neural instances)
    int32_t* instance_to_neural_idx;
    uint32_t max_instance_id;  // For bounds checking

    // Metrics buffer for divergence tracking (unused, kept for compatibility)
    TraversalMetrics* metrics;
};

// ============================================================================
// Traversal Payload (passed through HIPRT traversal)
// ============================================================================

struct TraversalPayload {
    TraversalMetrics* metrics;
    NeuralAssetData* neural_data;
};

// ============================================================================
// Activation Functions
// ============================================================================

__device__ __forceinline__ float relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// ============================================================================
// Hash Grid Encoding
// ============================================================================

__device__ __forceinline__ uint32_t hash_grid_index(
    uint32_t x, uint32_t y, uint32_t z, uint32_t hashmap_size
) {
    constexpr uint32_t primes[3] = {1u, 2654435761u, 805459861u};
    uint32_t result = 0;
    result ^= x * primes[0];
    result ^= y * primes[1];
    result ^= z * primes[2];
    return result;
}

__device__ __forceinline__ void hash_encode(
    const float3& position,
    const HashGridParams& params,
    float* output,
    TraversalMetrics* metrics
) {
    for (uint32_t level = 0; level < params.n_levels; ++level) {
        float scale = params.base_resolution * powf(params.per_level_scale, (float)level) - 1.0f;
        uint32_t grid_resolution = (uint32_t)ceilf(scale) + 1;
        uint32_t grid_volume = grid_resolution * grid_resolution * grid_resolution;

        float3 pos_scaled = make_float3(
            position.x * scale + 0.5f,
            position.y * scale + 0.5f,
            position.z * scale + 0.5f
        );

        uint32_t x0 = (uint32_t)floorf(pos_scaled.x);
        uint32_t y0 = (uint32_t)floorf(pos_scaled.y);
        uint32_t z0 = (uint32_t)floorf(pos_scaled.z);

        float fx = pos_scaled.x - (float)x0;
        float fy = pos_scaled.y - (float)y0;
        float fz = pos_scaled.z - (float)z0;

        uint32_t level_offset_grid_points = params.offset_table[level];
        uint32_t hashmap_size = params.offset_table[level + 1] - level_offset_grid_points;
        uint32_t level_offset_features = level_offset_grid_points * params.n_features_per_level;

        for (uint32_t f = 0; f < params.n_features_per_level; ++f) {
            float values[8];

            for (int i = 0; i < 8; ++i) {
                uint32_t dx = (i & 1);
                uint32_t dy = (i & 2) >> 1;
                uint32_t dz = (i & 4) >> 2;

                bool use_direct_index = (grid_volume <= hashmap_size);

                // Track hash vs direct indexing divergence
                if (metrics != nullptr) {
                    metrics->hash_divergence += measure_divergence(use_direct_index);
                }

                uint32_t hash_idx;
                if (use_direct_index) {
                    hash_idx = (x0 + dx) + (y0 + dy) * grid_resolution +
                               (z0 + dz) * grid_resolution * grid_resolution;
                } else {
                    hash_idx = hash_grid_index(x0 + dx, y0 + dy, z0 + dz, hashmap_size);
                }

                hash_idx = hash_idx % hashmap_size;
                uint32_t table_idx = level_offset_features + hash_idx * params.n_features_per_level + f;
                values[i] = params.hash_table[table_idx];
            }

            // Trilinear interpolation
            float c00 = values[0] * (1.0f - fx) + values[1] * fx;
            float c01 = values[2] * (1.0f - fx) + values[3] * fx;
            float c10 = values[4] * (1.0f - fx) + values[5] * fx;
            float c11 = values[6] * (1.0f - fx) + values[7] * fx;
            float c0 = c00 * (1.0f - fy) + c01 * fy;
            float c1 = c10 * (1.0f - fy) + c11 * fy;
            float result = c0 * (1.0f - fz) + c1 * fz;

            output[level * params.n_features_per_level + f] = result;
        }
    }
}

// ============================================================================
// MLP Forward Pass
// ============================================================================

__device__ __forceinline__ void matmul_add_bias(
    const float* input,
    const float* weights,
    const float* bias,
    float* output,
    uint32_t in_dim,
    uint32_t out_dim
) {
    for (uint32_t i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < in_dim; ++j) {
            sum += weights[i * in_dim + j] * input[j];
        }
        output[i] = sum;
    }
}

__device__ __forceinline__ void apply_activation(float* data, uint32_t size, char act_type) {
    if (act_type == 'r' || act_type == 'R') {
        for (uint32_t i = 0; i < size; ++i) data[i] = relu(data[i]);
    } else if (act_type == 's' || act_type == 'S') {
        for (uint32_t i = 0; i < size; ++i) data[i] = sigmoid(data[i]);
    }
}

__device__ __forceinline__ void mlp_forward(
    const float* input,
    const MLPParams& params,
    float* output,
    float* scratch,
    TraversalMetrics* metrics
) {
    const float* current_input = input;
    float* layer_output = scratch;

    for (uint32_t l = 0; l < params.n_layers; ++l) {
        const MLPLayer& layer = params.layers[l];

        matmul_add_bias(current_input, layer.weights, layer.biases,
                        layer_output, layer.in_dim, layer.out_dim);

        bool is_hidden_layer = (l < params.n_layers - 1);

        // Track hidden vs output layer divergence
        if (metrics != nullptr) {
            metrics->mlp_divergence += measure_divergence(is_hidden_layer);
        }

        if (is_hidden_layer) {
            apply_activation(layer_output, layer.out_dim, 'R');
        } else {
            apply_activation(layer_output, layer.out_dim, params.output_activation[0]);
        }

        current_input = layer_output;
        layer_output += layer.out_dim;
    }

    const MLPLayer& last_layer = params.layers[params.n_layers - 1];
    for (uint32_t i = 0; i < last_layer.out_dim; ++i) {
        output[i] = current_input[i];
    }
}

// ============================================================================
// Neural Inference
// ============================================================================

__device__ __forceinline__ void neural_inference(
    const float3& position,
    const float3& direction,
    const NeuralNetworkParams& net_params,
    float& visibility,
    float3& normal,
    float& depth,
    TraversalMetrics* metrics
) {
    float scratch[512];

    // Hash encode position
    float* position_encoding = scratch;
    hash_encode(position, net_params.hash_encoding, position_encoding, metrics);
    uint32_t pos_encoding_dim = net_params.hash_encoding.n_levels *
                                net_params.hash_encoding.n_features_per_level;

    // Encode direction
    float* direction_input = scratch + 64;
    direction_input[0] = direction.x;
    direction_input[1] = direction.y;
    direction_input[2] = direction.z;
    for (uint32_t i = 3; i < 16; ++i) direction_input[i] = 1.0f;

    float* direction_encoding = scratch + 80;
    float* dir_scratch = scratch + 96;
    mlp_forward(direction_input, net_params.direction_encoder, direction_encoding, dir_scratch, metrics);

    // Concatenate encodings
    float* concatenated = scratch + 240;
    for (uint32_t i = 0; i < pos_encoding_dim; ++i) {
        concatenated[i] = position_encoding[i];
    }
    uint32_t dir_encoding_dim = net_params.direction_encoder.layers[
        net_params.direction_encoder.n_layers - 1].out_dim;
    for (uint32_t i = 0; i < dir_encoding_dim; ++i) {
        concatenated[pos_encoding_dim + i] = direction_encoding[i];
    }

    float* decoder_scratch = scratch + 288;

    // Visibility decoder
    float vis_output[16];
    mlp_forward(concatenated, net_params.visibility_decoder, vis_output, decoder_scratch, metrics);
    visibility = vis_output[0];

    // Normal decoder
    float norm_output[16];
    mlp_forward(concatenated, net_params.normal_decoder, norm_output, decoder_scratch, metrics);
    normal = make_float3(norm_output[0], norm_output[1], norm_output[2]);

    // Depth decoder
    float depth_output[16];
    mlp_forward(concatenated, net_params.depth_decoder, depth_output, decoder_scratch, metrics);
    depth = depth_output[0];
}

// ============================================================================
// Ray-AABB Intersection
// ============================================================================

__device__ __forceinline__ bool intersect_aabb(
    const hiprtRay& ray,
    const float3& aabb_min,
    const float3& aabb_max,
    float& t_near,
    float& t_far
) {
    float3 invDir = make_float3(1.0f / ray.direction.x,
                                 1.0f / ray.direction.y,
                                 1.0f / ray.direction.z);

    float3 t0 = make_float3(
        (aabb_min.x - ray.origin.x) * invDir.x,
        (aabb_min.y - ray.origin.y) * invDir.y,
        (aabb_min.z - ray.origin.z) * invDir.z
    );

    float3 t1 = make_float3(
        (aabb_max.x - ray.origin.x) * invDir.x,
        (aabb_max.y - ray.origin.y) * invDir.y,
        (aabb_max.z - ray.origin.z) * invDir.z
    );

    float3 tmin = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
    float3 tmax = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

    t_near = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.z, ray.minT));
    t_far = fminf(fminf(tmax.x, tmax.y), fminf(tmax.z, ray.maxT));

    return t_near <= t_far;
}

// ============================================================================
// Custom Intersection Function for Neural Assets
// ============================================================================
//
// This function is called by HIPRT when a ray hits an AABB primitive
// with geomType = GEOM_TYPE_NEURAL.
//
// Function signature must match HIPRT requirements:
//   __device__ bool funcName(const hiprtRay& ray, const void* data,
//                            void* payload, hiprtHit& hit)
//
// Note: hit.primID is pre-set by HIPRT before calling this function
//
// ============================================================================

__device__ bool intersectNeuralAABB(
    const hiprtRay& ray,
    const void* data,
    void* payload,
    hiprtHit& hit
) {
    // Data contains flat AABB bounds: [min_x, min_y, min_z, max_x, max_y, max_z] per primitive
    if (data == nullptr) {
        return false;
    }

    // Data contains flat AABB bounds indexed by neural asset order (not primID)
    // We'll get the correct index after looking up neural_idx from instanceID
    // For now, use primID=0 bounds as a fallback for the AABB test
    // The actual neural_idx bounds will be used after we determine which neural asset this is
    const float* aabb_data = reinterpret_cast<const float*>(data);

    // Initially use primID for AABB bounds (will be corrected after instance lookup)
    // Each neural geometry has only 1 primitive, so primID is always 0 within that geometry
    // But aabb_data is indexed by neural asset order, so we'll re-read after getting neural_idx
    uint32_t initial_idx = hit.primID * 6;
    float min_x = aabb_data[initial_idx + 0];
    float min_y = aabb_data[initial_idx + 1];
    float min_z = aabb_data[initial_idx + 2];
    float max_x = aabb_data[initial_idx + 3];
    float max_y = aabb_data[initial_idx + 4];
    float max_z = aabb_data[initial_idx + 5];

    // Ray-AABB intersection using slab method
    float inv_dx = 1.0f / ray.direction.x;
    float inv_dy = 1.0f / ray.direction.y;
    float inv_dz = 1.0f / ray.direction.z;

    float t1 = (min_x - ray.origin.x) * inv_dx;
    float t2 = (max_x - ray.origin.x) * inv_dx;
    float t3 = (min_y - ray.origin.y) * inv_dy;
    float t4 = (max_y - ray.origin.y) * inv_dy;
    float t5 = (min_z - ray.origin.z) * inv_dz;
    float t6 = (max_z - ray.origin.z) * inv_dz;

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    // Check for valid AABB intersection
    if (tmax < 0.0f || tmin > tmax || tmin < ray.minT || tmin > ray.maxT) {
        return false;
    }

    // Check if this potential hit is closer than existing hit
    if (hit.t > 0.0f && tmin >= hit.t) {
        return false;
    }

    // Get payload for neural network parameters and metrics
    TraversalPayload* trav_payload = reinterpret_cast<TraversalPayload*>(payload);
    NeuralAssetData* neural_data = (trav_payload != nullptr) ? trav_payload->neural_data : nullptr;
    TraversalMetrics* metrics = (trav_payload != nullptr) ? trav_payload->metrics : nullptr;

    // Track neural intersection test
    if (metrics != nullptr) {
        metrics->neural_tests++;
    }

    // If no neural data available, fall back to simple AABB intersection
    if (neural_data == nullptr || neural_data->neural_params == nullptr) {
        hit.t = tmin;
        // Compute hit point for normal calculation
        float px = ray.origin.x + tmin * ray.direction.x;
        float py = ray.origin.y + tmin * ray.direction.y;
        float pz = ray.origin.z + tmin * ray.direction.z;

        // Determine which face was hit (for normal)
        float eps = 1e-4f;
        if (fabsf(px - min_x) < eps) {
            hit.normal = hiprtFloat3{-1.0f, 0.0f, 0.0f};
        } else if (fabsf(px - max_x) < eps) {
            hit.normal = hiprtFloat3{1.0f, 0.0f, 0.0f};
        } else if (fabsf(py - min_y) < eps) {
            hit.normal = hiprtFloat3{0.0f, -1.0f, 0.0f};
        } else if (fabsf(py - max_y) < eps) {
            hit.normal = hiprtFloat3{0.0f, 1.0f, 0.0f};
        } else if (fabsf(pz - min_z) < eps) {
            hit.normal = hiprtFloat3{0.0f, 0.0f, -1.0f};
        } else {
            hit.normal = hiprtFloat3{0.0f, 0.0f, 1.0f};
        }
        hit.uv = hiprtFloat2{0.0f, 0.0f};
        return true;
    }

    // Get neural network parameters using instance ID to map to neural asset index
    // Each neural asset is a separate geometry instance, so we use instanceID (not primID)
    // The instance_to_neural_idx array maps scene instance IDs to neural asset indices
    if (hit.instanceID >= neural_data->max_instance_id) {
        return false;
    }

    int32_t neural_idx = neural_data->instance_to_neural_idx[hit.instanceID];
    if (neural_idx < 0 || (uint32_t)neural_idx >= neural_data->num_assets) {
        // This instance is not a neural asset, fall back to AABB hit
        hit.t = tmin;
        hit.normal = hiprtFloat3{0.0f, 1.0f, 0.0f};
        hit.uv = hiprtFloat2{0.0f, 0.0f};
        return true;
    }

    const NeuralNetworkParams& net_params = neural_data->neural_params[neural_idx];

    // Check if network parameters are valid
    if (net_params.hash_encoding.hash_table == nullptr) {
        // No valid network, fall back to AABB surface hit
        hit.t = tmin;
        hit.normal = hiprtFloat3{0.0f, 1.0f, 0.0f};
        hit.uv = hiprtFloat2{0.0f, 0.0f};
        return true;
    }

    // AABB dimensions for normalizing positions to [0,1]^3
    float3 aabb_min = make_float3(min_x, min_y, min_z);
    float3 aabb_size = make_float3(max_x - min_x, max_y - min_y, max_z - min_z);

    // Compute the AABB entry point (where ray enters the bounding box)
    float3 entry_pos = make_float3(
        ray.origin.x + tmin * ray.direction.x,
        ray.origin.y + tmin * ray.direction.y,
        ray.origin.z + tmin * ray.direction.z
    );

    // Normalize entry position to [0,1]^3 within the AABB
    float3 normalized_pos = make_float3(
        (entry_pos.x - aabb_min.x) / aabb_size.x,
        (entry_pos.y - aabb_min.y) / aabb_size.y,
        (entry_pos.z - aabb_min.z) / aabb_size.z
    );

    // Clamp to valid range
    normalized_pos.x = fmaxf(0.0f, fminf(1.0f, normalized_pos.x));
    normalized_pos.y = fmaxf(0.0f, fminf(1.0f, normalized_pos.y));
    normalized_pos.z = fmaxf(0.0f, fminf(1.0f, normalized_pos.z));

    float ray_len = sqrtf(
        ray.direction.x * ray.direction.x +
        ray.direction.y * ray.direction.y +
        ray.direction.z * ray.direction.z
    );
    // Ray direction (already normalized)
    float3 ray_dir = make_float3(
        ray.direction.x / ray_len,
        ray.direction.y / ray_len,
        ray.direction.z / ray_len
    );

    // Query neural network at AABB entry point
    float visibility;
    float3 normal;
    float depth;

    neural_inference(
        normalized_pos,
        ray_dir,
        net_params,
        visibility,
        normal,
        depth,
        metrics
    );

    // Track divergence in visibility decisions
    const float VIS_THRESHOLD = 0.5f;
    bool is_hit = (visibility >= VIS_THRESHOLD);
    if (metrics != nullptr) {
        metrics->neural_divergence += measure_divergence(is_hit);
    }

    // Check if neural network predicts a hit
    if (!is_hit) {
        if (metrics != nullptr) {
            metrics->early_reject_divergence += measure_divergence(true);
        }
        return false;
    }

    // Compute hit distance using depth output
    // Depth is the distance from AABB entry to the actual surface hit
    // Scale depth by AABB diagonal to convert from normalized to world space
    float aabb_diagonal = sqrtf(aabb_size.x * aabb_size.x +
                                aabb_size.y * aabb_size.y +
                                aabb_size.z * aabb_size.z);
    float hit_t = tmin + depth*1.319 / ray_len;

    // Validate hit is within AABB bounds and ray range
    if (hit_t < ray.minT || hit_t > tmax || hit_t > ray.maxT) {
        // Depth outside valid range, use AABB entry point
        hit_t = tmin;
    }

    // Check if this hit is closer than existing hit
    if (hit.t > 0.0f && hit_t >= hit.t) {
        return false;
    }

    // Record the hit
    hit.t = hit_t;

    // Normalize the neural network's normal output
    float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normal_len > 0.001f) {
        hit.normal = hiprtFloat3{
            normal.x / normal_len,
            normal.y / normal_len,
            normal.z / normal_len
        };
    } else {
        // Fallback normal if neural output is degenerate
        hit.normal = hiprtFloat3{0.0f, 1.0f, 0.0f};
    }

    // Store visibility in UV for potential use in shading
    hit.uv = hiprtFloat2{visibility, depth};

    return true;
}

// ============================================================================
// Pastel Color Map for Instance Coloring
// ============================================================================

__device__ __forceinline__ float3 getPastelColor(uint32_t instanceID) {
    // 16 distinct pastel colors
    const float3 pastel_colors[16] = {
        make_float3(0.98f, 0.68f, 0.68f),  // Pastel red
        make_float3(0.68f, 0.85f, 0.98f),  // Pastel blue
        make_float3(0.68f, 0.98f, 0.76f),  // Pastel green
        make_float3(0.98f, 0.92f, 0.68f),  // Pastel yellow
        make_float3(0.88f, 0.68f, 0.98f),  // Pastel purple
        make_float3(0.98f, 0.82f, 0.68f),  // Pastel orange
        make_float3(0.68f, 0.98f, 0.98f),  // Pastel cyan
        make_float3(0.98f, 0.68f, 0.88f),  // Pastel pink
        make_float3(0.78f, 0.98f, 0.68f),  // Pastel lime
        make_float3(0.98f, 0.78f, 0.82f),  // Pastel coral
        make_float3(0.68f, 0.78f, 0.98f),  // Pastel periwinkle
        make_float3(0.88f, 0.98f, 0.68f),  // Pastel chartreuse
        make_float3(0.98f, 0.68f, 0.78f),  // Pastel rose
        make_float3(0.68f, 0.98f, 0.88f),  // Pastel mint
        make_float3(0.92f, 0.68f, 0.98f),  // Pastel lavender
        make_float3(0.98f, 0.88f, 0.78f),  // Pastel peach
    };
    return pastel_colors[instanceID % 16];
}

// ============================================================================
// Camera Parameters
// ============================================================================

struct CameraParams {
    float3 position;
    float3 direction;
    float3 up;
    float3 right;
    float fov;
    float aspect;
};

// ============================================================================
// Light Parameters
// ============================================================================

struct LightParams {
    float3 position;
    float3 color;
    float intensity;
};

// ============================================================================
// Mesh Normal Data for Smooth Shading
// ============================================================================

struct MeshData {
    float* vertex_normals;      // [num_vertices * 3] float array (x,y,z,x,y,z,...)
    uint32_t* triangle_indices; // [num_triangles * 3] vertex indices
    uint32_t num_vertices;
    uint32_t num_triangles;
};

// ============================================================================
// Render Kernel
// ============================================================================

extern "C" __global__ void renderKernel(
    hiprtScene scene,
    hiprtFuncTable funcTable,
    NeuralAssetData* neural_data,
    MeshData* mesh_data,
    uint32_t num_mesh_instances,
    CameraParams camera,
    LightParams light,
    uchar4* frame_buffer,
    TraversalMetrics* metrics_buffer,
    uint32_t width,
    uint32_t height
) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const uint32_t pixel_idx = y * width + x;

    // Initialize per-pixel metrics
    TraversalMetrics metrics = {};

    // Create traversal payload with metrics and neural data
    TraversalPayload payload;
    payload.metrics = &metrics;
    payload.neural_data = neural_data;

    // Generate primary ray
    float u = (2.0f * ((float)x + 0.5f) / (float)width - 1.0f) * camera.aspect * tanf(camera.fov * 0.5f);
    float v = (1.0f - 2.0f * ((float)y + 0.5f) / (float)height) * tanf(camera.fov * 0.5f);

    hiprtRay ray;
    ray.origin = hiprtFloat3{camera.position.x, camera.position.y, camera.position.z};
    ray.direction = hiprtFloat3{
        camera.direction.x + u * camera.right.x + v * camera.up.x,
        camera.direction.y + u * camera.right.y + v * camera.up.y,
        camera.direction.z + u * camera.right.z + v * camera.up.z
    };
    ray.minT = 0.001f;
    ray.maxT = 1e16f;

    // Normalize direction
    float len = sqrtf(ray.direction.x * ray.direction.x +
                      ray.direction.y * ray.direction.y +
                      ray.direction.z * ray.direction.z);
    ray.direction.x /= len;
    ray.direction.y /= len;
    ray.direction.z /= len;

    // DEBUG: Print ray for center pixel to verify rays are correct
    if (x == width/2 && y == height/2) {
        printf("DEBUG: Center pixel (%d,%d) ray origin=(%f,%f,%f) dir=(%f,%f,%f)\n",
               x, y, ray.origin.x, ray.origin.y, ray.origin.z,
               ray.direction.x, ray.direction.y, ray.direction.z);
        printf("DEBUG: scene=%p funcTable=%p neural_data=%p\n",
               (void*)scene, (void*)funcTable, (void*)neural_data);
    }

    // Create traversal object with payload for neural intersection
    hiprtSceneTraversalClosest traversal(
        scene,
        ray,
        hiprtFullRayMask,
        hiprtTraversalHintDefault,
        &payload,  // Pass payload so custom intersection can access neural data
        funcTable,
        RAY_TYPE_PRIMARY,
        0.0f
    );

    // Get closest hit
    hiprtHit hit = traversal.getNextHit();

    // DEBUG: Print hit result for center pixel
    if (x == width/2 && y == height/2) {
        printf("DEBUG: Center pixel hit=%d instanceID=%d primID=%d t=%f\n",
               hit.hasHit() ? 1 : 0, hit.instanceID, hit.primID, hit.t);
    }

    // Track hit/miss divergence
    metrics.node_divergence += measure_divergence(hit.hasHit());

    // Compute instance entropy for spatial coherence
    metrics.instance_entropy = warpInstanceEntropy(hit.hasHit() ? hit.instanceID : -1);

    // Compute color
    float3 color;
    if (hit.hasHit()) {
        // Hit position
        float3 hit_pos = make_float3(
            ray.origin.x + hit.t * ray.direction.x,
            ray.origin.y + hit.t * ray.direction.y,
            ray.origin.z + hit.t * ray.direction.z
        );

        // Normal calculation with barycentric interpolation for smooth shading
        float3 normal;

        // Check if this is a mesh instance with vertex normal data
        bool use_smooth_normal = (mesh_data != nullptr &&
                                   hit.instanceID < num_mesh_instances &&
                                   mesh_data[hit.instanceID].vertex_normals != nullptr);

        if (use_smooth_normal) {
            // Get mesh data for this instance
            const MeshData& mesh = mesh_data[hit.instanceID];

            // Get triangle vertex indices
            uint32_t prim_idx = hit.primID * 3;
            uint32_t i0 = mesh.triangle_indices[prim_idx + 0];
            uint32_t i1 = mesh.triangle_indices[prim_idx + 1];
            uint32_t i2 = mesh.triangle_indices[prim_idx + 2];

            // Get vertex normals
            float3 n0 = make_float3(
                mesh.vertex_normals[i0 * 3 + 0],
                mesh.vertex_normals[i0 * 3 + 1],
                mesh.vertex_normals[i0 * 3 + 2]
            );
            float3 n1 = make_float3(
                mesh.vertex_normals[i1 * 3 + 0],
                mesh.vertex_normals[i1 * 3 + 1],
                mesh.vertex_normals[i1 * 3 + 2]
            );
            float3 n2 = make_float3(
                mesh.vertex_normals[i2 * 3 + 0],
                mesh.vertex_normals[i2 * 3 + 1],
                mesh.vertex_normals[i2 * 3 + 2]
            );

            // Barycentric coordinates from hit.uv
            // hit.uv.x = u (weight for v1), hit.uv.y = v (weight for v2)
            // weight for v0 = 1 - u - v
            float u = hit.uv.x;
            float v = hit.uv.y;
            float w = 1.0f - u - v;

            // Interpolate normal: N = w*n0 + u*n1 + v*n2
            normal = make_float3(
                w * n0.x + u * n1.x + v * n2.x,
                w * n0.y + u * n1.y + v * n2.y,
                w * n0.z + u * n1.z + v * n2.z
            );
        } else {
            // Fall back to geometric normal from hit data
            normal = make_float3(hit.normal.x, hit.normal.y, hit.normal.z);
        }

        // Normalize the normal
        float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        if (normal_len > 0.0f) {
            normal.x /= normal_len;
            normal.y /= normal_len;
            normal.z /= normal_len;
        }

        // Light direction
        float3 light_dir = make_float3(
            light.position.x - hit_pos.x,
            light.position.y - hit_pos.y,
            light.position.z - hit_pos.z
        );
        float light_dist = sqrtf(light_dir.x * light_dir.x +
                                  light_dir.y * light_dir.y +
                                  light_dir.z * light_dir.z);
        light_dir.x /= light_dist;
        light_dir.y /= light_dist;
        light_dir.z /= light_dist;

        // Shadow ray
        hiprtRay shadow_ray;
        shadow_ray.origin = hiprtFloat3{
            hit_pos.x + 0.001f * normal.x,
            hit_pos.y + 0.001f * normal.y,
            hit_pos.z + 0.001f * normal.z
        };
        shadow_ray.direction = hiprtFloat3{light_dir.x, light_dir.y, light_dir.z};
        shadow_ray.minT = 0.001f;
        shadow_ray.maxT = light_dist - 0.001f;

        metrics.shadow_tests++;

        hiprtSceneTraversalAnyHit shadow_traversal(
            scene,
            shadow_ray,
            hiprtFullRayMask,
            hiprtTraversalHintShadowRays,
            &payload,
            funcTable,
            RAY_TYPE_SHADOW,
            0.0f
        );

        hiprtHit shadow_hit = shadow_traversal.getNextHit();
        bool in_shadow = shadow_hit.hasHit();

        metrics.shadow_divergence += measure_divergence(in_shadow);

        // Check if normal is valid, use fallback if not
        if (normal_len < 0.001f) {
            normal = make_float3(0.0f, 1.0f, 0.0f);
        }

        // Get pastel diffuse color based on instance ID
        float3 diffuse_color = getPastelColor(hit.instanceID);

        // Simple diffuse shading
        float shadow_factor = in_shadow ? 0.3f : 1.0f;
        float ndotl = fmaxf(0.0f, normal.x * light_dir.x +
                                  normal.y * light_dir.y +
                                  normal.z * light_dir.z);

        // Reasonable attenuation - divide intensity by 100 to normalize
        float attenuation = (light.intensity) / (light_dist * light_dist + 1.0f) / 10.0f;
        float diffuse = ndotl * shadow_factor * attenuation;

        // Combine diffuse color with light color
        color = make_float3(
            fminf(diffuse * diffuse_color.x * light.color.x, 1.0f),
            fminf(diffuse * diffuse_color.y * light.color.y, 1.0f),
            fminf(diffuse * diffuse_color.z * light.color.z, 1.0f)
        );
    } else {
        // Background color
        color = make_float3(0.1f, 0.1f, 0.15f);
    }

    // Clamp and convert to 8-bit
    uchar4 pixel;
    pixel.x = (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f);
    pixel.y = (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f);
    pixel.z = (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f);
    pixel.w = 255;

    // Write outputs
    frame_buffer[pixel_idx] = pixel;
    metrics_buffer[pixel_idx] = metrics;
}

)KERNEL_SOURCE";
}

// Function name sets for HIPRT kernel compilation
// These map geometry types to custom intersection functions
inline const char* getIntersectFuncName() {
    return "intersectNeuralAABB";
}

inline const char* getRenderKernelName() {
    return "renderKernel";
}
