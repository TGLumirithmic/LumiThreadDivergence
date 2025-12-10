#pragma once

#include "hiprt_context.h"
#include "geometry_builder.h"
#include <vector>
#include <array>

namespace hiprt {

/**
 * Instance data for a geometry instance in the scene
 */
struct Instance {
    hiprtGeometry geometry;         // The geometry to instance
    std::array<float, 12> transform; // 3x4 row-major transform matrix
    uint32_t instance_id;           // User-defined instance ID
    uint32_t mask;                  // Visibility mask (default: all bits set)

    Instance() : geometry(nullptr), transform{1,0,0,0, 0,1,0,0, 0,0,1,0},
                 instance_id(0), mask(0xFFFFFFFF) {}
};

/**
 * Handle wrapper for HIPRT scene with automatic cleanup
 */
class SceneHandle {
public:
    SceneHandle() = default;
    SceneHandle(hiprtScene scene, const HIPRTContext* ctx)
        : scene_(scene), context_(ctx) {}

    ~SceneHandle() {
        if (scene_ && context_) {
            hiprtDestroyScene(context_->get_context(), scene_);
        }
    }

    // No copy
    SceneHandle(const SceneHandle&) = delete;
    SceneHandle& operator=(const SceneHandle&) = delete;

    // Move OK
    SceneHandle(SceneHandle&& other) noexcept
        : scene_(other.scene_), context_(other.context_) {
        other.scene_ = nullptr;
        other.context_ = nullptr;
    }

    SceneHandle& operator=(SceneHandle&& other) noexcept {
        if (this != &other) {
            if (scene_ && context_) {
                hiprtDestroyScene(context_->get_context(), scene_);
            }
            scene_ = other.scene_;
            context_ = other.context_;
            other.scene_ = nullptr;
            other.context_ = nullptr;
        }
        return *this;
    }

    hiprtScene get() const { return scene_; }
    bool valid() const { return scene_ != nullptr; }

private:
    hiprtScene scene_ = nullptr;
    const HIPRTContext* context_ = nullptr;
};

/**
 * Scene Builder
 *
 * Builds HIPRT scenes (TLAS) from geometry instances.
 * Supports multiple instances with different transforms.
 */
class SceneBuilder {
public:
    explicit SceneBuilder(const HIPRTContext& context);
    ~SceneBuilder();

    // No copy
    SceneBuilder(const SceneBuilder&) = delete;
    SceneBuilder& operator=(const SceneBuilder&) = delete;

    /**
     * Add an instance to the scene
     *
     * @param geometry The geometry to instance
     * @param transform 3x4 row-major transformation matrix (default: identity)
     * @param instance_id User-defined instance ID for hit identification
     * @param mask Visibility mask for ray filtering
     */
    void add_instance(
        hiprtGeometry geometry,
        const std::array<float, 12>& transform = {1,0,0,0, 0,1,0,0, 0,0,1,0},
        uint32_t instance_id = 0,
        uint32_t mask = 0xFFFFFFFF
    );

    /**
     * Add an instance with identity transform
     */
    void add_instance(hiprtGeometry geometry, uint32_t instance_id);

    /**
     * Clear all instances
     */
    void clear();

    /**
     * Get number of instances
     */
    size_t instance_count() const { return instances_.size(); }

    /**
     * Build the scene (TLAS)
     *
     * @param quality Build quality for the scene BVH
     * @return Handle to the built scene
     */
    SceneHandle build(BuildQuality quality = BuildQuality::FAST);

private:
    const HIPRTContext& context_;
    std::vector<Instance> instances_;

    // Temporary buffers
    void* temp_buffer_ = nullptr;
    size_t temp_buffer_size_ = 0;

    void ensure_temp_buffer(size_t required_size);
    void free_temp_buffer();
};

/**
 * Create an identity transform
 */
inline std::array<float, 12> identity_transform() {
    return {1,0,0,0, 0,1,0,0, 0,0,1,0};
}

/**
 * Create a translation transform
 */
inline std::array<float, 12> translation_transform(float x, float y, float z) {
    return {1,0,0,x, 0,1,0,y, 0,0,1,z};
}

/**
 * Create a scale transform
 */
inline std::array<float, 12> scale_transform(float sx, float sy, float sz) {
    return {sx,0,0,0, 0,sy,0,0, 0,0,sz,0};
}

} // namespace hiprt
