#pragma once

#include "hiprt_context.h"
#include <vector>
#include <array>

namespace hiprt {

/**
 * Build quality settings for BVH construction
 */
enum class BuildQuality {
    FAST,           // LBVH-style, hiprtBuildFlagBitPreferFastBuild
    BALANCED,       // Default HIPRT settings
    HIGH_QUALITY    // SAH-style, hiprtBuildFlagBitPreferHighQualityBuild
};

/**
 * Simple vertex structure (position only for now)
 */
struct Vertex {
    float x, y, z;
};

/**
 * Triangle indices
 */
struct Triangle {
    uint32_t v0, v1, v2;
};

/**
 * Axis-aligned bounding box
 */
struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
};

/**
 * Handle wrapper for HIPRT geometry with automatic cleanup
 */
class GeometryHandle {
public:
    GeometryHandle() = default;
    GeometryHandle(hiprtGeometry geom, const HIPRTContext* ctx)
        : geometry_(geom), context_(ctx) {}

    ~GeometryHandle() {
        if (geometry_ && context_) {
            hiprtDestroyGeometry(context_->get_context(), geometry_);
        }
    }

    // No copy
    GeometryHandle(const GeometryHandle&) = delete;
    GeometryHandle& operator=(const GeometryHandle&) = delete;

    // Move OK
    GeometryHandle(GeometryHandle&& other) noexcept
        : geometry_(other.geometry_), context_(other.context_) {
        other.geometry_ = nullptr;
        other.context_ = nullptr;
    }

    GeometryHandle& operator=(GeometryHandle&& other) noexcept {
        if (this != &other) {
            if (geometry_ && context_) {
                hiprtDestroyGeometry(context_->get_context(), geometry_);
            }
            geometry_ = other.geometry_;
            context_ = other.context_;
            other.geometry_ = nullptr;
            other.context_ = nullptr;
        }
        return *this;
    }

    hiprtGeometry get() const { return geometry_; }
    bool valid() const { return geometry_ != nullptr; }

private:
    hiprtGeometry geometry_ = nullptr;
    const HIPRTContext* context_ = nullptr;
};

/**
 * Geometry Builder
 *
 * Builds HIPRT geometries (BVHs) for triangles and custom AABBs.
 * Supports configurable build quality (LBVH vs SAH).
 */
class GeometryBuilder {
public:
    explicit GeometryBuilder(const HIPRTContext& context);
    ~GeometryBuilder();

    // No copy
    GeometryBuilder(const GeometryBuilder&) = delete;
    GeometryBuilder& operator=(const GeometryBuilder&) = delete;

    /**
     * Build triangle geometry (BLAS)
     *
     * @param vertices Array of vertex positions
     * @param triangles Array of triangle indices
     * @param quality Build quality (FAST for LBVH, HIGH_QUALITY for SAH)
     * @return Handle to built geometry
     */
    GeometryHandle build_triangle_geometry(
        const std::vector<Vertex>& vertices,
        const std::vector<Triangle>& triangles,
        BuildQuality quality = BuildQuality::FAST
    );

    /**
     * Build AABB geometry for custom primitives (neural assets)
     *
     * @param aabbs Array of axis-aligned bounding boxes
     * @param geom_type Custom geometry type ID (for function table lookup)
     * @param quality Build quality
     * @return Handle to built geometry
     */
    GeometryHandle build_aabb_geometry(
        const std::vector<AABB>& aabbs,
        uint32_t geom_type,
        BuildQuality quality = BuildQuality::FAST
    );

    /**
     * Get build flags for a given quality setting
     */
    static hiprtBuildFlags get_build_flags(BuildQuality quality);

private:
    const HIPRTContext& context_;

    // Temporary buffers for geometry builds (reused)
    void* temp_buffer_ = nullptr;
    size_t temp_buffer_size_ = 0;

    void ensure_temp_buffer(size_t required_size);
    void free_temp_buffer();
};

} // namespace hiprt
