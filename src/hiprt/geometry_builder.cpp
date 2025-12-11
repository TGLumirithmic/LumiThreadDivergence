#include "geometry_builder.h"
#include <cstring>
#include <iostream>

namespace hiprt {

GeometryBuilder::GeometryBuilder(const HIPRTContext& context)
    : context_(context) {
}

GeometryBuilder::~GeometryBuilder() {
    free_temp_buffer();
}

hiprtBuildFlags GeometryBuilder::get_build_flags(BuildQuality quality) {
    switch (quality) {
        case BuildQuality::FAST:
            return hiprtBuildFlagBitPreferFastBuild;
        case BuildQuality::HIGH_QUALITY:
            return hiprtBuildFlagBitPreferHighQualityBuild;
        case BuildQuality::BALANCED:
        default:
            return hiprtBuildFlagBitPreferBalancedBuild;
    }
}

void GeometryBuilder::ensure_temp_buffer(size_t required_size) {
    if (temp_buffer_size_ >= required_size) {
        return;
    }

    free_temp_buffer();

    ORO_CHECK(oroMalloc(&temp_buffer_, required_size));
    temp_buffer_size_ = required_size;
}

void GeometryBuilder::free_temp_buffer() {
    if (temp_buffer_) {
        oroFree((oroDeviceptr)temp_buffer_);
        temp_buffer_ = nullptr;
        temp_buffer_size_ = 0;
    }
}

GeometryHandle GeometryBuilder::build_triangle_geometry(
    const std::vector<Vertex>& vertices,
    const std::vector<Triangle>& triangles,
    BuildQuality quality
) {
    if (vertices.empty() || triangles.empty()) {
        throw std::runtime_error("Cannot build geometry with empty vertices or triangles");
    }

    // Allocate device memory for vertices
    void* d_vertices = nullptr;
    size_t vertices_size = vertices.size() * sizeof(Vertex);
    ORO_CHECK(oroMalloc(&d_vertices, vertices_size));

    // Allocate device memory for triangles
    void* d_triangles = nullptr;
    size_t triangles_size = triangles.size() * sizeof(Triangle);
    ORO_CHECK(oroMalloc(&d_triangles, triangles_size));

    // Copy data to device
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_vertices,
                            const_cast<void*>(static_cast<const void*>(vertices.data())),
                            vertices_size));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_triangles,
                            const_cast<void*>(static_cast<const void*>(triangles.data())),
                            triangles_size));

    // Setup triangle mesh input - zero-initialize all fields
    hiprtTriangleMeshPrimitive mesh{};
    mesh.vertices = d_vertices;  // hiprtDevicePtr is void*
    mesh.vertexCount = static_cast<uint32_t>(vertices.size());
    mesh.vertexStride = sizeof(Vertex);
    mesh.triangleIndices = d_triangles;  // hiprtDevicePtr is void*
    mesh.triangleCount = static_cast<uint32_t>(triangles.size());
    mesh.triangleStride = sizeof(Triangle);
    // trianglePairIndices and trianglePairCount are zero from {} init

    // Setup geometry build input - zero-initialize (has union + nodeList)
    hiprtGeometryBuildInput geomInput{};
    geomInput.type = hiprtPrimitiveTypeTriangleMesh;
    geomInput.primitive.triangleMesh = mesh;
    geomInput.geomType = 0;  // Default geometry type for triangles

    // Get build sizes
    hiprtBuildOptions buildOptions{};
    buildOptions.buildFlags = get_build_flags(quality);

    std::cout << "  Getting temp buffer size..." << std::endl;
    size_t geomTempSize = 0;
    hiprtDevicePtr geomTemp = nullptr;

    HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(
        context_.get_context(), geomInput, buildOptions, geomTempSize));
    std::cout << "  Temp buffer size: " << geomTempSize << " bytes" << std::endl;

    // Allocate temporary buffer
    ensure_temp_buffer(geomTempSize);
    geomTemp = temp_buffer_;
    std::cout << "  Temp buffer allocated" << std::endl;

    // Create and build geometry
    std::cout << "  Creating geometry..." << std::endl;
    hiprtGeometry geometry = nullptr;
    HIPRT_CHECK(hiprtCreateGeometry(
        context_.get_context(), geomInput, buildOptions, geometry));
    std::cout << "  Geometry created" << std::endl;

    std::cout << "  Building geometry..." << std::endl;
    HIPRT_CHECK(hiprtBuildGeometry(
        context_.get_context(), hiprtBuildOperationBuild,
        geomInput, buildOptions, geomTemp, context_.get_api_stream(), geometry));
    std::cout << "  Geometry built" << std::endl;

    // Synchronize to ensure build is complete
    context_.synchronize();

    // Free device buffers (geometry retains its own copy)
    oroFree((oroDeviceptr)d_vertices);
    oroFree((oroDeviceptr)d_triangles);

    std::cout << "Built triangle geometry: " << triangles.size() << " triangles, "
              << vertices.size() << " vertices" << std::endl;

    return GeometryHandle(geometry, &context_);
}

GeometryHandle GeometryBuilder::build_aabb_geometry(
    const std::vector<AABB>& aabbs,
    uint32_t geom_type,
    BuildQuality quality
) {
    if (aabbs.empty()) {
        throw std::runtime_error("Cannot build geometry with empty AABBs");
    }

    // Convert AABB format to hiprtFloat4 pairs (min, max)
    // HIPRT expects AABBs as float4 pairs: {min.xyz, _, max.xyz, _}
    std::vector<hiprtFloat4> aabb_data(aabbs.size() * 2);
    for (size_t i = 0; i < aabbs.size(); ++i) {
        aabb_data[i * 2 + 0] = {aabbs[i].min_x, aabbs[i].min_y, aabbs[i].min_z, 0.0f};
        aabb_data[i * 2 + 1] = {aabbs[i].max_x, aabbs[i].max_y, aabbs[i].max_z, 0.0f};
        std::cout << "  AABB[" << i << "]: min=(" << aabbs[i].min_x << "," << aabbs[i].min_y << "," << aabbs[i].min_z
                  << ") max=(" << aabbs[i].max_x << "," << aabbs[i].max_y << "," << aabbs[i].max_z << ")" << std::endl;
    }

    // Allocate device memory for AABBs
    void* d_aabbs = nullptr;
    size_t aabbs_size = aabb_data.size() * sizeof(hiprtFloat4);
    ORO_CHECK(oroMalloc(&d_aabbs, aabbs_size));

    // Copy data to device
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_aabbs,
                            const_cast<void*>(static_cast<const void*>(aabb_data.data())),
                            aabbs_size));

    // Setup AABB list input - use memset for complete zero-initialization
    hiprtAABBListPrimitive aabbList;
    std::memset(&aabbList, 0, sizeof(hiprtAABBListPrimitive));
    aabbList.aabbs = d_aabbs;
    aabbList.aabbCount = static_cast<uint32_t>(aabbs.size());
    aabbList.aabbStride = 2 * sizeof(hiprtFloat4);  // Each AABB is 2 float4s

    // Setup geometry build input - use memset for complete zero-initialization
    // This is critical: the union and all fields must be properly zeroed
    hiprtGeometryBuildInput geomInput;
    std::memset(&geomInput, 0, sizeof(hiprtGeometryBuildInput));
    geomInput.type = hiprtPrimitiveTypeAABBList;
    geomInput.primitive.aabbList = aabbList;
    geomInput.geomType = geom_type;  // Custom geometry type for neural assets

    // Get build sizes
    hiprtBuildOptions buildOptions;
    std::memset(&buildOptions, 0, sizeof(hiprtBuildOptions));
    // Use fast build to ensure internal deep copy of AABB data
    buildOptions.buildFlags = hiprtBuildFlagBitPreferFastBuild;

    size_t geomTempSize = 0;
    hiprtDevicePtr geomTemp = nullptr;

    std::cout << "  Getting AABB geometry temp buffer size..." << std::endl;
    HIPRT_CHECK(hiprtGetGeometryBuildTemporaryBufferSize(
        context_.get_context(), geomInput, buildOptions, geomTempSize));
    std::cout << "  AABB temp buffer size: " << geomTempSize << " bytes" << std::endl;

    // Allocate temporary buffer
    ensure_temp_buffer(geomTempSize);
    geomTemp = temp_buffer_;

    // Create and build geometry
    std::cout << "  Creating AABB geometry (geomType=" << geom_type << ")..." << std::endl;
    hiprtGeometry geometry = nullptr;
    HIPRT_CHECK(hiprtCreateGeometry(
        context_.get_context(), geomInput, buildOptions, geometry));
    std::cout << "  AABB geometry created" << std::endl;

    std::cout << "  Building AABB geometry..." << std::endl;
    HIPRT_CHECK(hiprtBuildGeometry(
        context_.get_context(), hiprtBuildOperationBuild,
        geomInput, buildOptions, geomTemp, context_.get_api_stream(), geometry));
    std::cout << "  AABB geometry built" << std::endl;

    // Synchronize to ensure build is complete
    context_.synchronize();

    // Free device buffers - safe with hiprtBuildFlagBitPreferFastBuild which deep copies
    oroFree((oroDeviceptr)d_aabbs);

    std::cout << "Built AABB geometry: " << aabbs.size() << " AABBs, geomType=" << geom_type << std::endl;

    return GeometryHandle(geometry, &context_);
}

} // namespace hiprt
