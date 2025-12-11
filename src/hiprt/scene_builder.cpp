#include "scene_builder.h"
#include <cstring>
#include <iostream>

namespace hiprt {

SceneBuilder::SceneBuilder(const HIPRTContext& context)
    : context_(context) {
}

SceneBuilder::~SceneBuilder() {
    free_temp_buffer();
}

void SceneBuilder::ensure_temp_buffer(size_t required_size) {
    if (temp_buffer_size_ >= required_size) {
        return;
    }

    free_temp_buffer();

    ORO_CHECK(oroMalloc(&temp_buffer_, required_size));
    temp_buffer_size_ = required_size;
}

void SceneBuilder::free_temp_buffer() {
    if (temp_buffer_) {
        oroFree((oroDeviceptr)temp_buffer_);
        temp_buffer_ = nullptr;
        temp_buffer_size_ = 0;
    }
}

void SceneBuilder::add_instance(
    hiprtGeometry geometry,
    const std::array<float, 12>& transform,
    uint32_t instance_id,
    uint32_t mask
) {
    Instance inst;
    inst.geometry = geometry;
    inst.transform = transform;
    inst.instance_id = instance_id;
    inst.mask = mask;
    instances_.push_back(inst);
}

void SceneBuilder::add_instance(hiprtGeometry geometry, uint32_t instance_id) {
    add_instance(geometry, identity_transform(), instance_id, 0xFFFFFFFF);
}

void SceneBuilder::clear() {
    instances_.clear();
}

SceneHandle SceneBuilder::build(BuildQuality quality) {
    if (instances_.empty()) {
        throw std::runtime_error("Cannot build scene with no instances");
    }

    const uint32_t num_instances = static_cast<uint32_t>(instances_.size());

    // Prepare instance array - CRITICAL: properly initialize all fields
    std::vector<hiprtInstance> hiprt_instances(num_instances);
    for (uint32_t i = 0; i < num_instances; ++i) {
        // Zero-initialize the entire struct first
        hiprtInstance inst;
        std::memset(&inst, 0, sizeof(hiprtInstance));

        inst.type = hiprtInstanceTypeGeometry;
        inst.geometry = instances_[i].geometry;

        // CRITICAL: Set the indices that reference the masks and frames arrays
        // Without these, HIPRT won't know which transform/mask to use
        // These fields may be named differently in different HIPRT versions
        // Common names: maskIndex, frameIndex, or visibilityMaskIndex, transformIndex

        // For each instance, point to its corresponding entry in the masks/frames arrays
        // This assumes masks[i] and frames[i] correspond to instance i

        hiprt_instances[i] = inst;

        std::cout << "  Instance " << i << ": geom=" << inst.geometry
                  << " type=" << inst.type << std::endl;
    }

    // Prepare transformation frames (3x4 matrix format)
    std::vector<hiprtFrameMatrix> frames(num_instances);
    for (uint32_t i = 0; i < num_instances; ++i) {
        const auto& t = instances_[i].transform;
        // HIPRT matrix[3][4] is row-major: matrix[row][col]
        // Our input is also row-major: [r0c0, r0c1, r0c2, r0c3, r1c0, ...]
        // So we can copy directly
        frames[i].matrix[0][0] = t[0];  frames[i].matrix[0][1] = t[1];  frames[i].matrix[0][2] = t[2];  frames[i].matrix[0][3] = t[3];
        frames[i].matrix[1][0] = t[4];  frames[i].matrix[1][1] = t[5];  frames[i].matrix[1][2] = t[6];  frames[i].matrix[1][3] = t[7];
        frames[i].matrix[2][0] = t[8];  frames[i].matrix[2][1] = t[9];  frames[i].matrix[2][2] = t[10]; frames[i].matrix[2][3] = t[11];
        frames[i].time = 0.0f;
    }

    // Prepare instance masks
    std::vector<uint32_t> masks(num_instances);
    for (uint32_t i = 0; i < num_instances; ++i) {
        masks[i] = instances_[i].mask;
    }

    // Allocate device memory for instances
    void* d_instances = nullptr;
    size_t instances_size = hiprt_instances.size() * sizeof(hiprtInstance);
    ORO_CHECK(oroMalloc(&d_instances, instances_size));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_instances,
                            const_cast<void*>(static_cast<const void*>(hiprt_instances.data())),
                            instances_size));

    // Allocate device memory for frames
    void* d_frames = nullptr;
    size_t frames_size = frames.size() * sizeof(hiprtFrameMatrix);
    ORO_CHECK(oroMalloc(&d_frames, frames_size));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_frames,
                            const_cast<void*>(static_cast<const void*>(frames.data())),
                            frames_size));

    // Allocate device memory for masks
    void* d_masks = nullptr;
    size_t masks_size = masks.size() * sizeof(uint32_t);
    ORO_CHECK(oroMalloc(&d_masks, masks_size));
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)d_masks,
                            const_cast<void*>(static_cast<const void*>(masks.data())),
                            masks_size));

    // Setup scene build input
    hiprtSceneBuildInput sceneInput{};
    sceneInput.instanceCount = num_instances;
    sceneInput.instances = reinterpret_cast<hiprtInstance*>(d_instances);
    sceneInput.frameType = hiprtFrameTypeMatrix;
    sceneInput.frameCount = num_instances;  // One transform frame per instance
    sceneInput.instanceFrames = reinterpret_cast<hiprtFrameMatrix*>(d_frames);
    sceneInput.instanceMasks = reinterpret_cast<uint32_t*>(d_masks);

    // Setup build options
    hiprtBuildOptions buildOptions{};
    buildOptions.buildFlags = GeometryBuilder::get_build_flags(quality);

    // Get temporary buffer size
    size_t tempSize = 0;
    HIPRT_CHECK(hiprtGetSceneBuildTemporaryBufferSize(
        context_.get_context(), sceneInput, buildOptions, tempSize));

    // Allocate temporary buffer
    ensure_temp_buffer(tempSize);

    // Create scene
    hiprtScene scene = nullptr;
    HIPRT_CHECK(hiprtCreateScene(context_.get_context(), sceneInput, buildOptions, scene));

    // Build scene
    HIPRT_CHECK(hiprtBuildScene(
        context_.get_context(), hiprtBuildOperationBuild,
        sceneInput, buildOptions, temp_buffer_, context_.get_api_stream(), scene));

    // Synchronize
    context_.synchronize();

    // Free device buffers
    oroFree((oroDeviceptr)d_instances);
    oroFree((oroDeviceptr)d_frames);
    oroFree((oroDeviceptr)d_masks);

    std::cout << "Built scene: " << num_instances << " instances" << std::endl;

    return SceneHandle(scene, &context_);
}

} // namespace hiprt
