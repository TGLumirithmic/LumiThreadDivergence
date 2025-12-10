#include "hiprt_context.h"
#include <iostream>
#include <cstring>

namespace hiprt {

HIPRTContext::~HIPRTContext() {
    cleanup();
}

HIPRTContext::HIPRTContext(HIPRTContext&& other) noexcept
    : initialized_(other.initialized_)
    , oro_device_(other.oro_device_)
    , oro_ctx_(other.oro_ctx_)
    , stream_(other.stream_)
    , device_props_(other.device_props_)
    , context_(other.context_)
{
    other.initialized_ = false;
    other.oro_ctx_ = nullptr;
    other.stream_ = nullptr;
    other.context_ = nullptr;
}

HIPRTContext& HIPRTContext::operator=(HIPRTContext&& other) noexcept {
    if (this != &other) {
        cleanup();
        initialized_ = other.initialized_;
        oro_device_ = other.oro_device_;
        oro_ctx_ = other.oro_ctx_;
        stream_ = other.stream_;
        device_props_ = other.device_props_;
        context_ = other.context_;

        other.initialized_ = false;
        other.oro_ctx_ = nullptr;
        other.stream_ = nullptr;
        other.context_ = nullptr;
    }
    return *this;
}

bool HIPRTContext::initialize(int device_id) {
    if (initialized_) {
        std::cerr << "HIPRTContext already initialized" << std::endl;
        return false;
    }

    try {
        // Initialize Orochi - try CUDA first (for NVIDIA GPUs)
        oroApi api = ORO_API_CUDADRIVER;
        if (oroInitialize(api, 0) != oroSuccess) {
            // Fallback to HIP
            api = ORO_API_HIP;
            if (oroInitialize(api, 0) != oroSuccess) {
                std::cerr << "Failed to initialize Orochi (neither CUDA nor HIP available)" << std::endl;
                return false;
            }
            std::cout << "Initialized Orochi with HIP backend" << std::endl;
        } else {
            std::cout << "Initialized Orochi with CUDA backend" << std::endl;
        }

        // Initialize the driver
        ORO_CHECK(oroInit(0));

        // Get device
        int device_count = 0;
        ORO_CHECK(oroGetDeviceCount(&device_count));
        if (device_count == 0) {
            std::cerr << "No GPU devices found" << std::endl;
            return false;
        }
        if (device_id >= device_count) {
            std::cerr << "Invalid device ID " << device_id << " (only " << device_count << " devices)" << std::endl;
            return false;
        }

        ORO_CHECK(oroDeviceGet(&oro_device_, device_id));

        // Get device properties
        ORO_CHECK(oroGetDeviceProperties(&device_props_, oro_device_));
        std::cout << "Using GPU: " << device_props_.name << std::endl;
        std::cout << "  Compute capability: " << device_props_.major << "." << device_props_.minor << std::endl;
        std::cout << "  Total memory: " << (device_props_.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;

        // Create context
        ORO_CHECK(oroCtxCreate(&oro_ctx_, 0, oro_device_));

        // Create stream
        ORO_CHECK(oroStreamCreate(&stream_));

        // Determine device type for HIPRT
        hiprtDeviceType deviceType;
        if (api == ORO_API_CUDADRIVER) {
            deviceType = hiprtDeviceNVIDIA;
        } else {
            deviceType = hiprtDeviceAMD;
        }

        // Create HIPRT context
        // IMPORTANT: HIPRT needs the raw driver handles, not the Orochi wrapper types
        hiprtContextCreationInput contextInput;
        contextInput.ctxt = oroGetRawCtx(oro_ctx_);
        contextInput.device = oroGetRawDevice(oro_device_);
        contextInput.deviceType = deviceType;

        HIPRT_CHECK(hiprtCreateContext(HIPRT_API_VERSION, contextInput, context_));

        // Enable verbose logging for debugging
        hiprtSetLogLevel(context_, hiprtLogLevelInfo | hiprtLogLevelWarn | hiprtLogLevelError);

        std::cout << "HIPRT context created successfully" << std::endl;
        std::cout << "  HIPRT version: " << HIPRT_MAJOR_VERSION << "."
                  << HIPRT_MINOR_VERSION << "." << HIPRT_PATCH_VERSION << std::endl;

        initialized_ = true;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize HIPRT context: " << e.what() << std::endl;
        cleanup();
        return false;
    }
}

void HIPRTContext::cleanup() {
    if (!initialized_) return;

    if (context_) {
        hiprtDestroyContext(context_);
        context_ = nullptr;
    }

    if (stream_) {
        oroStreamDestroy(stream_);
        stream_ = nullptr;
    }

    if (oro_ctx_) {
        oroCtxDestroy(oro_ctx_);
        oro_ctx_ = nullptr;
    }

    initialized_ = false;
}

void HIPRTContext::synchronize() const {
    if (stream_) {
        ORO_CHECK(oroStreamSynchronize(stream_));
    }
}

void* HIPRTContext::allocate(size_t size) const {
    void* ptr = nullptr;
    ORO_CHECK(oroMalloc((oroDeviceptr*)&ptr, size));
    return ptr;
}

void HIPRTContext::free(void* ptr) const {
    if (ptr) {
        oroFree((oroDeviceptr)ptr);
    }
}

void HIPRTContext::copy_to_device(void* dst, const void* src, size_t size) const {
    // Cast away const - Orochi API doesn't use const for source, but it's safe
    ORO_CHECK(oroMemcpyHtoD((oroDeviceptr)dst, const_cast<void*>(src), size));
}

void HIPRTContext::copy_to_host(void* dst, const void* src, size_t size) const {
    ORO_CHECK(oroMemcpyDtoH(dst, (oroDeviceptr)src, size));
}

} // namespace hiprt
