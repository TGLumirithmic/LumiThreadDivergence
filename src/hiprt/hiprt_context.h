#pragma once

// Include Orochi first - it provides abstraction over CUDA/HIP
// Do NOT include cuda_runtime.h as it conflicts with Orochi types
#include <Orochi/Orochi.h>
#include <hiprt/hiprt.h>
#include <string>
#include <stdexcept>

namespace hiprt {

// Error checking macro for HIPRT
#define HIPRT_CHECK(call)                                                           \
    do {                                                                            \
        hiprtError err = call;                                                      \
        if (err != hiprtSuccess) {                                                  \
            throw std::runtime_error(std::string("HIPRT error: ") +                 \
                                   std::to_string(err) + " at " + __FILE__ +        \
                                   ":" + std::to_string(__LINE__));                 \
        }                                                                           \
    } while (0)

// Error checking macro for Orochi/CUDA
#define ORO_CHECK(call)                                                             \
    do {                                                                            \
        oroError err = call;                                                        \
        if (err != oroSuccess) {                                                    \
            throw std::runtime_error(std::string("Orochi error: ") +                \
                                   std::to_string(err) + " at " + __FILE__ +        \
                                   ":" + std::to_string(__LINE__));                 \
        }                                                                           \
    } while (0)

/**
 * HIPRT Context Manager
 *
 * Manages the HIP/CUDA device context and HIPRT context.
 * Uses Orochi for cross-platform GPU abstraction.
 */
class HIPRTContext {
public:
    HIPRTContext() = default;
    ~HIPRTContext();

    // No copy
    HIPRTContext(const HIPRTContext&) = delete;
    HIPRTContext& operator=(const HIPRTContext&) = delete;

    // Move is ok
    HIPRTContext(HIPRTContext&& other) noexcept;
    HIPRTContext& operator=(HIPRTContext&& other) noexcept;

    /**
     * Initialize the context for GPU rendering
     * @param device_id GPU device ID (default 0)
     * @return true on success
     */
    bool initialize(int device_id = 0);

    /**
     * Clean up resources
     */
    void cleanup();

    /**
     * Get the HIPRT context handle
     */
    hiprtContext get_context() const { return context_; }

    /**
     * Get the Orochi device context
     */
    oroCtx get_oro_context() const { return oro_ctx_; }

    /**
     * Get the Orochi stream
     */
    oroStream get_stream() const { return stream_; }

    /**
     * Get the raw API stream for HIPRT calls
     * oroStream is already a pointer type (hipStream_t), cast to void*
     */
    hiprtApiStream get_api_stream() const { return reinterpret_cast<hiprtApiStream>(stream_); }

    /**
     * Get device properties
     */
    const oroDeviceProp& get_device_props() const { return device_props_; }

    /**
     * Synchronize the stream
     */
    void synchronize() const;

    /**
     * Check if context is initialized
     */
    bool is_initialized() const { return initialized_; }

    /**
     * Allocate device memory
     */
    void* allocate(size_t size) const;

    /**
     * Free device memory
     */
    void free(void* ptr) const;

    /**
     * Copy host to device
     */
    void copy_to_device(void* dst, const void* src, size_t size) const;

    /**
     * Copy device to host
     */
    void copy_to_host(void* dst, const void* src, size_t size) const;

private:
    bool initialized_ = false;

    // Orochi handles
    oroDevice oro_device_ = 0;
    oroCtx oro_ctx_ = nullptr;
    oroStream stream_ = nullptr;
    oroDeviceProp device_props_{};

    // HIPRT context
    hiprtContext context_ = nullptr;
};

} // namespace hiprt
