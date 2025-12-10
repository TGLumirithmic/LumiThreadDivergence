/**
 * Simple test to verify HIPRT linking and context creation
 * Compile: nvcc -I/opt/hiprt/include test_hiprt.cpp hiprt_context.cpp -L/opt/hiprt/lib -lhiprt -lcuda -o test_hiprt
 */

#include "hiprt_context.h"
#include <iostream>

int main() {
    std::cout << "=== HIPRT Linking and Context Test ===" << std::endl;

    hiprt::HIPRTContext context;

    std::cout << "\n1. Initializing HIPRT context..." << std::endl;
    if (!context.initialize(0)) {
        std::cerr << "FAILED: Could not initialize HIPRT context" << std::endl;
        return 1;
    }

    std::cout << "\n2. Context initialized successfully!" << std::endl;
    std::cout << "   HIPRT context handle: " << (context.get_context() ? "valid" : "null") << std::endl;
    std::cout << "   Orochi context handle: " << (context.get_oro_context() ? "valid" : "null") << std::endl;
    std::cout << "   Stream handle: " << (context.get_stream() ? "valid" : "null") << std::endl;

    std::cout << "\n3. Testing memory allocation..." << std::endl;
    void* d_test = context.allocate(1024);
    if (d_test) {
        std::cout << "   Allocated 1KB device memory: " << d_test << std::endl;
        context.free(d_test);
        std::cout << "   Freed device memory successfully" << std::endl;
    } else {
        std::cerr << "FAILED: Could not allocate device memory" << std::endl;
        return 1;
    }

    std::cout << "\n4. Testing synchronization..." << std::endl;
    context.synchronize();
    std::cout << "   Stream synchronized successfully" << std::endl;

    std::cout << "\n=== All tests PASSED ===" << std::endl;
    return 0;
}
