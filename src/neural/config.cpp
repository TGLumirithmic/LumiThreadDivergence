#include "config.h"
#include <iostream>
#include <cmath>

namespace neural {

void NetworkConfig::print() const {
    std::cout << "=== Network Configuration ===" << std::endl;
    std::cout << "Hash Encoding:" << std::endl;
    std::cout << "  Levels: " << n_levels << std::endl;
    std::cout << "  Features per level: " << n_features_per_level << std::endl;
    std::cout << "  Log2 hashmap size: " << log2_hashmap_size << std::endl;
    std::cout << "  Base resolution: " << base_resolution << std::endl;
    std::cout << "  Max resolution: " << max_resolution << std::endl;
    std::cout << "  Per-level scale: " << compute_per_level_scale() << std::endl;
    std::cout << "  Encoding output dims: " << encoding_n_output_dims() << std::endl;

    std::cout << "\nShared MLP:" << std::endl;
    std::cout << "  Neurons per layer: " << n_neurons << std::endl;
    std::cout << "  Activation: " << activation << std::endl;

    std::cout << "\nDecoders:" << std::endl;
    std::cout << "  Visibility: " << visibility_decoder.n_output_dims << "D ("
              << visibility_decoder.output_activation << ")" << std::endl;
    std::cout << "  Normal: " << normal_decoder.n_output_dims << "D ("
              << normal_decoder.output_activation << ")" << std::endl;
    std::cout << "  Depth: " << depth_decoder.n_output_dims << "D ("
              << depth_decoder.output_activation << ")" << std::endl;
    std::cout << "=============================" << std::endl;
}

NetworkConfig NetworkConfig::instant_ngp_default() {
    NetworkConfig config;
    // Use defaults defined in the header
    return config;
}

NetworkConfig NetworkConfig::from_json(const std::string& json_path) {
    // TODO: Implement JSON parsing
    // For Phase 1, we'll use hardcoded defaults
    std::cout << "Warning: JSON loading not yet implemented. Using defaults." << std::endl;
    return instant_ngp_default();
}

} // namespace neural
