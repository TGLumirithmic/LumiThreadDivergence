#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

namespace neural {

// Tensor data structure for storing loaded weights
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;
    std::string name;

    size_t total_size() const {
        size_t size = 1;
        for (auto dim : shape) {
            size *= dim;
        }
        return size;
    }

    void print_info() const;
};

// Weight loader for PyTorch checkpoints
class WeightLoader {
public:
    WeightLoader() = default;

    // Load weights from PyTorch .pth file
    // For Phase 1, this will be a simple binary format
    // Later can be extended to use libtorch or custom parser
    bool load_from_file(const std::string& path);

    // Get tensor by name
    const Tensor* get_tensor(const std::string& name) const;

    // Get all tensor names
    std::vector<std::string> get_tensor_names() const;

    // Print all loaded tensors
    void print_summary() const;

    // Check if weights are loaded
    bool is_loaded() const { return !tensors_.empty(); }

    // Get decoder weights by name
    const Tensor* get_encoder_weight(int layer) const;
    const Tensor* get_encoder_bias(int layer) const;
    const Tensor* get_decoder_weight(const std::string& decoder_name, int layer) const;
    const Tensor* get_decoder_bias(const std::string& decoder_name, int layer) const;

private:
    std::map<std::string, Tensor> tensors_;

    // Helper to read binary tensor file
    bool load_binary_format(const std::string& path);

    // Helper to parse tensor name patterns
    std::string make_key(const std::string& prefix, int layer, const std::string& suffix) const;
};

} // namespace neural
