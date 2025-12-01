#include "weight_loader.h"
#include <iostream>
#include <fstream>
#include <cstring>

namespace neural {

void Tensor::print_info() const {
    std::cout << "Tensor: " << name << std::endl;
    std::cout << "  Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Total elements: " << total_size() << std::endl;
    std::cout << "  Data size: " << data.size() << " floats" << std::endl;
}

bool WeightLoader::load_from_file(const std::string& path) {
    tensors_.clear();

    // This loader reads a custom binary format (not .pth directly)
    // Use the provided Python script (scripts/convert_checkpoint.py) to convert
    // PyTorch .pth files to this format
    //
    // Binary format:
    //   Header: "TCNN" (4 bytes magic number)
    //   For each tensor:
    //     - uint32_t: name_length
    //     - char[name_length]: tensor name (no null terminator in file)
    //     - uint32_t: num_dims
    //     - uint64_t[num_dims]: shape dimensions
    //     - float[product(shape)]: weight data in row-major order
    //   End marker: name_length = 0

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open weight file: " << path << std::endl;
        std::cerr << "Make sure to convert PyTorch .pth files using scripts/convert_checkpoint.py" << std::endl;
        return false;
    }

    // Check magic number
    char magic[4];
    file.read(magic, 4);
    if (std::strncmp(magic, "TCNN", 4) != 0) {
        std::cerr << "Invalid weight file format. Expected 'TCNN' magic number." << std::endl;
        std::cerr << "Use scripts/convert_checkpoint.py to convert PyTorch checkpoints." << std::endl;
        return false;
    }

    std::cout << "Loading weights from: " << path << std::endl;

    while (true) {
        Tensor tensor;

        // Read name length
        uint32_t name_length;
        file.read(reinterpret_cast<char*>(&name_length), sizeof(uint32_t));

        if (file.eof() || name_length == 0) break;  // End of file or end marker

        // Read name
        std::vector<char> name_buffer(name_length + 1);
        file.read(name_buffer.data(), name_length);
        name_buffer[name_length] = '\0';
        tensor.name = std::string(name_buffer.data());

        // Read shape
        uint32_t num_dims;
        file.read(reinterpret_cast<char*>(&num_dims), sizeof(uint32_t));
        tensor.shape.resize(num_dims);

        for (uint32_t i = 0; i < num_dims; ++i) {
            uint64_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(uint64_t));
            tensor.shape[i] = static_cast<size_t>(dim);
        }

        // Read data
        size_t total_elements = tensor.total_size();
        tensor.data.resize(total_elements);
        file.read(reinterpret_cast<char*>(tensor.data.data()), total_elements * sizeof(float));

        if (!file.good()) {
            std::cerr << "Error reading tensor: " << tensor.name << std::endl;
            return false;
        }

        std::cout << "  Loaded: " << tensor.name << " - [";
        for (size_t i = 0; i < tensor.shape.size(); ++i) {
            std::cout << tensor.shape[i];
            if (i < tensor.shape.size() - 1) std::cout << "x";
        }
        std::cout << "]" << std::endl;

        tensors_[tensor.name] = std::move(tensor);
    }

    file.close();
    std::cout << "Loaded " << tensors_.size() << " tensors." << std::endl;
    return !tensors_.empty();
}

const Tensor* WeightLoader::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it != tensors_.end()) {
        return &it->second;
    }
    return nullptr;
}

std::vector<std::string> WeightLoader::get_tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    return names;
}

void WeightLoader::print_summary() const {
    std::cout << "\n=== Weight Summary ===" << std::endl;
    std::cout << "Total tensors: " << tensors_.size() << std::endl;
    for (const auto& pair : tensors_) {
        pair.second.print_info();
    }
    std::cout << "=====================" << std::endl;
}

std::string WeightLoader::make_key(const std::string& prefix, int layer, const std::string& suffix) const {
    return prefix + std::to_string(layer) + "." + suffix;
}

const Tensor* WeightLoader::get_encoder_weight(int layer) const {
    return get_tensor("encoder.layer." + std::to_string(layer) + ".weight");
}

const Tensor* WeightLoader::get_encoder_bias(int layer) const {
    return get_tensor("encoder.layer." + std::to_string(layer) + ".bias");
}

const Tensor* WeightLoader::get_decoder_weight(const std::string& decoder_name, int layer) const {
    return get_tensor(decoder_name + ".layer." + std::to_string(layer) + ".weight");
}

const Tensor* WeightLoader::get_decoder_bias(const std::string& decoder_name, int layer) const {
    return get_tensor(decoder_name + ".layer." + std::to_string(layer) + ".bias");
}

} // namespace neural
