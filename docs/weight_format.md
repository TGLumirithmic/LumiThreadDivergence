# Weight Format Documentation

## Overview

This document describes how to convert PyTorch Instant-NGP checkpoints to the binary format used by this renderer.

## PyTorch to Binary Conversion

### Using the Conversion Script

The repository includes a Python script to convert PyTorch `.pth` files to our binary format:

```bash
# Basic usage - output file is auto-generated
python scripts/convert_checkpoint.py model.pth

# Specify output file
python scripts/convert_checkpoint.py model.pth output.bin

# Print checkpoint info without converting
python scripts/convert_checkpoint.py model.pth --info
```

### Expected PyTorch Checkpoint Structure

The PyTorch checkpoint should contain a state dictionary with the following naming convention for the multi-decoder architecture:

#### Hash Encoding Parameters
- Hash grid data is embedded in the network and managed by tiny-cuda-nn directly
- No explicit hash table weights need to be stored

#### Shared Encoder (MLP before decoder split)
- `encoder.layer0.weight` - First hidden layer weights [n_neurons × encoding_dim]
- `encoder.layer0.bias` - First hidden layer biases [n_neurons]
- `encoder.layer1.weight` - Second hidden layer weights [n_neurons × n_neurons]
- `encoder.layer1.bias` - Second hidden layer biases [n_neurons]
- ... (for each hidden layer)

#### Visibility Decoder (Any-Hit)
- `visibility.layer0.weight` - Output layer weights [1 × n_neurons]
- `visibility.layer0.bias` - Output layer bias [1]

#### Normal Decoder
- `normal.layer0.weight` - Output layer weights [3 × n_neurons]
- `normal.layer0.bias` - Output layer biases [3]

#### Depth Decoder (Closest-Hit)
- `depth.layer0.weight` - Output layer weights [1 × n_neurons]
- `depth.layer0.bias` - Output layer bias [1]

### Binary Format Specification

The custom binary format is designed for fast loading without external dependencies:

```
File Structure:
┌─────────────────────────────────────┐
│ Magic Header: "TCNN" (4 bytes)     │
├─────────────────────────────────────┤
│ Tensor 1:                           │
│   ├─ name_length (uint32)          │
│   ├─ name (char[name_length])      │
│   ├─ num_dims (uint32)              │
│   ├─ shape (uint64[num_dims])       │
│   └─ data (float32[product(shape)]) │
├─────────────────────────────────────┤
│ Tensor 2: ...                       │
├─────────────────────────────────────┤
│ ...                                 │
├─────────────────────────────────────┤
│ End Marker: 0 (uint32)              │
└─────────────────────────────────────┘
```

**Data Layout:**
- All integers are little-endian
- Floats are IEEE 754 single precision
- Tensor data is in row-major (C-style) order
- No padding between fields

## Example: Training a Compatible Model

Here's a PyTorch example showing the expected model structure:

```python
import torch
import torch.nn as nn

class InstantNGPMultiDecoder(nn.Module):
    def __init__(self, encoding_dim=32, hidden_dim=64, n_hidden=2):
        super().__init__()

        # Shared encoder (hash encoding output -> hidden features)
        encoder_layers = []
        encoder_layers.append(nn.Linear(encoding_dim, hidden_dim))
        for i in range(n_hidden - 1):
            encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.encoder = nn.ModuleList(encoder_layers)

        # Three decoders
        self.visibility = nn.Linear(hidden_dim, 1)  # Sigmoid activation
        self.normal = nn.Linear(hidden_dim, 3)      # Normalized
        self.depth = nn.Linear(hidden_dim, 1)       # Distance

    def forward(self, x):
        # Shared encoding
        for layer in self.encoder:
            x = torch.relu(layer(x))

        # Decoders
        vis = torch.sigmoid(self.visibility(x))
        norm = self.normal(x)
        norm = norm / (torch.norm(norm, dim=-1, keepdim=True) + 1e-8)
        depth = self.depth(x)

        return vis, norm, depth

# Save checkpoint
model = InstantNGPMultiDecoder()
torch.save({
    'state_dict': model.state_dict(),
    'config': {
        'encoding_dim': 32,
        'hidden_dim': 64,
        'n_hidden': 2
    }
}, 'model.pth')
```

## Loading Weights in C++

The `WeightLoader` class handles loading the binary format:

```cpp
#include "neural/weight_loader.h"

neural::WeightLoader loader;
if (!loader.load_from_file("data/models/model.bin")) {
    std::cerr << "Failed to load weights" << std::endl;
    return -1;
}

// Access tensors by name
const auto* weight = loader.get_tensor("encoder.layer0.weight");
if (weight) {
    std::cout << "Loaded encoder layer 0 weight: "
              << weight->shape[0] << "x" << weight->shape[1] << std::endl;
}

// Or use helper methods
const auto* vis_weight = loader.get_decoder_weight("visibility", 0);
```

## Validation

After conversion, you can validate the binary file:

```bash
# Build and run the Phase 1 test
mkdir build && cd build
cmake ..
make test_network

# Run test (will load weights and print summary)
./bin/test_network ../data/models/model.bin
```

The test program will:
1. Load the binary weights
2. Print a summary of all tensors
3. Initialize tiny-cuda-nn with the loaded weights
4. Run test queries and visualize outputs

## Troubleshooting

### "Invalid weight file format"
- Make sure you used the conversion script
- Check that the input `.pth` file is a valid PyTorch checkpoint
- Use `--info` flag to inspect the checkpoint structure

### "Missing tensor: encoder.layer0.weight"
- Your PyTorch checkpoint uses different naming conventions
- Modify the conversion script or retrain with the expected names

### Shape mismatches
- Verify that your network configuration matches the checkpoint
- Check that `n_neurons` and `n_hidden_layers` are correct
- Ensure encoding dimension matches (n_levels × n_features_per_level)
