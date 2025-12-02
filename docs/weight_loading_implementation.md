# Weight Loading Implementation

## Overview

The weight loading system converts PyTorch checkpoint weights into tiny-cuda-nn network parameters.

## Implementation Details

### Location
- **File:** [src/neural/network.cpp](../src/neural/network.cpp)
- **Function:** `load_network_weights()`
- **Called from:** `initialize_from_weights()`

### Process Flow

1. **Weight File Loading** ([src/neural/weight_loader.cpp](../src/neural/weight_loader.cpp))
   - Binary format with magic header "TCNN"
   - Stores tensors as: name, shape, data (float32)
   - Convert from PyTorch using [scripts/convert_checkpoint.py](../scripts/convert_checkpoint.py)

2. **Network Creation**
   - Creates tiny-cuda-nn networks with specified architecture
   - Three decoders: visibility (1D), normal (3D), depth (1D)
   - Uses FullyFusedMLP for efficiency

3. **Weight Loading** (`load_network_weights`)
   ```cpp
   bool load_network_weights(
       void* network_ptr,        // tiny-cuda-nn Network pointer
       const WeightLoader& loader,  // Loaded weights
       const std::string& prefix,   // "visibility", "normal", or "depth"
       uint32_t n_output           // Expected output dimensions
   )
   ```

### Weight Loading Steps

#### 1. Query Network Parameters
```cpp
auto* network = reinterpret_cast<Network<precision_t>*>(network_ptr);
size_t n_params = network->n_params();
```

#### 2. Load Tensors from File
For each decoder (e.g., "visibility"):
- **Weight tensor:** `visibility.layer0.weight` - shape [n_output, n_input]
- **Bias tensor:** `visibility.layer0.bias` - shape [n_output]

#### 3. Validate Shapes
```cpp
if (n_output_actual != n_output) {
    // Error: dimension mismatch
}
```

#### 4. Flatten Parameters
Tiny-cuda-nn expects a flat parameter array:
```
[weight_00, weight_01, ..., weight_nm, bias_0, bias_1, ..., bias_n]
```

Copy weights and biases sequentially:
```cpp
size_t offset = 0;
// Copy all weight values
for (size_t i = 0; i < weight_tensor->data.size(); ++i) {
    params_fp32[offset + i] = weight_tensor->data[i];
}
offset += weight_tensor->data.size();

// Copy all bias values
for (size_t i = 0; i < bias_tensor->data.size(); ++i) {
    params_fp32[offset + i] = bias_tensor->data[i];
}
```

#### 5. Convert to Half Precision
Tiny-cuda-nn uses `__half` (FP16) for efficiency:
```cpp
std::vector<precision_t> params_fp16(n_params);
for (size_t i = 0; i < n_params; ++i) {
    params_fp16[i] = (precision_t)params_fp32[i];
}
```

#### 6. Upload to GPU
```cpp
precision_t* d_params;
cudaMalloc(&d_params, n_params * sizeof(precision_t));
cudaMemcpy(d_params, params_fp16.data(),
           n_params * sizeof(precision_t),
           cudaMemcpyHostToDevice);
```

#### 7. Set Network Parameters
```cpp
network->set_params(d_params, d_params + n_params);
cudaFree(d_params);  // Network has copied internally
```

## Expected Tensor Naming Convention

### Visibility Decoder
- `visibility.layer0.weight` - [1 × n_hidden]
- `visibility.layer0.bias` - [1]

### Normal Decoder
- `normal.layer0.weight` - [3 × n_hidden]
- `normal.layer0.bias` - [3]

### Depth Decoder
- `depth.layer0.weight` - [1 × n_hidden]
- `depth.layer0.bias` - [1]

Where `n_hidden` = encoding output dimensions (default: 32)

## PyTorch Model Structure

Your PyTorch model should follow this structure:

```python
import torch
import torch.nn as nn

class MultiDecoderNetwork(nn.Module):
    def __init__(self, encoding_dim=32, hidden_dim=64):
        super().__init__()

        # Three decoder heads (no shared encoder in this version)
        self.visibility = nn.Linear(encoding_dim, 1)
        self.normal = nn.Linear(encoding_dim, 3)
        self.depth = nn.Linear(encoding_dim, 1)

    def forward(self, encoded_features):
        vis = torch.sigmoid(self.visibility(encoded_features))
        norm = self.normal(encoded_features)
        norm = norm / (torch.norm(norm, dim=-1, keepdim=True) + 1e-8)
        depth = self.depth(encoded_features)
        return vis, norm, depth

# Save checkpoint
model = MultiDecoderNetwork()
torch.save({
    'state_dict': model.state_dict()
}, 'model.pth')
```

Then convert:
```bash
python scripts/convert_checkpoint.py model.pth data/models/model.bin
```

## Validation

After loading, the test program verifies:
1. All tensors found in checkpoint
2. Shape compatibility with network architecture
3. Parameter count matches expected

Output shows:
```
Loading weights into networks...
  Loading weights for visibility decoder...
    Network expects 64 parameters
    Loading weight matrix: [1 x 32]
    Loading bias vector: [1]
    Successfully loaded 33 parameters
  Loading weights for normal decoder...
    ...
  All weights loaded successfully!
```

## Troubleshooting

### Parameter Count Mismatch
```
Warning: Parameter count mismatch. Loaded 33 but network expects 64
```

**Cause:** Network has more layers than just the output layer.

**Solution:** Check network configuration:
- `n_hidden_layers` should match PyTorch model
- Encoding dimensions must match

### Tensor Not Found
```
Warning: Could not find tensors visibility.layer0.weight
```

**Cause:** PyTorch checkpoint uses different naming.

**Solution:** Either:
1. Rename tensors in PyTorch before saving
2. Modify `convert_checkpoint.py` to map names
3. Update `load_network_weights()` to use correct names

### Shape Mismatch
```
Error: Output dimension mismatch. Expected 1 but got 3
```

**Cause:** Wrong decoder being loaded for tensor.

**Solution:** Verify checkpoint has correct structure:
```bash
python scripts/convert_checkpoint.py model.pth --info
```

## Current Limitations

1. **Single Layer Decoders:** Currently only loads one layer per decoder
2. **No Shared Encoder:** Shared MLP encoder weights not yet implemented
3. **No Hash Grid Weights:** Hash encoding uses random initialization
4. **FP16 Conversion:** Simple cast, could use more sophisticated conversion

## Future Enhancements

- [ ] Load multi-layer MLPs
- [ ] Load shared encoder weights
- [ ] Load hash grid parameters (if trainable)
- [ ] Support different precision modes
- [ ] Validate loaded weights with test predictions
