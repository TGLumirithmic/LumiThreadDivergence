# Weight Loading Implementation Changes

## Summary

Updated the checkpoint conversion and weight loading system to:
1. Use sequential layer numbering (0,1,2,3,4...) instead of PyTorch Sequential format (0,2,4,6,8...)
2. Remove bias loading (FullyFusedMLP doesn't support biases)
3. Properly load the output layer into tiny-cuda-nn's parameter buffer

## Key Findings

### tiny-cuda-nn FullyFusedMLP Structure

With `n_hidden_layers=N`, tiny-cuda-nn creates:
- **1 first/input layer**: `[network_width, input_width]`
- **(N-1) hidden matmul layers**: `[network_width, network_width]`
- **1 output layer**: `[padded_output_width, network_width]` where `padded_output_width = next_multiple(output_dim, 16)`
- **Total layers**: N+1

Example with `n_hidden_layers=4`, `input=48`, `network_width=32`, `output=1`:
- Layer 0: `[32, 48]` = 1,536 params
- Layers 1-3: 3 × `[32, 32]` = 3,072 params
- Output layer: `[16, 32]` = 512 params (padded from 1 to 16)
- **Total: 5,120 parameters**

**Note**: The conversion script automatically pads the output layer when converting the checkpoint, so your PyTorch model can have `[1, 32]` output and it will be converted to `[16, 32]`.

### Bias Handling

**FullyFusedMLP does NOT support biases!**

- No bias parameters are stored in the parameter buffer
- First layer can implicitly learn bias through input padding with constant 1.0
- PyTorch models must use `bias=False` for all layers
- First layer should have input dimension = actual_input + 1 (for bias term)

## Changes Made

### 1. Conversion Script ([convert_checkpoint.py](../scripts/convert_checkpoint.py))

**Added automatic preprocessing**:

1. **Layer renumbering** - `renumber_decoder_layers()` converts layer numbering:
   - **Before**: `decoder.0.weight`, `decoder.2.weight`, `decoder.4.weight` ...
   - **After**: `decoder.0.weight`, `decoder.1.weight`, `decoder.2.weight` ...

2. **Output padding** - `pad_output_layers()` pads output layers to multiple of 16:
   - Visibility decoder: `[1, 32]` → `[16, 32]`
   - Normal decoder: `[3, 32]` → `[16, 32]`
   - Depth decoder: `[1, 32]` → `[16, 32]`
   - This matches tiny-cuda-nn's tensor core requirements

Usage:
```bash
python scripts/convert_checkpoint.py model.pth output.bin
# Automatically renumbers layers AND pads output dimensions
```

### 2. C++ Weight Loading ([network.cpp](../src/neural/network.cpp))

**Updated logic**:
- Loads `n_decoder_layers + 2` total layers (input + hidden + output)
- Uses sequential numbering: `prefix.0.weight`, `prefix.1.weight`, etc.
- **No bias loading** - biases are ignored/not expected
- Output layer is loaded into tiny-cuda-nn's parameter buffer

**Layer loading sequence** (for `n_decoder_layers=4`):
```cpp
// Load input + hidden layers (0-4)
for (i = 0; i < 5; i++) {
    load_weight(prefix + "." + std::to_string(i) + ".weight");
}
// Load output layer (5)
load_weight(prefix + ".5.weight");
```

### 3. Configuration ([config.h](../src/neural/config.h))

Added parameters:
- `n_decoder_layers = 4` - Number of hidden layers (not counting output)
- `n_neurons = 32` - Neuron count per hidden layer

## PyTorch Model Requirements

Your new PyTorch model must:

1. **No biases**: Use `bias=False` for all `nn.Linear` layers
2. **Input padding**: First layer input = actual_dim + 1 (pad with constant 1.0 during forward)
3. **Sequential naming**: Save checkpoint with sequential layer numbering

Example model structure:
```python
class BiasFreeMLP(nn.Module):
    def __init__(self, input_dim=48, hidden_dim=32, output_dim=1, n_hidden=4):
        super().__init__()
        # First layer: (input + 1) for padding -> hidden
        self.first_layer = nn.Linear(input_dim + 1, hidden_dim, bias=False)

        # Hidden layers: hidden -> hidden (n_hidden - 1 layers)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(n_hidden - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        # Pad input with 1.0
        ones = torch.ones(x.shape[0], 1, device=x.device)
        x = torch.cat([x, ones], dim=1)

        # Forward through layers
        x = F.relu(self.first_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)
```

## Testing

To verify the changes work:

1. **Train model** without biases and with input padding
2. **Save checkpoint** with sequential layer naming
3. **Convert** using the updated conversion script:
   ```bash
   python scripts/convert_checkpoint.py model.pth model.bin
   ```
4. **Load in C++** - should see output like:
   ```
   Network expects 5120 parameters
   Layer 0: Loading weight matrix [32 x 49]
   Layer 1: Loading weight matrix [32 x 32]
   Layer 2: Loading weight matrix [32 x 32]
   Layer 3: Loading weight matrix [32 x 32]
   Output layer: Loading weight matrix [1 x 32]
   Successfully loaded 5120 parameters from 5 layers
   ```

## Migration Notes

For existing checkpoints with biases (old format):
- The conversion script will renumber layers automatically
- However, **biases will be ignored** during loading
- You **must retrain** your model without biases for correct results

## References

- [PyTorch Compatibility Guide](pytorch_compatibility.md) - Detailed PyTorch implementation guide
- [tiny-cuda-nn FullyFusedMLP source](https://github.com/NVlabs/tiny-cuda-nn) - Original implementation
