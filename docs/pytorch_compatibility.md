# PyTorch Model Compatibility with tiny-cuda-nn

## FullyFusedMLP Bias Handling

**Important:** tiny-cuda-nn's `FullyFusedMLP` does NOT support biases!

### Background

The FullyFusedMLP network in tiny-cuda-nn is optimized for performance and does not include bias parameters. To achieve equivalent functionality in PyTorch for training, you need to implement a workaround.

### Solution: Input Padding

To simulate biases in the first layer, pad the input with a constant value (typically 1.0) and include an extra weight column that acts as the bias term.

#### PyTorch Implementation

```python
import torch
import torch.nn as nn

class BiasFreeMLP(nn.Module):
    """
    MLP without bias parameters, compatible with tiny-cuda-nn FullyFusedMLP.
    Uses input padding to simulate first-layer bias.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_hidden_layers=4):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        # First layer: input_dim+1 (padded) -> hidden_dim
        # The +1 accounts for the padded constant that simulates bias
        self.first_layer = nn.Linear(input_dim + 1, hidden_dim, bias=False)

        # Hidden layers: hidden_dim -> hidden_dim (NO bias)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(n_hidden_layers - 1)
        ])

        # Output layer: hidden_dim -> output_dim (NO bias)
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)

        self.activation = nn.ReLU()

    def forward(self, x):
        # Pad input with constant 1.0 to simulate bias in first layer
        batch_size = x.shape[0]
        ones = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        x = torch.cat([x, ones], dim=1)  # [B, input_dim] -> [B, input_dim+1]

        # First layer with padded input
        x = self.activation(self.first_layer(x))

        # Hidden layers (no bias)
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        # Output layer
        x = self.output_layer(x)
        return x
```

#### Weight Dimensions

For a network with:
- Input dimension: 48 (e.g., 32 from position encoding + 16 from direction encoding)
- Hidden dimension: 32
- Output dimension: 1
- Hidden layers: 4

The weight shapes will be:
- `first_layer.weight`: `[32, 49]` (49 = 48 input + 1 padded constant)
- `hidden_layers[0].weight`: `[32, 32]`
- `hidden_layers[1].weight`: `[32, 32]`
- `hidden_layers[2].weight`: `[32, 32]`
- `output_layer.weight`: `[1, 32]`

**Total parameters:** 32×49 + 3×(32×32) + 1×32 = 1,568 + 3,072 + 32 = **4,672**

Note: This matches tiny-cuda-nn's parameter count when `n_hidden_layers=3` (3 hidden layers + 1 output layer).

### Checkpoint Naming Convention

Use this naming pattern for your PyTorch checkpoint (the conversion script will handle renumbering):

```python
# Save decoder weights with sequential or PyTorch Sequential naming
# Sequential numbering (recommended):
state_dict = {
    'visibility_decoder.0.weight': visibility_mlp.first_layer.weight,
    'visibility_decoder.1.weight': visibility_mlp.hidden_layers[0].weight,
    'visibility_decoder.2.weight': visibility_mlp.hidden_layers[1].weight,
    'visibility_decoder.3.weight': visibility_mlp.hidden_layers[2].weight,
    'visibility_decoder.4.weight': visibility_mlp.output_layer.weight,
    # Similar for normal_decoder and depth_decoder
}

# OR PyTorch Sequential format (will be converted):
state_dict = {
    'visibility_decoder.0.weight': visibility_mlp.first_layer.weight,
    'visibility_decoder.2.weight': visibility_mlp.hidden_layers[0].weight,
    'visibility_decoder.4.weight': visibility_mlp.hidden_layers[1].weight,
    'visibility_decoder.6.weight': visibility_mlp.hidden_layers[2].weight,
    'visibility_decoder.8.weight': visibility_mlp.output_layer.weight,
}
```

The conversion script automatically renumbers from PyTorch Sequential format (0, 2, 4, 6...) to sequential format (0, 1, 2, 3...).

### Important Notes

1. **Only the first layer gets implicit bias** via input padding
2. **Hidden and output layers have NO bias** - pure matrix multiplications
3. **This is by design** - FullyFusedMLP is optimized for speed, not flexibility
4. **Train accordingly** - your PyTorch model must use `bias=False` for all layers except handle first layer via input padding

### Testing Compatibility

To verify your PyTorch model is compatible:

```python
# Check that no bias parameters exist
for name, param in model.named_parameters():
    assert 'bias' not in name, f"Found bias parameter: {name}"

# Verify weight dimensions
first_layer_weight = model.first_layer.weight
assert first_layer_weight.shape[1] == input_dim + 1, "First layer must account for padded input"
```
