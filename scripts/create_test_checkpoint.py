#!/usr/bin/env python3
"""
Create a simple test checkpoint for validating weight loading.
This creates a minimal checkpoint with the three decoders.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

class TestMultiDecoderNetwork(nn.Module):
    """
    Simple multi-decoder network for testing weight loading.
    Matches the architecture expected by the C++ code.
    """
    def __init__(self, encoding_dim=32):
        super().__init__()

        # Three decoder heads (single layer each)
        self.visibility = nn.Linear(encoding_dim, 1)
        self.normal = nn.Linear(encoding_dim, 3)
        self.depth = nn.Linear(encoding_dim, 1)

    def forward(self, encoded_features):
        """
        Forward pass through all decoders.

        Args:
            encoded_features: [batch, encoding_dim] tensor

        Returns:
            vis: [batch, 1] visibility probability
            norm: [batch, 3] normalized normal vector
            depth: [batch, 1] depth value
        """
        vis = torch.sigmoid(self.visibility(encoded_features))

        norm = self.normal(encoded_features)
        norm = norm / (torch.norm(norm, dim=-1, keepdim=True) + 1e-8)

        depth = self.depth(encoded_features)

        return vis, norm, depth


def create_test_checkpoint(output_path, encoding_dim=32, seed=42):
    """
    Create a test checkpoint with random but reproducible weights.

    Args:
        output_path: Where to save the .pth file
        encoding_dim: Hash encoding output dimension
        seed: Random seed for reproducibility
    """
    print(f"Creating test checkpoint: {output_path}")
    print(f"  Encoding dimension: {encoding_dim}")
    print(f"  Random seed: {seed}")

    # Set seed for reproducibility
    torch.manual_seed(seed)

    # Create model
    model = TestMultiDecoderNetwork(encoding_dim=encoding_dim)

    # Initialize with small random weights
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=0.1)
            elif 'bias' in name:
                nn.init.zeros_(param)

    # Print architecture
    print("\nModel architecture:")
    for name, param in model.named_parameters():
        print(f"  {name}: {list(param.shape)}")

    # Save checkpoint
    checkpoint = {
        'state_dict': model.state_dict(),
        'config': {
            'encoding_dim': encoding_dim,
            'seed': seed
        }
    }

    torch.save(checkpoint, output_path)
    print(f"\nCheckpoint saved to: {output_path}")

    # Print file size
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.2f} MB")

    return model


def test_forward_pass(model, encoding_dim=32):
    """
    Test that the model runs a forward pass correctly.
    """
    print("\nTesting forward pass...")

    # Create random input
    batch_size = 4
    encoded = torch.randn(batch_size, encoding_dim)

    # Run forward pass
    with torch.no_grad():
        vis, norm, depth = model(encoded)

    print(f"  Input shape: {list(encoded.shape)}")
    print(f"  Visibility output: {list(vis.shape)}")
    print(f"  Normal output: {list(norm.shape)}")
    print(f"  Depth output: {list(depth.shape)}")

    # Check shapes
    assert vis.shape == (batch_size, 1), "Visibility shape mismatch"
    assert norm.shape == (batch_size, 3), "Normal shape mismatch"
    assert depth.shape == (batch_size, 1), "Depth shape mismatch"

    # Check visibility is in [0, 1]
    assert (vis >= 0).all() and (vis <= 1).all(), "Visibility not in [0, 1]"

    # Check normals are normalized
    norm_magnitude = torch.norm(norm, dim=-1)
    assert torch.allclose(norm_magnitude, torch.ones_like(norm_magnitude), atol=1e-5), \
        "Normals not unit length"

    print("  ✓ All checks passed!")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Create a test checkpoint for neural network weight loading'
    )
    parser.add_argument(
        'output',
        type=str,
        nargs='?',
        default='data/models/test_model.pth',
        help='Output .pth file path (default: data/models/test_model.pth)'
    )
    parser.add_argument(
        '--encoding-dim',
        type=int,
        default=32,
        help='Encoding dimension (default: 32, matches 16 levels × 2 features)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create checkpoint
    model = create_test_checkpoint(
        args.output,
        encoding_dim=args.encoding_dim,
        seed=args.seed
    )

    # Test it
    test_forward_pass(model, encoding_dim=args.encoding_dim)

    # Print next steps
    print("\n" + "="*60)
    print("Next steps:")
    print(f"  1. Convert to binary format:")
    print(f"     python scripts/convert_checkpoint.py {args.output}")
    print(f"  2. Run test program:")
    print(f"     ./build/bin/test_network {args.output.replace('.pth', '.bin')}")
    print("="*60)


if __name__ == '__main__':
    main()
