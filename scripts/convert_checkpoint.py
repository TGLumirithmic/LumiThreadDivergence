#!/usr/bin/env python3
"""
Convert PyTorch checkpoint (.pth) to custom binary format for tiny-cuda-nn loading.

Usage:
    python convert_checkpoint.py input.pth output.bin

Binary format:
    - Magic header: "TCNN" (4 bytes)
    - For each tensor:
        - uint32: name_length
        - char[name_length]: tensor name
        - uint32: num_dimensions
        - uint64[num_dimensions]: shape
        - float32[product(shape)]: data in row-major order
    - End marker: uint32 name_length = 0
"""

import torch
import struct
import sys
import argparse
import numpy as np
from pathlib import Path


def write_tensor(f, name, tensor):
    """Write a single tensor to the binary file."""
    # Convert to CPU and float32 if needed
    data = tensor.detach().cpu().float().numpy()

    # Write name
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('I', len(name_bytes)))
    f.write(name_bytes)

    # Write shape
    f.write(struct.pack('I', len(data.shape)))
    for dim in data.shape:
        f.write(struct.pack('Q', dim))

    # Write data (flatten in row-major order)
    f.write(data.tobytes())

    print(f"  Wrote: {name} - shape {list(data.shape)}")


def next_multiple(val, multiple):
    """Round up to next multiple."""
    return ((val + multiple - 1) // multiple) * multiple


def pad_output_layers(state_dict, verbose=True):
    """
    Pad decoder output layer weight matrices to multiple of 16.

    tiny-cuda-nn requires output dimensions to be multiples of 16 for tensor core operations.
    This function pads the output layer weight matrices from [n, width] to [padded_n, width]
    where padded_n is the next multiple of 16.
    """
    padded_dict = {}

    for name, tensor in state_dict.items():
        new_tensor = tensor

        # Check if this is an output layer for a decoder
        # Pattern: {decoder_name}.{layer_num}.weight where this is the last layer
        if any(decoder in name for decoder in ['visibility_decoder', 'normal_decoder', 'depth_decoder']):
            if 'weight' in name and len(tensor.shape) == 2:
                # Check if this might be an output layer (small first dimension)
                # For decoders: visibility=1, normal=3, depth=1
                output_dim = tensor.shape[0]

                # Only pad if output dimension is small (likely an output layer)
                # and not already a multiple of 16
                if output_dim <= 16 and output_dim % 16 != 0:
                    padded_dim = next_multiple(output_dim, 16)
                    input_dim = tensor.shape[1]

                    if verbose:
                        print(f"  Padding {name}: [{output_dim}, {input_dim}] -> [{padded_dim}, {input_dim}]")

                    # Create padded tensor with zeros
                    padded_tensor = torch.zeros(padded_dim, input_dim, dtype=tensor.dtype)
                    # Copy original data
                    padded_tensor[:output_dim, :] = tensor
                    new_tensor = padded_tensor

        padded_dict[name] = new_tensor

    return padded_dict


def renumber_decoder_layers(state_dict):
    """
    Renumber decoder and direction encoder layers from PyTorch Sequential format (0,2,4,6,8...)
    to sequential numbering (0,1,2,3,4...).

    PyTorch Sequential modules use even indices for layers and odd for activations.
    We convert to sequential numbering for simpler loading in C++.
    """
    renamed_dict = {}

    for name, tensor in state_dict.items():
        new_name = name

        # Check if this is a decoder or direction encoder layer
        if any(decoder in name for decoder in ['visibility_decoder', 'normal_decoder', 'depth_decoder', 'direction_encoder']):
            # Extract the layer number from pattern like "decoder.X.weight" or "direction_encoder.X.weight"
            parts = name.split('.')
            if len(parts) >= 3 and parts[1].isdigit():
                old_layer_num = int(parts[1])
                # Convert from 2*i to i (0,2,4,6,8 -> 0,1,2,3,4)
                if old_layer_num % 2 == 0:
                    new_layer_num = old_layer_num // 2
                    parts[1] = str(new_layer_num)
                    new_name = '.'.join(parts)

        renamed_dict[new_name] = tensor

    return renamed_dict


def convert_checkpoint(input_path, output_path, verbose=True, renumber_layers=True):
    """Convert PyTorch checkpoint to binary format."""

    if verbose:
        print(f"Loading PyTorch checkpoint: {input_path}")

    # Load checkpoint
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False

    # Extract state dict (handle different checkpoint formats)
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Assume the checkpoint is already a state dict
            state_dict = checkpoint
    else:
        print("Unexpected checkpoint format")
        return False

    if verbose:
        print(f"Found {len(state_dict)} tensors in checkpoint")

    # Write binary file
    with open(output_path, 'wb') as f:
        # Write magic header
        f.write(b'TCNN')

        # Write each tensor
        for name, tensor in state_dict.items():
            if 'bias' in name:
                print('WARNING: Skipping bias layer', name)
                continue
            write_tensor(f, name, tensor)

        # Write end marker
        f.write(struct.pack('I', 0))

    if verbose:
        print(f"Successfully converted to: {output_path}")

    return True


def print_checkpoint_info(input_path):
    """Print information about a PyTorch checkpoint without converting."""
    print(f"Checkpoint info: {input_path}")

    checkpoint = torch.load(input_path, map_location='cpu')

    if isinstance(checkpoint, dict):
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")

        # Get state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        print(f"\nModel tensors ({len(state_dict)} total):")
        for name, tensor in state_dict.items():
            print(f"  {name}: {list(tensor.shape)} ({tensor.dtype})")
    else:
        print("Checkpoint is not a dictionary")


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch checkpoints to binary format for tiny-cuda-nn'
    )
    parser.add_argument('input', type=str, help='Input PyTorch checkpoint (.pth)')
    parser.add_argument('output', type=str, nargs='?', help='Output binary file')
    parser.add_argument('--info', action='store_true',
                       help='Print checkpoint info without converting')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    if args.info:
        print_checkpoint_info(input_path)
        return 0

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.bin')

    # Convert
    success = convert_checkpoint(input_path, output_path, verbose=not args.quiet)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
