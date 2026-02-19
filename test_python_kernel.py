import logging
import matplotlib.pyplot as plt
import numpy as np
import struct
import torch

from pathlib import Path

logger = logging.getLogger(__name__)

def read_samples_from_binary(
    filepath: Path
):
    """
    Read samples from the binary file without performing any comparisons.
    Returns a list of dicts, each containing the decoded fields.
    """

    # Compute header size exactly as writer uses it
    header_size = (
        4 +    # magic
        4 +    # version
        4 +    # num_samples
        1 +    # has_visibility
        1 +    # has_depth
        1 +    # has_normal
        1 +    # has_kd
        1 +    # kd_uses_dfl
        1 +    # has_hash_grid
        1 +    # has_direction_encodings
        1 +    # padding
        4 +    # dfl_num_edges
        4 +    # hash_grid_dim
        256    # mesh_name
    )

    samples = []

    with open(filepath, "rb") as f:
        magic = struct.unpack('I', f.read(4))[0]
        if magic != 0x4E525450:
            logger.error(f"❌ INVALID MAGIC NUMBER: 0x{magic:08X} (expected 0x4E525450)")
            return
        logger.info(f"✓ Magic number valid: 0x{magic:08X}")

        version = struct.unpack('I', f.read(4))[0]
        logger.info(f"✓ Version: {version}")

        num_samples = struct.unpack('I', f.read(4))[0]
        logger.info(f"✓ Number of samples: {num_samples:,}")

        vis_flag = struct.unpack('B', f.read(1))[0]
        depth_flag = struct.unpack('B', f.read(1))[0]
        normal_flag = struct.unpack('B', f.read(1))[0]
        kd_flag = struct.unpack('B', f.read(1))[0]
        dfl_flag = struct.unpack('B', f.read(1))[0]
        hash_grid_flag = struct.unpack('B', f.read(1))[0]
        dir_encodings_flag = struct.unpack('B', f.read(1))[0]
        logger.info(f"✓ Flags: vis={vis_flag}, depth={depth_flag}, normal={normal_flag}, kd={kd_flag}, dfl={dfl_flag}, hash_grid={hash_grid_flag} dir_encodings={dir_encodings_flag}")

        # Read header padding
        _ = f.read(1)

        dfl_edges = struct.unpack('I', f.read(4))[0]
        logger.info(f"✓ DFL edges: {dfl_edges}")

        hash_grid_dimension = struct.unpack('I', f.read(4))[0]
        logger.info(f"✓ Hash grid dimension: {hash_grid_dimension}")

        mesh_name_bytes = f.read(256)
        mesh_name = mesh_name_bytes.split(b'\x00')[0].decode('utf-8')
        logger.info(f"✓ Mesh name: '{mesh_name}'")

        header_size = f.tell()
        logger.info(f"✓ Header size: {header_size} bytes")

        for i in range(num_samples):
            sample = {}

            # Origin (3 floats)
            origin_bytes = f.read(12)
            sample["origin"] = struct.unpack("fff", origin_bytes)

            # Direction (3 floats)
            direction_bytes = f.read(12)
            sample["direction"] = struct.unpack("fff", direction_bytes)

            # Grid indices (3 uint32)
            grid_bytes = f.read(12)
            sample["grid_indices"] = struct.unpack("III", grid_bytes)

            # Face ID
            face_bytes = f.read(4)
            sample["face_id"] = struct.unpack("I", face_bytes)[0]

            # Hemisphere
            theta = struct.unpack("B", f.read(1))[0]
            phi   = struct.unpack("B", f.read(1))[0]
            sample["hemisphere"] = (theta, phi)

            # Hit (byte)
            hit = struct.unpack("B", f.read(1))[0]
            sample["hit"] = hit

            # Padding
            sample["padding"] = struct.unpack("B", f.read(1))[0]

            # Optional feature blocks
            if vis_flag:
                sample["visibility"] = struct.unpack("f", f.read(4))[0]

            if depth_flag:
                sample["depth"] = struct.unpack("f", f.read(4))[0]

            if normal_flag:
                normal_bytes = f.read(12)
                sample["normal"] = struct.unpack("fff", normal_bytes)

            if kd_flag:
                if dfl_flag:
                    num_logits = 3 * dfl_edges
                    kd_data = struct.unpack(f"{num_logits}f", f.read(4 * num_logits))
                    sample["kd_logits"] = kd_data
                else:
                    sample["kd"] = struct.unpack("fff", f.read(12))

            if hash_grid_flag:
                hash_bytes = f.read(hash_grid_dimension * 4)
                sample["hash_grid"] = struct.unpack(f"{hash_grid_dimension}f", hash_bytes)

            if dir_encodings_flag:
                dir_bytes = f.read(16 * 4)
                sample["direction_encodings"] = struct.unpack("16f", dir_bytes)

            samples.append(sample)

    return samples

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_filepath = Path("data/test/predictions.bin")
    samples = read_samples_from_binary(test_filepath)
    logger.info(f"Read {len(samples)} samples from {test_filepath}")
    print(samples[0])

    model_state_dict = torch.load('data/models/test_ngp.pth', map_location='cpu')['model_state_dict']

    vis_decoder_params = model_state_dict['visibility_decoder.params']

    input_dim = 48
    output_dim = 16
    hidden_dim = 32
    num_layers = 5

    layer_weights = []
    # Slice flat params array into matrices
    offset = 0
    for layer_idx in range(num_layers):
        if layer_idx == 0:
            in_dim = input_dim
        else:
            in_dim = hidden_dim
        if layer_idx == num_layers - 1:
            out_dim = output_dim
        else:
            out_dim = hidden_dim

        weight_size = in_dim * out_dim

        weight_flat = vis_decoder_params[offset:offset + weight_size].numpy()
        offset += weight_size

        weight_matrix = weight_flat.reshape(out_dim, in_dim)

        layer_weights.append(weight_matrix)

    vis_pred_logits = np.zeros(len(samples))
    vis_gt = np.zeros(len(samples))

    for idx, sample in enumerate(samples):
        hash_encoding = sample['hash_grid']
        dir_encoding = sample['direction_encodings']

        vis_gt[idx] = sample['visibility']

        input_vector = np.concatenate([
            np.array(hash_encoding),
            np.array(dir_encoding)
        ])

        for layer_idx in range(num_layers):
            weight_matrix = layer_weights[layer_idx]
            input_vector = weight_matrix @ input_vector
            if layer_idx < num_layers - 1:
                input_vector = np.maximum(input_vector, 0)

        vis_pred_logits[idx] = input_vector[0]

        if (idx + 1) % 10000 == 0:
            logger.info(f"Processed {idx + 1} / {len(samples)} samples")
    
    error = np.abs(vis_pred_logits - vis_gt)
    mean_error = np.mean(error)
    max_error = np.max(error)
    logger.info(f"Visibility prediction mean absolute error: {mean_error:.6f}")
    logger.info(f"Visibility prediction max absolute error: {max_error:.6f}")

    plt.hist(vis_pred_logits, bins=100)
    plt.savefig('vis_pred_hist.png')

    plt.close()
    vis_probs = 1.0 / (1.0 + np.exp(-vis_pred_logits))
    plt.hist(vis_probs, bins=100, log=True)
    plt.savefig('vis_pred_probs_hist.png')

    vis_pred = vis_probs >= 0.5
    vis_gt_binary = vis_gt >= 0.0
    accuracy = np.mean(vis_pred == vis_gt_binary)
    logger.info(f"Visibility prediction accuracy (threshold=0.5): {accuracy * 100:.2f}%")