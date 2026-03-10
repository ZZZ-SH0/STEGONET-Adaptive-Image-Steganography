import struct
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import time


def extract_message_adp(stego_path, embedding_map, threshold_percentile=70):
    """Extract hidden message from stego image using embedding map"""

    stego_img = cv2.imread(stego_path)
    stego_rgb = cv2.cvtColor(stego_img, cv2.COLOR_BGR2RGB)
    H, W, C = stego_rgb.shape

    if embedding_map.shape != (H, W):
        print(f"⚠️  Resizing embedding map from {embedding_map.shape} to ({W}, {H})")
        embedding_map = cv2.resize(embedding_map, (W, H))

    threshold = np.percentile(embedding_map, threshold_percentile)
    mask = embedding_map >= threshold

    y_coords, x_coords = np.where(mask)
    priorities = embedding_map[y_coords, x_coords]
    sorted_idx = np.argsort(priorities)[::-1]

    y_coords = y_coords[sorted_idx]
    x_coords = x_coords[sorted_idx]

    print(f"Extracting from {len(y_coords):,} pixels (threshold: {threshold:.4f})...")

    # -------------------------
    # Extract header (32 bits)
    # -------------------------
    header_bits = []
    bit_idx = 0

    for i in range(len(y_coords)):
        if len(header_bits) >= 32:
            break

        y, x = y_coords[i], x_coords[i]

        for ch in range(3):
            if len(header_bits) >= 32:
                break

            header_bits.append(str(stego_rgb[y, x, ch] & 1))
            bit_idx += 1

    if len(header_bits) < 32:
        raise ValueError("Not enough pixels to extract header")

    msg_len = struct.unpack(
        '>I', int(''.join(header_bits), 2).to_bytes(4, 'big')
    )[0]

    print(f"Decoded message length: {msg_len} bytes")

    total_bits_needed = 32 + msg_len * 8
    total_bits_available = len(y_coords) * 3

    if total_bits_needed > total_bits_available:
        raise ValueError("Not enough embedding capacity")

    print(f"Extracting {msg_len * 8} message bits...")

    # -------------------------
    # Extract message bits
    # -------------------------
    msg_bits = []

    for i in range(len(y_coords)):
        if bit_idx >= total_bits_needed:
            break

        y, x = y_coords[i], x_coords[i]

        for ch in range(3):

            if bit_idx >= total_bits_needed:
                break

            if bit_idx >= 32:
                msg_bits.append(str(stego_rgb[y, x, ch] & 1))

            bit_idx += 1

    print(f"Extracted {len(msg_bits)} message bits")

    # -------------------------
    # Convert bits → bytes
    # -------------------------
    msg_bytes = bytearray()

    for i in range(0, len(msg_bits), 8):

        if i + 8 <= len(msg_bits):
            byte_val = int(''.join(msg_bits[i:i+8]), 2)
            msg_bytes.append(byte_val)

    # -------------------------
    # Try decoding
    # -------------------------
    encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']

    for enc in encodings:
        try:
            message = msg_bytes.decode(enc)
            print(f"✓ Decoded using {enc}")
            return message

        except UnicodeDecodeError:
            continue

    print("⚠️ Decoding failed — returning lossy message")

    return msg_bytes.decode('utf-8', errors='replace')


def decode_and_compare(
        stego_path,
        embedding_map_path,
        original_message_path,
        cover_image_path=None,
        threshold=70):

    print("=" * 70)
    print("DECODING AND VERIFICATION")
    print("=" * 70)

    # -------------------------
    # Load embedding map
    # -------------------------
    embedding_map = np.load(embedding_map_path)

    print(f"Loaded embedding map: {embedding_map_path}")

    # -------------------------
    # Extract message
    # -------------------------
    extracted_message = extract_message_adp(
        stego_path,
        embedding_map,
        threshold
    )

    # -------------------------
    # Load original message
    # -------------------------
    if os.path.exists(original_message_path):

        with open(original_message_path, 'r', encoding='utf-8') as f:
            original_message = f.read()

    else:
        original_message = None
        print("Original message file not found")

    # -------------------------
    # Compare messages
    # -------------------------
    if original_message:

        print("\nComparing messages...\n")

        matches = sum(
            c1 == c2 for c1, c2 in zip(original_message, extracted_message)
        )

        percent = matches / len(original_message) * 100

        print(f"Character Match: {percent:.2f}%")
        print(f"Perfect Match: {original_message == extracted_message}")

    else:

        print("No original message available for comparison")

    return extracted_message


def decode_from_stego(stego_image, cover_image, threshold):
    """
    Decode message from stego image.

    Required files in repository:
    - embedding_map.npy
    - original_message.txt
    """

    embedding_map_file = 'embedding_map.npy'
    original_message_file = 'original_message.txt'

    decoded = decode_and_compare(
        stego_path=stego_image,
        embedding_map_path=embedding_map_file,
        original_message_path=original_message_file,
        cover_image_path=cover_image,
        threshold=threshold
    )

    return decoded


# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------
# Replace the file paths below with your own files.
#
# stego_image:
#     Path to the stego image generated by the embedding step
#
# cover_image:
#     Path to the original cover image used during embedding
#
# Recommended repository structure:
#
# demo/
#   cover_image.png
#   stego_image.png
#
# -----------------------------------------------------------

if __name__ == "__main__":

    start = time.perf_counter()

    decoded_message = decode_from_stego(
        stego_image="demo/stego_image.png",   # TODO: replace with your stego image
        cover_image="demo/cover_image.png",   # TODO: replace with your cover image
        threshold=70
    )

    end = time.perf_counter()

    print(f"Decoding Time: {end - start:.3f} seconds")