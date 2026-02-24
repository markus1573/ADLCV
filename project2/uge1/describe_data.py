"""
Metadata description script for the sprites dataset.
Prints structural and storage information about the raw .npy file
without loading pixel statistics or producing visualisations.
"""

import os
import numpy as np


DATA_PATH = "./data/sprites.npy"

# These match the defaults used in SpritesDataset / ddpm_train.py
TRAINING_SUBSET = 40_000
BATCH_SIZE      = 8


def human_bytes(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.2f} TB"


def describe_metadata(path: str) -> None:
    # -- File-level info --------------------------------------------------
    abs_path  = os.path.abspath(path)
    file_size = os.path.getsize(abs_path)

    # -- Array-level info -------------------------------------------------
    data = np.load(path)
    n, h, w, c = data.shape
    channel_label = {1: "Grayscale", 3: "RGB", 4: "RGBA"}.get(c, f"{c}-channel")
    bytes_per_elem = data.dtype.itemsize
    array_size_bytes = data.nbytes

    # -- Training usage ---------------------------------------------------
    n_train        = min(TRAINING_SUBSET, n)
    steps_per_epoch = n_train // BATCH_SIZE

    # -- Print ------------------------------------------------------------
    W = 50
    print("=" * W)
    print("         SPRITES DATASET  –  METADATA")
    print("=" * W)

    print("\n[File]")
    print(f"  Path              : {abs_path}")
    print(f"  Format            : NumPy .npy")
    print(f"  Size on disk      : {human_bytes(file_size)}")

    print("\n[Array]")
    print(f"  Shape             : {data.shape}  (N, H, W, C)")
    print(f"  Total images      : {n:,}")
    print(f"  Image resolution  : {h} x {w} pixels")
    print(f"  Color space       : {channel_label}  ({c} channel{'s' if c > 1 else ''})")
    print(f"  Data type         : {data.dtype}  ({bytes_per_elem} byte{'s' if bytes_per_elem > 1 else ''} per element)")
    print(f"  Pixel value range : [{data.min()} – {data.max()}]")
    print(f"  Memory (loaded)   : {human_bytes(array_size_bytes)}")

    print("\n[Training usage  (ddpm_train.py defaults)]")
    print(f"  Subset used       : {n_train:,} / {n:,} images")
    print(f"  Batch size        : {BATCH_SIZE}")
    print(f"  Steps per epoch   : {steps_per_epoch:,}")
    print(f"  Normalisation     : [0, 255]  →  [-1, 1]  (ToTensor + Normalize(0.5))")

    print("\n" + "=" * W)


if __name__ == "__main__":
    import sys

    output_path = "assets/dataset_metadata.txt"
    os.makedirs("assets", exist_ok=True)

    # Tee output to both stdout and file
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    with open(output_path, "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        describe_metadata(DATA_PATH)
        sys.stdout = sys.__stdout__

    print(f"\nMetadata saved to {output_path}")
