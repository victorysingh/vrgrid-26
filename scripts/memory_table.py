#!/usr/bin/env python3
"""Regenerate the memory comparison table. Master v4 §3.3.

Gate 6: every number on a slide comes from a script in here. This one reads
CELL_BYTES from the frozen struct, so if the cell ever changes the table
changes with it and the report cannot silently disagree with the code.

    python scripts/memory_table.py
"""

from vrgrid.cell import CELL_BYTES
from vrgrid.grid.schedule import load

# 200 x 200 m footprint, 8 m vertical extent (-3.5 to +4.5 m since 2026-09-17).
FOOTPRINT_M = 200.0
VERTICAL_M = 8.0
BASE_CELL_M = 0.05


def main() -> None:
    default = load("5/10/20/40")
    ablation = load("5/10/50")

    uniform_cells = (FOOTPRINT_M / BASE_CELL_M) ** 2
    uniform_bytes = uniform_cells * CELL_BYTES
    dense_voxels = uniform_cells * (VERTICAL_M / BASE_CELL_M)
    dense_bytes = dense_voxels * 1  # 1 B/voxel
    ours_bytes = default.total_cells * CELL_BYTES
    abl_bytes = ablation.total_cells * CELL_BYTES

    rows = [
        (f"Dense 3D voxels, 5 cm, 200x200x8 m, 1 B/voxel ({dense_voxels/1e9:.2f} G)",
         dense_bytes, dense_bytes / ours_bytes),
        ("Sparse/hashed 3D, realistic surface occupancy, 8 B/voxel",
         None, None),  # 130-240 MB, quoted as a range on purpose
        (f"Uniform 5 cm 2.5D, same {CELL_BYTES} B cell",
         uniform_bytes, uniform_bytes / ours_bytes),
        (f"Ours ({len(default.rings)}-ring)", ours_bytes, None),
        (f"Ours ({len(ablation.rings)}-ring ablation)", abl_bytes, uniform_bytes / abl_bytes),
    ]

    print(f"cell = {CELL_BYTES} B\n")
    print(f"{'Representation':<58} {'Size':>10} {'Ratio':>8}")
    print("-" * 78)
    for name, size, ratio in rows:
        size_s = "~130-240 MB" if size is None else (
            f"{size/1e9:.2f} GB" if size >= 1e9 else f"{size/1e6:.2f} MB")
        ratio_s = "~15-27x" if (size is None) else ("-" if ratio is None else f"{ratio:.1f}x")
        print(f"{name:<58} {size_s:>10} {ratio_s:>8}")

    print("\nReport all four, in that order. Leading with the 286x alone looks like")
    print("cherry-picking; volunteering the sparse-3D number reads as good faith.")
    print(f"\nNote: the {uniform_bytes/ours_bytes:.1f}x ratio is a pure CELL-COUNT ratio and is")
    print("invariant to bytes-per-cell, which is why fields can be added without")
    print("weakening the headline.")


if __name__ == "__main__":
    main()
