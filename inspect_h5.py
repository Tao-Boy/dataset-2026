#!/usr/bin/env python3
"""
Inspect an HDF5 dataset file produced by generate_dataset.py.

Usage:  python inspect_h5.py [path/to/file.h5]

If no path is given, defaults to output/RML2026_modern.h5
"""

import sys
import os
import json
import numpy as np
import h5py


def inspect(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"HDF5 Dataset Inspector")
    print(f"{'='*60}")
    print(f"File: {filepath}")
    print(f"Size: {os.path.getsize(filepath) / 1024**2:.1f} MB")

    with h5py.File(filepath, "r") as f:
        print(f"\n--- Datasets ---")

        # X
        if "X" in f:
            X = f["X"]
            print(f"/X              shape={X.shape}, dtype={X.dtype}")
            print(f"                I = X[:,0,:], Q = X[:,1,:]")
            # Show a snippet
            print(f"                X[0, :, :4] = {X[0, :, :4]}")
        else:
            print("/X              MISSING")

        # modulation
        if "modulation" in f:
            mod = f["modulation"]
            print(f"/modulation     shape={mod.shape}, dtype={mod.dtype}")
            vals = mod[:]
            unique = np.unique(vals)
            print(f"                unique values: {unique}")
        else:
            print("/modulation     MISSING")

        # snr
        if "snr" in f:
            snr = f["snr"]
            print(f"/snr            shape={snr.shape}, dtype={snr.dtype}")
            vals = snr[:]
            unique = np.unique(vals)
            print(f"                unique values: {sorted(unique)}")
        else:
            print("/snr            MISSING")

        # metadata
        if "metadata" in f:
            print(f"\n--- Metadata ---")
            meta = f["metadata"]
            for key in meta:
                val = meta[key][()]
                if isinstance(val, bytes):
                    val = val.decode("utf-8")
                if key == "config_json":
                    val = json.loads(val)
                    val = json.dumps(val, indent=2)
                print(f"  {key}:")
                # Indent multi-line values
                for line in str(val).split("\n"):
                    print(f"    {line}")
        else:
            print(f"\n--- Metadata ---")
            print("  MISSING")

        # Per-modulation-SNR breakdown
        if "X" in f and "modulation" in f and "snr" in f and "metadata" in f:
            print(f"\n--- Per-modulation, per-SNR sample counts ---")
            mod_data = f["modulation"][:]
            snr_data = f["snr"][:]
            mod_names = []
            if "modulation_names" in f["metadata"]:
                mn = f["metadata"]["modulation_names"][:]
                mod_names = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in mn
                ]

            snr_vals = sorted(np.unique(snr_data))
            # Header
            header = f"{'Modulation':<10}"
            for s in snr_vals:
                header += f" {s:>5d}"
            print(header)
            print("-" * len(header))

            for mod_idx in sorted(np.unique(mod_data)):
                name = mod_names[mod_idx] if mod_idx < len(mod_names) else f"mod_{mod_idx}"
                row = f"{name:<10}"
                for s in snr_vals:
                    count = np.sum((mod_data == mod_idx) & (snr_data == s))
                    row += f" {count:>5d}"
                print(row)

        print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/RML2026_modern.h5"
    inspect(path)
