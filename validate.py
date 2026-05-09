"""
Post-generation validation — checks the HDF5 file for correctness.
"""

import numpy as np
import h5py


def validate_h5(filepath, mod_names, snr_values, nvecs_per_key, vec_length):
    """
    Validate an HDF5 dataset file.

    Returns True if all checks pass, False otherwise.
    Prints a summary of findings.
    """
    errors = []
    warnings = []

    print(f"\n{'='*60}")
    print(f"Validating: {filepath}")
    print(f"{'='*60}")

    # 1. Can we open the file?
    try:
        f = h5py.File(filepath, "r")
    except Exception as e:
        print(f"FAIL: Cannot open HDF5 file: {e}")
        return False

    # 2. Check datasets exist
    for name in ["X", "modulation", "snr"]:
        if name not in f:
            errors.append(f"Missing dataset: /{name}")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        f.close()
        return False

    X = f["X"]
    mod = f["modulation"]
    snr = f["snr"]

    N = X.shape[0]

    # 3. Check X shape and dtype
    if X.ndim != 3 or X.shape[1] != 2 or X.shape[2] != vec_length:
        errors.append(
            f"/X shape is {X.shape}, expected (N, 2, {vec_length})"
        )
    if X.dtype != np.float32:
        errors.append(f"/X dtype is {X.dtype}, expected float32")

    # 4. Check label lengths
    if mod.shape[0] != N:
        errors.append(f"/modulation length {mod.shape[0]} != N={N}")
    if snr.shape[0] != N:
        errors.append(f"/snr length {snr.shape[0]} != N={N}")

    # 5. Load data (for small datasets; for large ones we chunk)
    X_data = X[:]
    mod_data = mod[:]
    snr_data = snr[:]

    # 6. Check for NaN and Inf
    nan_count = np.sum(np.isnan(X_data))
    inf_count = np.sum(np.isinf(X_data))
    if nan_count > 0:
        errors.append(f"/X contains {nan_count} NaN values")
    if inf_count > 0:
        errors.append(f"/X contains {inf_count} Inf values")

    # 7. Check mod indices are in range
    invalid_mod = ~np.isin(mod_data, range(len(mod_names)))
    if np.any(invalid_mod):
        errors.append(
            f"Found invalid modulation indices: {np.unique(mod_data[invalid_mod])}"
        )

    # 8. Check SNR values are expected
    invalid_snr = ~np.isin(snr_data, snr_values)
    if np.any(invalid_snr):
        warnings.append(
            f"Found unexpected SNR values: {np.unique(snr_data[invalid_snr])}"
        )

    # 9. Count samples per (mod, SNR) combination
    print(f"\n  Total samples: {N}")
    print(f"  Expected shape: (N, 2, {vec_length})")
    print(f"  Actual shape:   {X.shape}")
    print(f"  NaN count: {nan_count}, Inf count: {inf_count}")
    print()

    n_expected = nvecs_per_key
    missing_combos = []
    for mod_idx, name in enumerate(mod_names):
        for s in snr_values:
            count = np.sum((mod_data == mod_idx) & (snr_data == s))
            if count != n_expected:
                missing_combos.append(
                    f"  {name} @ {s:+3d} dB: {count} samples (expected {n_expected})"
                )
            else:
                # Just print first and last few for brevity
                pass

    # 10. Check energy distribution
    energies = np.sqrt(np.sum(X_data ** 2, axis=(1, 2)))  # per-sample L2 norm
    mean_energy = np.mean(energies)
    std_energy = np.std(energies)

    print(f"  Per-sample L2 norm: mean={mean_energy:.4f}, std={std_energy:.4f}")
    print(f"  Energy range: [{np.min(energies):.4f}, {np.max(energies):.4f}]")

    if mean_energy < 0.1 or mean_energy > 10.0:
        warnings.append(f"Mean energy {mean_energy:.4f} seems far from 1.0")

    # 11. Print per-modulation-SNR summary (compact)
    print(f"\n  Samples per (modulation, SNR):")
    print(f"  {'Modulation':<8} {'SNR range':>10} {'Count':>8}  Status")
    print(f"  {'-'*8} {'-'*10} {'-'*8}  {'-'*6}")
    for mod_idx, name in enumerate(mod_names):
        counts = []
        for s in snr_values:
            c = np.sum((mod_data == mod_idx) & (snr_data == s))
            counts.append(c)
        min_c, max_c = min(counts), max(counts)
        expected = nvecs_per_key
        status = "OK" if min_c == max_c == expected else "MISMATCH"
        print(f"  {name:<8} {snr_values[0]:+4d}..{snr_values[-1]:+4d} {min_c:>5d}-{max_c:<5d} {status}")

    # 12. Check metadata
    if "metadata" in f:
        meta = f["metadata"]
        has_names = "modulation_names" in meta
        has_config = "config_json" in meta
        has_notes = "notes" in meta
        has_version = "version_info" in meta
        print(f"\n  Metadata: names={has_names}, config={has_config}, "
              f"notes={has_notes}, version={has_version}")
    else:
        warnings.append("No /metadata group found")

    f.close()

    # Print full mismatch list if there are any
    if missing_combos:
        print(f"\n  Sample count mismatches ({len(missing_combos)} total):")
        for line in missing_combos[:20]:  # show first 20
            print(line)
        if len(missing_combos) > 20:
            print(f"  ... and {len(missing_combos) - 20} more")

    # Print errors and warnings
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    # Final verdict
    if errors:
        print(f"\n  VALIDATION: FAILED ({len(errors)} errors, {len(warnings)} warnings)")
        return False
    elif warnings:
        print(f"\n  VALIDATION: PASSED with {len(warnings)} warnings")
        return True
    else:
        print(f"\n  VALIDATION: PASSED")
        return True
