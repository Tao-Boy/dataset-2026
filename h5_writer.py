"""
HDF5 writer — stores I/Q vectors, labels, and metadata.

Output schema
-------------
/X           float32  [N, 2, VECTOR_LENGTH]  X[:,0,:] = I, X[:,1,:] = Q
/modulation  int32    [N]                     Modulation index (0-based)
/snr         int32    [N]                     SNR in dB
/metadata/
    modulation_names   string array           e.g. ["BPSK", "QPSK", ...]
    config_json        string                 Key parameters from config
    notes              string                 Notes about this dataset
    version_info       string                 Python, NumPy, h5py, GNU Radio versions
"""

import json
import sys
import numpy as np
import h5py


def write_h5(filepath, X, mod_indices, snr_labels, mod_names, config_dict,
             notes, compression="gzip"):
    """
    Write the dataset to an HDF5 file.

    Parameters
    ----------
    filepath : str
        Output .h5 path.
    X : np.ndarray, float32, shape (N, 2, vec_len)
        I/Q data. X[:,0,:] = in-phase, X[:,1,:] = quadrature.
    mod_indices : np.ndarray, int32, shape (N,)
        Modulation type index for each sample.
    snr_labels : np.ndarray, int32 or float32, shape (N,)
        SNR value (dB) for each sample.
    mod_names : list of str
        Modulation name strings, index-aligned with mod_indices.
    config_dict : dict
        Key config parameters to store.
    notes : str
        Dataset description / notes about differences from original.
    compression : str
        h5py compression filter ("gzip" or "lzf").
    """
    with h5py.File(filepath, "w") as f:
        f.create_dataset(
            "X", data=X.astype(np.float32),
            compression=compression,
            chunks=True,
        )
        f.create_dataset(
            "modulation", data=mod_indices.astype(np.int32),
            compression=compression,
        )
        f.create_dataset(
            "snr", data=snr_labels.astype(np.int32),
            compression=compression,
        )

        meta = f.create_group("metadata")

        # Store modulation names as variable-length strings
        dt = h5py.string_dtype()
        meta.create_dataset("modulation_names", data=np.array(mod_names, dtype=dt))

        meta.create_dataset("config_json", data=json.dumps(config_dict, indent=2))

        meta.create_dataset("notes", data=notes)

        # Version info
        from gnuradio import gr as _gr
        version_info = (
            f"Python {sys.version}\n"
            f"NumPy {np.__version__}\n"
            f"SciPy (see requirements)\n"
            f"h5py {h5py.__version__}\n"
            f"GNU Radio {_gr.version()}"
        )
        meta.create_dataset("version_info", data=version_info)

    print(f"Wrote {filepath}")


def read_h5(filepath):
    """
    Read the HDF5 file and return its contents.

    Returns
    -------
    X, mod_indices, snr_labels, metadata : tuple
        metadata is a dict with modulation_names, config_json, notes, version_info.
    """
    with h5py.File(filepath, "r") as f:
        X = f["X"][:]
        mod_indices = f["modulation"][:]
        snr_labels = f["snr"][:]
        metadata = {}
        if "metadata" in f:
            meta = f["metadata"]
            if "modulation_names" in meta:
                metadata["modulation_names"] = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in meta["modulation_names"][:]
                ]
            if "config_json" in meta:
                metadata["config_json"] = (
                    meta["config_json"][()].decode("utf-8")
                    if isinstance(meta["config_json"][()], bytes)
                    else str(meta["config_json"][()])
                )
            if "notes" in meta:
                metadata["notes"] = (
                    meta["notes"][()].decode("utf-8")
                    if isinstance(meta["notes"][()], bytes)
                    else str(meta["notes"][()])
                )
            if "version_info" in meta:
                metadata["version_info"] = (
                    meta["version_info"][()].decode("utf-8")
                    if isinstance(meta["version_info"][()], bytes)
                    else str(meta["version_info"][()])
                )
    return X, mod_indices, snr_labels, metadata
