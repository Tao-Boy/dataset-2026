"""Tests for HDF5 format read/write."""

import os
import tempfile
import numpy as np

from h5_writer import write_h5, read_h5


def test_write_and_read_small_h5():
    """Write a tiny HDF5 file and verify its contents."""
    N = 50
    vec_len = 128

    X = np.random.randn(N, 2, vec_len).astype(np.float32)
    mod = np.random.randint(0, 3, size=N).astype(np.int32)
    snr = np.random.choice([-10, 0, 10], size=N).astype(np.int32)
    mod_names = ["BPSK", "QPSK", "8PSK"]

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.h5")
        write_h5(path, X, mod, snr, mod_names, {"test": True},
                 "Test notes", compression="gzip")

        # Read back
        X2, mod2, snr2, meta = read_h5(path)

        assert X2.shape == (N, 2, vec_len)
        assert X2.dtype == np.float32
        assert np.array_equal(mod2, mod)
        assert np.array_equal(snr2, snr)
        assert meta["modulation_names"] == mod_names
        assert "Test notes" in meta["notes"]
        assert "test" in meta["config_json"]

        # Verify no NaN/Inf
        assert not np.any(np.isnan(X2))
        assert not np.any(np.isinf(X2))


def test_h5_required_datasets_exist():
    """Verify all required datasets and metadata groups are written."""
    N = 20
    vec_len = 64

    X = np.random.randn(N, 2, vec_len).astype(np.float32)
    mod = np.zeros(N, dtype=np.int32)
    snr = np.zeros(N, dtype=np.int32)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test2.h5")
        write_h5(path, X, mod, snr, ["TEST"], {}, "", compression="gzip")

        import h5py
        with h5py.File(path, "r") as f:
            assert "X" in f
            assert "modulation" in f
            assert "snr" in f
            assert "metadata" in f
            assert "modulation_names" in f["metadata"]
            assert "config_json" in f["metadata"]
            assert "notes" in f["metadata"]
            assert "version_info" in f["metadata"]
