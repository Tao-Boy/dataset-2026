"""Smoke test: run the full generation pipeline with tiny parameters."""

import os
import tempfile
import numpy as np


def test_smoke_generation():
    """
    Run a minimal generation to verify the pipeline works end-to-end.
    Uses a small number of vectors, few modulations, few SNRs.
    """
    import config
    from worker import (
        _extract_vectors, _complex_to_iq_float, _estimate_samples_needed,
    )
    from modulations import generate_modulated_signal
    from channel import apply_channel
    from normalization import has_nan_or_inf
    from h5_writer import write_h5

    rng = np.random.default_rng(42)

    # Test parameters (much smaller than real run)
    test_mods = ["BPSK", "QPSK", "QAM16", "GFSK", "WBFM"]
    test_snrs = [0, 10]
    test_nvecs = 5
    vec_len = 128
    sps = 8

    min_output = _estimate_samples_needed(vec_len, test_nvecs)

    X_chunks = []
    mod_chunks = []
    snr_chunks = []

    for mod_idx, mod_name in enumerate(test_mods):
        for snr_db in test_snrs:
            vectors_for_key = []
            attempts = 0

            while len(vectors_for_key) < test_nvecs and attempts < 3:
                attempts += 1
                signal = generate_modulated_signal(
                    mod_name=mod_name,
                    sps=sps,
                    ebw=0.35,
                    audio_rate=40000,
                    sample_rate=200000,
                    gfsk_bt=0.35,
                    gfsk_sensitivity=0.1,
                    cpfsk_k=0.5,
                    rng=rng,
                    min_output_samples=min_output,
                )

                signal = apply_channel(
                    signal, snr_db, 200000,
                    rng_seed=rng.integers(0, 2**31),
                )

                new_vecs, _ = _extract_vectors(
                    signal, vec_len, test_nvecs - len(vectors_for_key), rng
                )
                if len(new_vecs) > 0:
                    vectors_for_key.append(new_vecs)

            if len(vectors_for_key) == 0:
                continue

            vectors_for_key = np.concatenate(vectors_for_key, axis=0)[:test_nvecs]
            iq = _complex_to_iq_float(vectors_for_key)
            X_chunks.append(iq)
            mod_chunks.append(np.full(len(iq), mod_idx, dtype=np.int32))
            snr_chunks.append(np.full(len(iq), snr_db, dtype=np.int32))

    if len(X_chunks) == 0:
        raise RuntimeError("No vectors generated at all")

    X_all = np.concatenate(X_chunks, axis=0)
    mod_all = np.concatenate(mod_chunks, axis=0)
    snr_all = np.concatenate(snr_chunks, axis=0)

    # Basic checks
    assert X_all.shape[0] > 0, "No samples generated"
    assert X_all.shape[1] == 2, "Missing I/Q dimension"
    assert X_all.shape[2] == vec_len, "Wrong vector length"
    assert X_all.dtype == np.float32
    assert not has_nan_or_inf(X_all), "NaN/Inf in output"
    assert len(mod_all) == len(X_all)
    assert len(snr_all) == len(X_all)

    # Write to temp file
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "smoke_test.h5")
        write_h5(path, X_all, mod_all, snr_all, test_mods,
                 {"test": True}, "Smoke test", compression="gzip")
        assert os.path.exists(path)

    print(f"Smoke test passed: {X_all.shape[0]} samples generated")
