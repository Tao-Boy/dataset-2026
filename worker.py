"""
Per-key worker: generate all vectors for one (modulation, SNR) combination.

Designed to be called from multiprocessing.Pool — each worker process
imports this module and runs `generate_one_key(task_dict)` independently.
"""

import sys
import os
import numpy as np

# Belt-and-suspenders: ensure GNU Radio uses mmap buffers even if the
# environment variable wasn't inherited (e.g. spawned, not forked).
os.environ.setdefault("GR_VMCIRCBUF_DEFAULT_FACTORY", "vmcircbuf_mmap_shm_open")

from gnuradio import gr
try:
    gr.prefs().set_string("vmcircbuf", "vmcircbuf_default_factory",
                          "vmcircbuf_mmap_shm_open")
except Exception:
    pass  # prefs API may vary; env var is the primary mechanism

from modulations import generate_modulated_signal
from channel import apply_channel
from normalization import l2_normalize, has_nan_or_inf


# ---------------------------------------------------------------------------
# Shared helpers (also imported by generate_dataset.py / test_smoke.py)
# ---------------------------------------------------------------------------

def _estimate_samples_needed(vec_length, nvecs, avg_stride_factor=2.0):
    """Estimate minimum output samples needed from the flowgraph.

    With random-offset extraction, we just need enough signal to have
    sufficient distinct start positions.  A moderate safety factor ensures
    each vector captures a different signal region.
    """
    min_samples = int(nvecs * vec_length * avg_stride_factor + 5000)
    return max(min_samples, 50000)


def _extract_vectors(signal, vec_length, nvecs, rng):
    """
    Extract `nvecs` vectors of length `vec_length` from `signal`
    using random start offsets.  Skips the first 500 samples to avoid
    filter/transient artifacts.

    Returns (vectors_complex, enough).
    `enough` is False if the signal was too short.
    """
    min_offset = 500
    max_offset = len(signal) - vec_length
    if max_offset <= min_offset:
        return np.array([], dtype=np.complex64), False

    n_extract = min(nvecs, max_offset - min_offset)
    offsets = rng.integers(min_offset, max_offset + 1, size=n_extract)

    vectors = np.empty((n_extract, vec_length), dtype=np.complex64)
    for i, o in enumerate(offsets):
        vectors[i] = l2_normalize(signal[o:o + vec_length])

    return vectors, len(vectors) >= nvecs


def _complex_to_iq_float(complex_vectors):
    """Convert complex vectors [N, vec_len] → [N, 2, vec_len] float32."""
    N, L = complex_vectors.shape
    out = np.zeros((N, 2, L), dtype=np.float32)
    out[:, 0, :] = complex_vectors.real.astype(np.float32)
    out[:, 1, :] = complex_vectors.imag.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Main worker entry point
# ---------------------------------------------------------------------------

def generate_one_key(task):
    """
    Generate all vectors for a single (modulation, SNR) combination.

    Parameters
    ----------
    task : dict
        Must contain:
          mod_name, snr_db, mod_idx, seed,
          vec_length, nvecs_per_key, sps, ebw,
          audio_rate, sample_rate, gfsk_bt, gfsk_sensitivity, cpfsk_k,
          apply_channel

    Returns
    -------
    (iq, mod_idx, snr_db) or None on failure.
      iq : np.ndarray, float32, shape (nvecs, 2, vec_length)
      mod_idx : int
      snr_db : int
    """
    mod_name = task["mod_name"]
    snr_db = task["snr_db"]
    mod_idx = task["mod_idx"]
    seed = task["seed"]

    vec_length = task["vec_length"]
    nvecs_per_key = task["nvecs_per_key"]
    sps = task["sps"]
    ebw = task["ebw"]
    audio_rate = task["audio_rate"]
    sample_rate = task["sample_rate"]
    gfsk_bt = task["gfsk_bt"]
    gfsk_sensitivity = task["gfsk_sensitivity"]
    cpfsk_k = task["cpfsk_k"]
    apply_ch = task["apply_channel"]
    ch_fd = task.get("channel_fd", 1.0)
    ch_k = task.get("channel_k", 4.0)
    ch_ntaps = task.get("channel_ntaps", 8)

    rng = np.random.default_rng(seed)
    min_output = _estimate_samples_needed(vec_length, nvecs_per_key,
                                           avg_stride_factor=3.0)

    vectors_for_key = []
    attempts = 0
    max_attempts = 5

    while len(vectors_for_key) < nvecs_per_key and attempts < max_attempts:
        attempts += 1

        try:
            signal = generate_modulated_signal(
                mod_name=mod_name,
                sps=sps,
                ebw=ebw,
                audio_rate=audio_rate,
                sample_rate=sample_rate,
                gfsk_bt=gfsk_bt,
                gfsk_sensitivity=gfsk_sensitivity,
                cpfsk_k=cpfsk_k,
                rng=rng,
                min_output_samples=min_output,
            )

            if apply_ch:
                signal = apply_channel(
                    signal, snr_db, sample_rate,
                    rng_seed=rng.integers(0, 2 ** 31),
                    fd=ch_fd, k=ch_k, ntaps=ch_ntaps,
                )

            new_vecs, _ = _extract_vectors(
                signal, vec_length,
                nvecs_per_key - len(vectors_for_key),
                rng,
            )
            if len(new_vecs) > 0:
                vectors_for_key.append(new_vecs)

        except (MemoryError, RuntimeError, SystemError) as e:
            # Buffer allocation failures (shm exhaustion etc.) are
            # recoverable — skip this attempt and retry.
            print(f"  [{mod_name} @ {snr_db:+d} dB] attempt {attempts} "
                  f"failed: {e}", file=sys.stderr, flush=True)
            continue

    if len(vectors_for_key) == 0:
        return None

    vectors_for_key = np.concatenate(vectors_for_key, axis=0)
    vectors_for_key = vectors_for_key[:nvecs_per_key]

    if has_nan_or_inf(vectors_for_key):
        return None

    iq = _complex_to_iq_float(vectors_for_key)
    return iq, mod_idx, snr_db
