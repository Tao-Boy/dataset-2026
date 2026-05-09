#!/usr/bin/env python3
"""
Dataset-2026: Generate a wireless modulation recognition dataset.

Run:  python generate_dataset.py

All settings are in config.py.  Edit that file to change parameters.
Output is an HDF5 file containing I/Q vectors, modulation labels, and SNR labels.

Uses multiprocessing to parallelise generation across (modulation, SNR) keys.
Set NUM_WORKERS in config.py to control parallelism.
"""

import sys
import os
import time
import json
import multiprocessing as mp
import numpy as np
from collections import OrderedDict

# Ensure we can import sibling modules from the project directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

# Use POSIX shared memory (mmap) for GNU Radio circular buffers instead of
# System V shm, which has low per-process limits and exhausts quickly with
# many concurrent workers ("shmget: No space left on device" → std::bad_alloc).
# Must be set before any GNU Radio import.
os.environ.setdefault("GR_VMCIRCBUF_DEFAULT_FACTORY", "vmcircbuf_mmap_shm_open")

import config
from sources import make_test_wav
from worker import generate_one_key
from h5_writer import write_h5
from validate import validate_h5


# ---------------------------------------------------------------------------
# Source file setup
# ---------------------------------------------------------------------------

def _ensure_sources():
    """Make sure test source files exist."""
    if not os.path.isdir(config.SOURCE_DIR):
        os.makedirs(config.SOURCE_DIR, exist_ok=True)

    text_path = config.TEST_TEXT_FILE
    if not os.path.exists(text_path):
        print(f"Creating test text file: {text_path}")
        with open(text_path, "w") as f:
            f.write(
                "This is a test text file for the Dataset-2026 project.\n"
                "It is used as a data source for digital modulations.\n"
                "The text is converted to bits and then modulated.\n"
                "You can replace this file with your own text.\n"
                "Hello world! This is a wireless modulation dataset.\n" * 50
            )

    wav_path = config.TEST_AUDIO_WAV
    if not os.path.exists(wav_path):
        print(f"Creating test audio file: {wav_path}")
        make_test_wav(wav_path, duration=3.0, sample_rate=44100)

    readme_path = os.path.join(config.SOURCE_DIR, "README.md")
    if not os.path.exists(readme_path):
        with open(readme_path, "w") as f:
            f.write(
                "# Source Files\n\n"
                "This directory holds input files for dataset generation.\n\n"
                "## test_text.txt\n"
                "Used for digital modulation data generation (optional).\n"
                "Replace with any plain text file.\n\n"
                "## test_audio.wav\n"
                "Used for analog modulation (WBFM, AM-DSB, AM-SSB).\n"
                "Requirements:\n"
                "- WAV format (PCM)\n"
                "- Mono (or will be converted to mono)\n"
                "- Sample rate: any (will be resampled automatically)\n"
                "- Bit depth: 16-bit recommended, 32-bit float also supported\n"
                "- Duration: at least 2 seconds recommended\n\n"
                "You can replace this with any speech or music WAV file.\n"
            )


# ---------------------------------------------------------------------------
# Task list builder
# ---------------------------------------------------------------------------

def _build_tasks(rng):
    """
    Build a list of (modulation, SNR) task dicts to send to workers.
    Each dict is self-contained (pickleable) so workers need no config import.
    """
    mod_name_to_idx = {name: i for i, name in enumerate(config.MODULATIONS)}
    tasks = []

    for mod_name in config.MODULATIONS:
        mod_idx = mod_name_to_idx[mod_name]
        for snr_db in config.SNR_VALUES:
            task = {
                "mod_name": mod_name,
                "snr_db": snr_db,
                "mod_idx": mod_idx,
                "seed": int(rng.integers(0, 2 ** 31)),
                "vec_length": config.VECTOR_LENGTH,
                "nvecs_per_key": config.NVECS_PER_KEY,
                "sps": config.SAMPLES_PER_SYMBOL,
                "ebw": config.EXCESS_BW,
                "audio_rate": config.AUDIO_RATE,
                "sample_rate": config.SAMPLE_RATE,
                "gfsk_bt": config.GFSK_BT,
                "gfsk_sensitivity": config.GFSK_SENSITIVITY,
                "cpfsk_k": config.CPFSK_K,
                "apply_channel": config.APPLY_CHANNEL,
                "channel_fd": config.CHANNEL_FD,
                "channel_k": config.CHANNEL_K,
                "channel_ntaps": config.CHANNEL_NTAPS,
            }
            tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    # ---- Num workers ----
    num_workers = config.NUM_WORKERS
    if num_workers <= 0:
        num_workers = os.cpu_count() or 4

    # ---- Print setup ----
    print("=" * 60)
    print("Dataset-2026 Generation")
    print("=" * 60)
    print(f"Output file:     {config.OUTPUT_H5}")
    print(f"Vector length:   {config.VECTOR_LENGTH}")
    print(f"Vectors per key: {config.NVECS_PER_KEY}")
    print(f"Modulations:     {len(config.MODULATIONS)}")
    print(f"SNR values:      {len(config.SNR_VALUES)} "
          f"({config.SNR_VALUES[0]:+d}..{config.SNR_VALUES[-1]:+d} dB)")
    print(f"Sample rate:     {config.SAMPLE_RATE} Hz")
    print(f"Samples/symbol:  {config.SAMPLES_PER_SYMBOL}")
    print(f"Channel enabled: {config.APPLY_CHANNEL}")
    print(f"Random seed:     {config.RANDOM_SEED}")
    print(f"Workers:         {num_workers}")
    print()

    # ---- Setup ----
    os.makedirs(os.path.dirname(config.OUTPUT_H5) or ".", exist_ok=True)
    _ensure_sources()

    rng = np.random.default_rng(config.RANDOM_SEED)

    # ---- Build task list ----
    tasks = _build_tasks(rng)
    total_keys = len(tasks)

    print(f"Total tasks: {total_keys}  "
          f"({len(config.MODULATIONS)} modulations × {len(config.SNR_VALUES)} SNRs)")
    print(f"Launching {num_workers} worker process(es)...\n")

    # ---- Generate (parallel) ----
    X_chunks = []
    mod_chunks = []
    snr_chunks = []
    failures = 0

    if num_workers == 1:
        # Single-process path — easier to debug, shows per-modulation progress
        for i, task in enumerate(tasks):
            mod_name = task["mod_name"]
            snr_db = task["snr_db"]
            result = generate_one_key(task)
            if result is None:
                print(f"  WARNING: no vectors for {mod_name} @ {snr_db:+d} dB",
                      file=sys.stderr)
                failures += 1
            else:
                iq, mod_idx, snr = result
                X_chunks.append(iq)
                mod_chunks.append(np.full(iq.shape[0], mod_idx, dtype=np.int32))
                snr_chunks.append(np.full(iq.shape[0], snr, dtype=np.int32))

            pct = (i + 1) / total_keys * 100
            bar_len = 30
            filled = int(bar_len * (i + 1) / total_keys)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  [{mod_name}] {bar} {pct:.0f}%", end="", flush=True)
        print("\n")

    else:
        # Multiprocessing path
        try:
            from tqdm import tqdm
            _has_tqdm = True
        except ImportError:
            _has_tqdm = False

        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_workers) as pool:
            if _has_tqdm:
                results = tqdm(
                    pool.imap_unordered(generate_one_key, tasks),
                    total=total_keys,
                    desc="  Generating",
                    unit="key",
                )
            else:
                results = pool.imap_unordered(generate_one_key, tasks)

            for result in results:
                if result is None:
                    failures += 1
                else:
                    iq, mod_idx, snr = result
                    X_chunks.append(iq)
                    mod_chunks.append(
                        np.full(iq.shape[0], mod_idx, dtype=np.int32)
                    )
                    snr_chunks.append(
                        np.full(iq.shape[0], snr, dtype=np.int32)
                    )

        if failures:
            print(f"\n  WARNING: {failures} task(s) produced no vectors",
                  file=sys.stderr)

    # ---- Assemble final arrays ----
    if len(X_chunks) == 0:
        print("ERROR: No vectors generated at all.", file=sys.stderr)
        sys.exit(1)

    X_all = np.concatenate(X_chunks, axis=0)
    mod_all = np.concatenate(mod_chunks, axis=0)
    snr_all = np.concatenate(snr_chunks, axis=0)

    print(f"\nTotal I/Q samples: {X_all.shape[0]}")
    print(f"X shape:           {X_all.shape}")
    print(f"X dtype:           {X_all.dtype}")

    # ---- Write HDF5 ----
    notes = (
        "Dataset-2026: Modern re-implementation of RadioML 2016.10A-style dataset.\n"
        "Key differences from original:\n"
        "- Output format: HDF5 instead of pickle.\n"
        "- Python 3 instead of Python 2.\n"
        "- No gr-mapper dependency (uses GNU Radio built-in constellations).\n"
        "- No gr-mediatools dependency (uses scipy.io.wavfile).\n"
        "- No mp3 support (WAV only).\n"
        "- Fixed noise_amp: 10**(-SNR/20) instead of 10**(-SNR/10).\n"
        "- L2 normalization (Euclidean norm) instead of sum(abs(x)).\n"
        "- SNR value is E_s/N_0 in dB, normalized to unit signal power.\n"
        "- Parallel generation via multiprocessing.\n"
    )

    config_dict = OrderedDict([
        ("vector_length", config.VECTOR_LENGTH),
        ("nvecs_per_key", config.NVECS_PER_KEY),
        ("sample_rate", config.SAMPLE_RATE),
        ("samples_per_symbol", config.SAMPLES_PER_SYMBOL),
        ("excess_bw", config.EXCESS_BW),
        ("modulations", config.MODULATIONS),
        ("snr_values", config.SNR_VALUES),
        ("apply_channel", config.APPLY_CHANNEL),
        ("random_seed", config.RANDOM_SEED),
        ("audio_rate", config.AUDIO_RATE),
        ("gfsk_bt", config.GFSK_BT),
        ("gfsk_sensitivity", config.GFSK_SENSITIVITY),
        ("cpfsk_k", config.CPFSK_K),
        ("num_workers", num_workers),
    ])

    write_h5(
        filepath=config.OUTPUT_H5,
        X=X_all,
        mod_indices=mod_all,
        snr_labels=snr_all,
        mod_names=config.MODULATIONS,
        config_dict=config_dict,
        notes=notes,
        compression=config.HDF5_COMPRESSION,
    )

    # ---- Validate ----
    ok = validate_h5(
        config.OUTPUT_H5,
        mod_names=config.MODULATIONS,
        snr_values=config.SNR_VALUES,
        nvecs_per_key=config.NVECS_PER_KEY,
        vec_length=config.VECTOR_LENGTH,
    )

    # ---- Done ----
    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f}s")
    if ok:
        print("Dataset generation completed successfully.")
    else:
        print("Dataset generation completed with validation errors.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
