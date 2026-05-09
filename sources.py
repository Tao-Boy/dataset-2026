"""
Sources — generate bit streams and read audio files for modulation input.
"""

import wave
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal


def random_symbols(num_bytes, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 256, size=num_bytes, dtype=np.uint8)


def random_bits(num_bits, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.integers(0, 2, size=num_bits, dtype=np.uint8)


def _wav_read_frames(path, max_frames=None):
    """
    Read a WAV file, optionally limiting to max_frames.
    Uses the built-in wave module so we never load the entire file
    if only a portion is needed.
    """
    with wave.open(path, 'r') as wf:
        rate = wf.getframerate()
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        total_frames = wf.getnframes()

        if max_frames is not None:
            frames_to_read = min(total_frames, max_frames)
        else:
            frames_to_read = total_frames

        raw = wf.readframes(frames_to_read)

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    elif sampwidth == 1:
        dtype = np.uint8
    else:
        dtype = np.int16

    data = np.frombuffer(raw, dtype=dtype)

    if nchannels > 1:
        data = data.reshape(-1, nchannels)

    return rate, data, sampwidth


def _pcm_to_float(data, sampwidth):
    """Convert PCM data to float in [-1, 1]."""
    if sampwidth == 2:
        return data.astype(np.float64) / 32768.0
    elif sampwidth == 4:
        return data.astype(np.float64) / 2147483648.0
    elif sampwidth == 1:
        return (data.astype(np.float64) - 128.0) / 128.0
    else:
        data = data.astype(np.float64)
        mx = np.max(np.abs(data))
        return data / mx if mx > 0 else data


def wav_to_float(wav_path, target_rate=None, required_duration=None):
    """
    Read a WAV file and return (audio_float, actual_rate).

    Only reads the portion of the file needed to satisfy
    required_duration, so large WAV files are handled efficiently.

    Parameters
    ----------
    wav_path : str
        Path to WAV file.
    target_rate : int
        If set, resample audio to this sample rate.
    required_duration : float
        If set and the audio is shorter, loop it.
        If the audio is longer, only this much is read (plus margin).

    Returns
    -------
    audio : np.ndarray, float64, 1-D
        Mono audio samples normalized to [-1, 1].
    rate : int
        Final sample rate (original or resampled).
    """
    if required_duration is not None and target_rate is not None:
        # Only read what we need (plus margin for resampling filter edges).
        with wave.open(wav_path, 'r') as wf:
            src_rate = wf.getframerate()

        # Resampling preserves duration, so needed source seconds equals
        # required_duration.  Add 20 % margin for filter tails + 0.5 s pad.
        needed_src_sec = required_duration * 1.2 + 0.5
        max_frames = int(needed_src_sec * src_rate) + 1000

        rate, data, sampwidth = _wav_read_frames(wav_path, max_frames=max_frames)
    elif required_duration is not None:
        # target_rate is None — just read enough for the duration
        with wave.open(wav_path, 'r') as wf:
            src_rate = wf.getframerate()
        max_frames = int(required_duration * 1.2 * src_rate) + 1000
        rate, data, sampwidth = _wav_read_frames(wav_path, max_frames=max_frames)
    else:
        # No duration limit — read the whole file, but cap at 30 seconds
        # to guard against accidentally huge WAV files.
        with wave.open(wav_path, 'r') as wf:
            src_rate = wf.getframerate()
        max_frames = 30 * src_rate
        rate, data, sampwidth = _wav_read_frames(wav_path, max_frames=max_frames)

    # Convert to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Convert to float
    data = _pcm_to_float(data, sampwidth)

    # Resample
    if target_rate is not None and target_rate != rate:
        gcd = np.gcd(rate, target_rate)
        up = target_rate // gcd
        down = rate // gcd
        data = scipy_signal.resample_poly(data.astype(np.float64), up, down)
        rms = np.sqrt(np.mean(data ** 2))
        if rms > 0:
            data = data / rms
        rate = target_rate

    # Loop if audio is too short
    if required_duration is not None:
        required_len = int(required_duration * rate)
        while len(data) < required_len:
            data = np.tile(data, 2)
        data = data[:required_len]

    return data.astype(np.float64), rate


def make_test_wav(path, duration=3.0, sample_rate=44100):
    """
    Create a test WAV file: a simple chord (440 Hz + 554 Hz tones)
    with a speech-like amplitude envelope.
    """
    t = np.arange(0, duration, 1 / sample_rate)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 554 * t)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    audio = audio * envelope
    audio = audio / np.max(np.abs(audio))
    int_audio = (audio * 32767).astype(np.int16)
    wavfile.write(path, sample_rate, int_audio)
    return path
