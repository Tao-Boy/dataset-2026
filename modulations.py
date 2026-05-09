"""
Modulation implementations for the 11 modulation types.

Digital modulations use GNU Radio blocks (generic_mod, gfsk_mod, cpfsk_bc).
Analog modulations use wfm_tx and baseband AM/SSB via Hilbert transform.
"""

import os
import sys
import numpy as np
from contextlib import redirect_stdout

os.environ.setdefault("GR_VMCIRCBUF_DEFAULT_FACTORY", "vmcircbuf_mmap_shm_open")

try:
    from gnuradio import gr, blocks, digital, analog, filter as gr_filter
except ImportError as e:
    sys.exit(
        "GNU Radio Python bindings not found.\n"
        "Install GNU Radio first:  sudo apt install gnuradio\n"
        "Or if installed to a non-standard prefix, set PYTHONPATH.\n"
        f"Original error: {e}"
    )

try:
    gr.prefs().set_string("vmcircbuf", "vmcircbuf_default_factory",
                          "vmcircbuf_mmap_shm_open")
except Exception:
    pass

from sources import wav_to_float, make_test_wav


# ---------------------------------------------------------------------------
# Constellation helpers
# ---------------------------------------------------------------------------

def _constellation_bpsk():
    """BPSK: points at -1 and +1 on the real axis (complex)."""
    return digital.constellation_bpsk()


def _constellation_qpsk():
    """QPSK: points in complex plane (Gray-coded by GNU Radio)."""
    return digital.constellation_qpsk()


def _constellation_8psk():
    """8PSK: 8 points on the unit circle."""
    return digital.constellation_8psk()


def _constellation_pam4():
    """PAM4: 4 real-valued points [-3, -1, +1, +3]."""
    points = [-3.0 + 0j, -1.0 + 0j, 1.0 + 0j, 3.0 + 0j]
    # These create a 1-D constellation
    return digital.constellation_calcdist(points, [], 4, 1)


def _constellation_qam16():
    """16-QAM: 4x4 square constellation."""
    return digital.constellation_16qam()


def _constellation_qam64():
    """64-QAM: 8x8 square constellation.

    GNU Radio 3.10 does not include a pre-built constellation_64qam().
    We build the 8x8 square constellation manually.
    Points are in the range [-7, -5, -3, -1, +1, +3, +5, +7] on each axis.
    """
    levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64)
    points = []
    for i in levels:
        for q in levels:
            points.append(complex(i, q))
    points = np.array(points, dtype=np.complex128)
    # Normalize to unit average power
    points = points / np.sqrt(np.mean(np.abs(points) ** 2))
    return digital.constellation_calcdist(points.tolist(), [], 4, 1)


# Map modulation name → (constellation factory, bits_per_symbol)
_DIGITAL_CONST = {
    "BPSK":  (_constellation_bpsk, 1),
    "QPSK":  (_constellation_qpsk, 2),
    "8PSK":  (_constellation_8psk, 3),
    "PAM4":  (_constellation_pam4, 2),
    "QAM16": (_constellation_qam16, 4),
    "QAM64": (_constellation_qam64, 6),
}

# Analog modulations (no constellation, no bits_per_symbol)
_ANALOG_MODS = {"WBFM", "AM-DSB", "AM-SSB"}

# Digital modulations that use generic_mod
_GENERIC_MOD_MODS = {"BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64"}


# ---------------------------------------------------------------------------
# Flowgraph construction helpers
# ---------------------------------------------------------------------------

def _make_generic_mod_flowgraph(const_mod, bits_per_sym, num_symbols, sps, ebw, rng):
    """
    Build a flowgraph for digital modulations using digital.generic_mod.

    digital.generic_mod expects *packed bytes* on input: each byte holds
    8 bits, and the modulator internally unpacks them (MSB first) into
    groups of bits_per_sym to form constellation symbol indices.

    So we generate random bits, pack them into bytes (MSB first), and
    feed the packed bytes to the modulator.

    Flow: [vector_source(packed bytes)] → [generic_mod] → [vector_sink(complex)]

    Returns (top_block, vector_sink).
    """
    total_bits = num_symbols * bits_per_sym
    bits = rng.integers(0, 2, size=total_bits, dtype=np.uint8)

    # Pad to a multiple of 8 so np.packbits works cleanly
    pad = (8 - (total_bits % 8)) % 8
    if pad > 0:
        bits = np.pad(bits, (0, pad), constant_values=0)

    # np.packbits packs MSB first, matching GR_MSB_FIRST in generic_mod
    packed_bytes = np.packbits(bits)

    src = blocks.vector_source_b(packed_bytes.tolist(), False)
    mod = digital.generic_mod(
        constellation=const_mod,
        differential=False,
        samples_per_symbol=sps,
        pre_diff_code=True,
        excess_bw=ebw,
    )
    snk = blocks.vector_sink_c()

    tb = gr.top_block("digital_mod")
    tb.connect(src, mod, snk)
    return tb, snk


def _make_gfsk_flowgraph(num_bits, sps, bt, sensitivity, rng):
    """
    Build a flowgraph for GFSK modulation.

    Flow: [vector_source(packed bytes)] → [gfsk_mod] → [vector_sink(complex)]

    digital.gfsk_mod expects *packed bytes* on input (8 bits per byte).
    It internally unpacks them into individual bits for modulation.
    """
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    # Pad to a multiple of 8 and pack bits into bytes (MSB first)
    pad = (8 - (num_bits % 8)) % 8
    if pad > 0:
        bits = np.pad(bits, (0, pad), constant_values=0)
    packed_bytes = np.packbits(bits)

    src = blocks.vector_source_b(packed_bytes.tolist(), False)
    mod = digital.gfsk_mod(
        samples_per_symbol=sps,
        sensitivity=sensitivity,
        bt=bt,
    )
    snk = blocks.vector_sink_c()

    tb = gr.top_block("gfsk_mod")
    tb.connect(src, mod, snk)
    return tb, snk


def _make_cpfsk_flowgraph(num_bits, sps, k, rng):
    """
    Build a flowgraph for CPFSK modulation.

    Flow: [vector_source(unpacked bits)] → [cpfsk_bc] → [vector_sink(complex)]

    analog.cpfsk_bc takes unpacked bits (each byte 0/1)
    and produces complex output at `sps` samples per bit.
    """
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)

    src = blocks.vector_source_b(bits.tolist(), False)
    mod = analog.cpfsk_bc(k, 1.0, sps)
    snk = blocks.vector_sink_c()

    tb = gr.top_block("cpfsk_mod")
    tb.connect(src, mod, snk)
    return tb, snk


def _make_wbfm_flowgraph(audio, audio_rate, quad_rate):
    """
    Build a flowgraph for WBFM modulation.

    Flow: [vector_source(float audio)] → [wfm_tx] → [vector_sink(complex)]

    audio_rate must divide quad_rate (e.g., 40000 and 200000).
    The wfm_tx constructor prints a debug line; we suppress it.
    """
    src = blocks.vector_source_f(audio.astype(np.float32).tolist(), False)
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
        mod = analog.wfm_tx(audio_rate=float(audio_rate), quad_rate=float(quad_rate))
    snk = blocks.vector_sink_c()

    tb = gr.top_block("wbfm_mod")
    tb.connect(src, mod, snk)
    return tb, snk


def _make_am_dsb_flowgraph(audio, sample_rate):
    """
    Build a flowgraph for AM-DSB modulation at baseband.

    Flow: [vector_source(float)] → [float_to_complex] → [add_const_cc(1.0)]
          → [vector_sink(complex)]

    Output is baseband AM:  (audio + 1.0) + j*0
    The carrier is at DC (0 Hz).
    """
    src = blocks.vector_source_f(audio.astype(np.float32).tolist(), False)
    cnv = blocks.float_to_complex()
    dc_add = blocks.add_const_cc(1.0)
    snk = blocks.vector_sink_c()

    tb = gr.top_block("am_dsb_mod")
    tb.connect(src, cnv, dc_add, snk)
    return tb, snk


def _make_am_ssb_flowgraph(audio, sample_rate):
    """
    Build a flowgraph for AM-SSB (upper sideband) at baseband.

    Flow: [vector_source(float)] → [add_const_ff(1.0)] → [hilbert_fc]
          → [vector_sink(complex)]

    hilbert_fc produces the analytic signal (positive frequencies only),
    which gives USB when the input has a DC component (carrier at 0 Hz).
    """
    src = blocks.vector_source_f(audio.astype(np.float32).tolist(), False)
    dc_add = blocks.add_const_ff(1.0)
    hilbert = gr_filter.hilbert_fc(401)
    snk = blocks.vector_sink_c()

    tb = gr.top_block("am_ssb_mod")
    tb.connect(src, dc_add, hilbert, snk)
    return tb, snk


# ---------------------------------------------------------------------------
# Analog modulation audio helper
# ---------------------------------------------------------------------------

def _load_analog_audio(target_rate, min_output_samples, sample_rate,
                       duration_multiplier=2.0):
    """Read the test WAV file at target_rate with enough duration."""
    wav_path = "source/test_audio.wav"
    if not os.path.exists(wav_path):
        make_test_wav(wav_path)
    required_duration = max(min_output_samples / sample_rate * duration_multiplier, 2.0)
    audio, _ = wav_to_float(wav_path, target_rate=target_rate,
                            required_duration=required_duration)
    return audio


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_modulated_signal(mod_name, sps, ebw, audio_rate, sample_rate,
                              gfsk_bt, gfsk_sensitivity, cpfsk_k, rng,
                              min_output_samples):
    """
    Generate a complex baseband signal for one modulation type.

    Parameters
    ----------
    mod_name : str
        One of: BPSK, QPSK, 8PSK, PAM4, QAM16, QAM64, GFSK, CPFSK,
                WBFM, AM-DSB, AM-SSB
    sps : int
        Samples per symbol (digital only).
    ebw : float
        Excess bandwidth for RRC shaping (digital only).
    audio_rate : int
        Audio sample rate (analog only).
    sample_rate : int
        Target output sample rate.
    gfsk_bt : float
        BT product for GFSK.
    gfsk_sensitivity : float
        FM modulator sensitivity for GFSK.
    cpfsk_k : float
        Modulation index for CPFSK.
    rng : np.random.Generator
        Random number generator.
    min_output_samples : int
        Generate at least this many output samples.

    Returns
    -------
    signal : np.ndarray, dtype=np.complex64
        Complex baseband signal.
    """
    if mod_name in _GENERIC_MOD_MODS:
        const_factory, bits_per_sym = _DIGITAL_CONST[mod_name]
        constellation = const_factory()
        # Estimate how many symbols we need: each symbol produces ~sps output samples
        num_symbols = max(min_output_samples // sps + 1000, 1000)
        tb, snk = _make_generic_mod_flowgraph(
            constellation, bits_per_sym, num_symbols, sps, ebw, rng
        )

    elif mod_name == "GFSK":
        num_bits = max(min_output_samples // sps + 1000, 1000)
        tb, snk = _make_gfsk_flowgraph(num_bits, sps, gfsk_bt,
                                       gfsk_sensitivity, rng)

    elif mod_name == "CPFSK":
        num_bits = max(min_output_samples // sps + 1000, 1000)
        tb, snk = _make_cpfsk_flowgraph(num_bits, sps, cpfsk_k, rng)

    elif mod_name == "WBFM":
        audio = _load_analog_audio(audio_rate, min_output_samples, sample_rate,
                                   duration_multiplier=3.0)
        tb, snk = _make_wbfm_flowgraph(audio, audio_rate, sample_rate)

    elif mod_name == "AM-DSB":
        audio = _load_analog_audio(sample_rate, min_output_samples, sample_rate,
                                   duration_multiplier=2.0)
        tb, snk = _make_am_dsb_flowgraph(audio, sample_rate)

    elif mod_name == "AM-SSB":
        audio = _load_analog_audio(sample_rate, min_output_samples, sample_rate,
                                   duration_multiplier=2.0)
        tb, snk = _make_am_ssb_flowgraph(audio, sample_rate)

    else:
        raise ValueError(f"Unknown modulation: {mod_name}")

    tb.run()

    return np.array(snk.data(), dtype=np.complex64)
