"""
Channel model: applies AWGN noise, multipath fading, frequency offset,
and sample rate offset using GNU Radio's dynamic_channel_model.

SNR / noise_amp conversion
---------------------------
The GNU Radio dynamic_channel_model's `noise_amp` parameter is the
*standard deviation* of the complex AWGN added to the signal.

For a signal with unit average power (after L2 normalization):
    SNR_dB = 10 * log10(signal_power / noise_power)
    noise_power = noise_std^2   (for complex AWGN, total power across I+Q)

Therefore:
    noise_std = 10 ** (-SNR_dB / 20.0)

The original radioML code used 10**(-snr/10.0), which gives the noise
*variance*, not the standard deviation. This is a known bug — the noise
was ~30% weaker than intended for a given SNR label.

Reference: gnuradio/gr-channels/include/gnuradio/channels/dynamic_channel_model.h
  "noise_amp Specifies the standard deviation of the AWGN process"
"""

import os
import sys
import numpy as np

os.environ.setdefault("GR_VMCIRCBUF_DEFAULT_FACTORY", "vmcircbuf_mmap_shm_open")

try:
    from gnuradio import gr, blocks, channels
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


def snr_to_noise_std(snr_db):
    """
    Convert SNR in dB to noise standard deviation.

    Assumes signal has unit average power (RMS = 1).
    SNR_dB = 10*log10(signal_power / noise_power)
    noise_power = noise_std^2 (variance of complex noise)
    noise_std = 10^(-SNR_dB / 20)

    Parameters
    ----------
    snr_db : float  SNR in dB.

    Returns
    -------
    noise_std : float  AWGN standard deviation.
    """
    return 10.0 ** (-snr_db / 20.0)


def apply_channel(signal, snr_db, sample_rate, rng_seed=None,
                  fd=1.0, k=4.0, ntaps=8,
                  delays=None, mags=None):
    """
    Pass the complex signal through the GNU Radio dynamic channel model.

    Parameters
    ----------
    signal : np.ndarray (complex64)  Input baseband signal.
    snr_db : float                   Target SNR in dB.
    sample_rate : float              Sample rate in Hz.
    rng_seed : int or None           Seed for the noise source.
    fd : float                       Maximum Doppler shift (Hz).
    k : float                        Rician K-factor.
    ntaps : int                      Multipath delay filter taps.
    delays : list or None            Multipath tap delays (default: [0.0, 0.9, 1.7]).
    mags : list or None              Multipath tap magnitudes (default: [1.0, 0.8, 0.3]).

    Returns
    -------
    output : np.ndarray (complex64)  Signal after channel impairments.
    """
    noise_std = snr_to_noise_std(snr_db)

    if rng_seed is None:
        rng_seed = 0

    if delays is None:
        delays = [0.0, 0.9, 1.7]
    if mags is None:
        mags = [1.0, 0.8, 0.3]

    epsilon = 0.01
    freq_offset = 0.5e3

    src = blocks.vector_source_c(signal.tolist(), False)

    chan = channels.dynamic_channel_model(
        sample_rate,          # samp_rate
        0.01,                 # samp_rate_std_dev (fraction)
        freq_offset,          # freq_std_dev (Hz)
        epsilon,              # epsilon (sample rate offset std dev)
        freq_offset,          # max_freq_offset (Hz)
        ntaps,                # ntaps_mpath
        fd,                   # doppler_freq
        k >= 1.0,             # LOS (True if Rician)
        k,                    # Rician K-factor
        delays,               # delays
        mags,                 # mags
        ntaps,                # ntaps for FIR filter
        noise_std,            # noise_amp (std dev!)
        int(rng_seed) ^ 0x1337,  # noise seed
    )

    snk = blocks.vector_sink_c()

    tb = gr.top_block("channel")
    tb.connect(src, chan, snk)
    tb.run()

    return np.array(snk.data(), dtype=np.complex64)
