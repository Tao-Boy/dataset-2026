"""
Dataset-2026 Configuration
==========================
Edit the values in this file to control dataset generation.
Then run: python generate_dataset.py
"""

# ---- Output ----
# Path to the output HDF5 file.
OUTPUT_H5 = "output/RML2026_modern.h5"

# ---- Source files ----
SOURCE_DIR = "source"
TEST_TEXT_FILE = "source/test_text.txt"
TEST_AUDIO_WAV = "source/test_audio.wav"

# ---- Random seed (for reproducibility) ----
RANDOM_SEED = 0x1337

# ---- Signal parameters ----
# Number of complex samples per output vector.
VECTOR_LENGTH = 128

# How many vectors to generate for each (modulation, SNR) combination.
NVECS_PER_KEY = 1000

# Target complex sample rate (samples/second).
SAMPLE_RATE = 200000

# Number of samples per symbol for digital modulations.
# Higher = smoother pulse shape but more samples per symbol.
SAMPLES_PER_SYMBOL = 8

# Excess bandwidth (roll-off factor) for RRC pulse shaping filter.
EXCESS_BW = 0.35

# ---- Modulation list ----
# Supported: BPSK, QPSK, 8PSK, PAM4, QAM16, QAM64, GFSK, CPFSK,
#            WBFM, AM-DSB, AM-SSB
MODULATIONS = [
    "BPSK",
    "QPSK",
    "8PSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "GFSK",
    "CPFSK",
    "WBFM",
    "AM-DSB",
    "AM-SSB",
]

# ---- SNR settings ----
# SNR values in dB. Positive = signal stronger than noise.
# Range covers both noisy and clean conditions.
SNR_VALUES = list(range(-20, 20, 2))

# Each SNR listed above is E_s / N_0 in dB.
# The channel model's noise_amp is the AWGN standard deviation.
# Conversion:  noise_std = 10**(-SNR_dB / 20.0)
# assuming the signal has unit average power (L2-normalized).

# ---- Channel settings ----
# Whether to apply channel impairments (AWGN noise, multipath fading,
# frequency offset, sample rate offset).
APPLY_CHANNEL = True

# Maximum Doppler shift for fading (Hz).  0 = no fading, static channel.
CHANNEL_FD = 1.0

# Rician factor K (ratio of LOS power to scattered power).
# Large K = mostly LOS, small K = mostly Rayleigh.
CHANNEL_K = 4.0

# Number of taps for the multipath delay filter.
CHANNEL_NTAPS = 8

# ---- Analog modulation audio settings ----
# Audio sample rate (samples/second). WAV input will be resampled to this.
AUDIO_RATE = 40000

# ---- HDF5 settings ----
# Compression filter: "gzip" (better compression) or "lzf" (faster).
HDF5_COMPRESSION = "gzip"

# ---- GFSK / CPFSK settings ----
# BT product for GFSK (bandwidth * symbol period).
GFSK_BT = 0.35
# Frequency modulator sensitivity for GFSK.
# Lower values give narrower frequency deviation.
GFSK_SENSITIVITY = 0.1
# Modulation index for CPFSK.
CPFSK_K = 0.5

# ---- Parallelism ----
# Number of worker processes for parallel generation.
# Each worker handles one (modulation, SNR) pair at a time.
# Set to 1 for single-process (easiest to debug).
# Set to 0 or -1 to auto-detect CPU count.
#
# GNU Radio uses shared-memory circular buffers internally.  High worker
# counts can exhaust System V shm limits ("shmget: No space left on device").
# This project defaults to POSIX mmap buffers (vmcircbuf_mmap_shm_open) which
# have much higher limits, but if you still see bad_alloc errors, reduce this.
NUM_WORKERS = 20

# ---- Source generation settings ----
# If True, generate random symbol data.
# If False, read symbols from TEST_TEXT_FILE.
USE_RANDOM_SYMBOLS = True

# How many symbol bytes to generate per flowgraph run.
# The actual run length is auto-computed to ensure we get enough vectors.
SYMBOLS_PER_RUN = 20000
