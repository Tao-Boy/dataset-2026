"""
Normalization — L2 (Euclidean) normalization of complex I/Q vectors.

Each vector is scaled so that its Euclidean norm = 1:
    norm = sqrt(sum(|x_i|^2))
    x_normalized = x / norm

If the vector is all zeros (norm == 0), it is returned as-is to avoid
NaN or Inf values.
"""

import numpy as np


def l2_normalize(vector):
    """
    L2-normalize a single complex vector to unit norm.

    Parameters
    ----------
    vector : np.ndarray, 1-D complex
        Input complex vector.

    Returns
    -------
    normalized : np.ndarray, 1-D complex
        Vector with unit L2 norm (or unchanged if input is all zeros).
    """
    energy = np.sqrt(np.sum(np.abs(vector) ** 2))
    if energy > 0:
        return vector / energy
    return vector


def l2_normalize_batch(vectors):
    """
    L2-normalize a batch of complex vectors.

    Parameters
    ----------
    vectors : np.ndarray, shape (N, vec_len), complex
        Batch of complex vectors.

    Returns
    -------
    normalized : np.ndarray, shape (N, vec_len), complex
        Normalized batch.
    """
    if vectors.ndim == 1:
        return l2_normalize(vectors)

    energy = np.sqrt(np.sum(np.abs(vectors) ** 2, axis=1, keepdims=True))
    # Avoid division by zero
    energy = np.where(energy == 0, 1.0, energy)
    return vectors / energy


def has_nan_or_inf(array):
    """Check if array contains NaN or Inf values."""
    return bool(np.any(np.isnan(array)) or np.any(np.isinf(array)))
