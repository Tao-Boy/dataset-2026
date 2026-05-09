"""Tests for normalization module."""

import numpy as np

from normalization import l2_normalize, l2_normalize_batch, has_nan_or_inf


class TestL2Normalize:
    def test_unit_vector_unchanged(self):
        """A vector with norm=1 should remain norm=1."""
        vec = np.array([1.0 + 0j], dtype=np.complex64)
        result = l2_normalize(vec)
        assert np.isclose(np.sqrt(np.sum(np.abs(result) ** 2)), 1.0)

    def test_scales_to_unit_norm(self):
        """Any non-zero vector should have norm=1 after normalization."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            vec = rng.normal(0, 2, 128) + 1j * rng.normal(0, 2, 128)
            vec = vec.astype(np.complex64)
            result = l2_normalize(vec)
            norm = np.sqrt(np.sum(np.abs(result) ** 2))
            assert np.isclose(norm, 1.0, atol=1e-6)

    def test_zero_vector_no_crash(self):
        """L2-normalizing a zero vector should not crash or produce NaN."""
        vec = np.zeros(128, dtype=np.complex64)
        result = l2_normalize(vec)
        assert not has_nan_or_inf(result)
        assert np.all(result == 0)

    def test_no_nan_inf_after_normalize(self):
        """Normalization should never produce NaN or Inf for random data."""
        rng = np.random.default_rng(123)
        for _ in range(10):
            vec = rng.normal(0, 1, 256) + 1j * rng.normal(0, 1, 256)
            vec = vec.astype(np.complex64)
            result = l2_normalize(vec)
            assert not has_nan_or_inf(result)


class TestL2NormalizeBatch:
    def test_batch_normalization(self):
        """Batch normalization should give each vector unit norm."""
        rng = np.random.default_rng(99)
        vectors = rng.normal(0, 2, (10, 128)) + 1j * rng.normal(0, 2, (10, 128))
        vectors = vectors.astype(np.complex64)
        result = l2_normalize_batch(vectors)
        norms = np.sqrt(np.sum(np.abs(result) ** 2, axis=1))
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_batch_with_zeros(self):
        """Batch with a zero vector should not crash."""
        rng = np.random.default_rng(55)
        vectors = rng.normal(0, 1, (5, 128)) + 1j * rng.normal(0, 1, (5, 128))
        vectors = vectors.astype(np.complex64)
        vectors[2] = 0  # zero out one vector
        result = l2_normalize_batch(vectors)
        assert not has_nan_or_inf(result)
        # Non-zero vectors should still normalize correctly
        for i in [0, 1, 3, 4]:
            norm = np.sqrt(np.sum(np.abs(result[i]) ** 2))
            assert np.isclose(norm, 1.0, atol=1e-6)


class TestHasNanOrInf:
    def test_normal_array(self):
        assert not has_nan_or_inf(np.array([1.0, 2.0, 3.0]))

    def test_with_nan(self):
        assert has_nan_or_inf(np.array([1.0, np.nan, 3.0]))

    def test_with_inf(self):
        assert has_nan_or_inf(np.array([1.0, np.inf, 3.0]))

    def test_with_neg_inf(self):
        assert has_nan_or_inf(np.array([-np.inf, 1.0]))
