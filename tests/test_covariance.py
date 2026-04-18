"""
Tests for covariance and correlation matrix utilities.
"""

import pytest

from soerp import N, correlation_matrix, covariance_matrix


class TestCovarianceMatrix:
    def test_covariance_matrix(self):
        x = N(1, 0.1)
        y = N(10, 0.1)
        z = x + 2 * y
        cov = covariance_matrix([x, y, z])
        assert cov[0][0] == pytest.approx(0.01, rel=1e-6)
        assert cov[1][1] == pytest.approx(0.01, rel=1e-6)
        assert cov[2][2] == pytest.approx(0.05, rel=1e-3)
        assert cov[0][1] == pytest.approx(0.0, abs=1e-10)
        assert cov[0][2] == pytest.approx(0.01, rel=1e-6)
        assert cov[1][2] == pytest.approx(0.02, rel=1e-6)
        # symmetry
        assert cov[1][0] == cov[0][1]
        assert cov[2][0] == cov[0][2]
        assert cov[2][1] == cov[1][2]

    def test_correlation_matrix_diagonal_is_one(self):
        x = N(1, 0.1)
        y = N(10, 0.5)
        corr = correlation_matrix([x, y])
        assert corr[0][0] == pytest.approx(1.0, abs=1e-10)
        assert corr[1][1] == pytest.approx(1.0, abs=1e-10)

    def test_correlation_matrix_independent(self):
        x = N(0, 1)
        y = N(0, 1)
        corr = correlation_matrix([x, y])
        assert corr[0][1] == pytest.approx(0.0, abs=1e-10)

    def test_correlation_matrix_perfect_correlation(self):
        x = N(5, 2)
        z = x + 0  # identical to x (same variable)
        corr = correlation_matrix([x, z])
        assert corr[0][1] == pytest.approx(1.0, rel=1e-6)
