"""
Tests for PDF estimation from moments using Cornish-Fisher expansion.
"""

import numpy as np
import pytest

from soerp import N, cornish_fisher_quantile, uv
from soerp.uncertain_variable import pdf as pdf_func


class TestPDFEstimation:
    """Test PDF estimation functionality."""

    def test_standard_normal_pdf_at_mean(self):
        """Standard normal PDF at mean should be ~0.3989."""
        x = N(0, 1)
        pdf_val = x.pdf(0.0)
        expected = 1 / np.sqrt(2 * np.pi)
        assert pdf_val == pytest.approx(expected, rel=1e-3)

    def test_standard_normal_pdf_symmetric(self):
        """Normal PDF should be symmetric around mean."""
        x = N(0, 1)
        pdf_neg = x.pdf(-1.0)
        pdf_pos = x.pdf(1.0)
        assert pdf_neg == pytest.approx(pdf_pos, rel=1e-10)

    def test_normal_pdf_normalized(self):
        """PDF should integrate to approximately 1 over a reasonable range."""
        x = N(0, 1)
        xs = np.linspace(-5, 5, 1000)
        pdf_vals = x.pdf(xs)
        integral = np.trapezoid(pdf_vals, xs)
        assert integral == pytest.approx(1.0, rel=0.05)

    def test_shifted_normal_pdf(self):
        """PDF should shift with mean."""
        x = N(5, 2)
        pdf_at_mean = x.pdf(5.0)
        pdf_at_minus_1 = x.pdf(3.0)
        assert pdf_at_mean > pdf_at_minus_1

    def test_scaled_normal_pdf(self):
        """Higher variance should give lower peak."""
        x1 = N(0, 1)
        x2 = N(0, 4)
        pdf1 = x1.pdf(0.0)
        pdf2 = x2.pdf(0.0)
        assert pdf2 < pdf1

    def test_skewed_distribution_pdf(self):
        """Skewed distribution should have asymmetric PDF."""
        x = uv([0, 1, 1.0, 10])
        pdf_left = x.pdf(-2)
        pdf_right = x.pdf(2)
        assert pdf_left > pdf_right

    def test_pdf_array_input(self):
        """PDF should work with array input."""
        x = N(0, 1)
        xs = np.array([-1.0, 0.0, 1.0])
        pdf_vals = x.pdf(xs)
        assert len(pdf_vals) == 3
        assert pdf_vals[1] > pdf_vals[0]
        assert pdf_vals[1] > pdf_vals[2]

    def test_pdf_single_value_float(self):
        """PDF should return float for single float input."""
        x = N(0, 1)
        result = x.pdf(0.0)
        assert isinstance(result, float)

    def test_pdf_sum_of_normals(self):
        """Sum of two normals should have PDF approximating convolution."""
        x1 = N(0, 1)
        x2 = N(0, 1)
        z = x1 + x2
        pdf_z = z.pdf(0.0)
        expected = 1 / np.sqrt(4 * np.pi)
        assert pdf_z == pytest.approx(expected, rel=0.1)

    def test_pdf_product_of_normals(self):
        """Product of two normals."""
        x1 = N(1, 0.5)
        x2 = N(2, 0.5)
        z = x1 * x2
        pdf_val = z.pdf(z.mean)
        assert pdf_val > 0

    def test_pdf_exponential_approximation(self):
        """Test PDF for exponential-like distribution."""
        x = uv([1, 1, 2, 6])
        pdf_at_mean = x.pdf(1.0)
        assert pdf_at_mean == pytest.approx(0.368, rel=0.2)

    def test_pdf_negative_variance_raises(self):
        """Negative variance should raise error."""
        x = uv([0, -1, 0, 3])
        with pytest.raises(ValueError, match="Variance must be positive"):
            x.pdf(0.0)

    def test_cornish_fisher_quantile_function(self):
        """Test the underlying quantile function."""
        result = cornish_fisher_quantile(0, 0, 1, 0, 0)
        assert result == pytest.approx(0.0, abs=1e-10)

        result_skew = cornish_fisher_quantile(0, 0, 1, 1, 0)
        assert result_skew != 0.0  # noqa RUF069

    def test_pdf_standalone_function(self):
        """Test standalone pdf function from uncertain_variable."""
        result = pdf_func(0.0, 0, 1)
        expected = 1 / np.sqrt(2 * np.pi)
        assert result == pytest.approx(expected, rel=1e-3)


class TestPDFConsistency:
    """Test PDF consistency with moment properties."""

    def test_pdf_decreases_away_from_mean_normal(self):
        """For normal, PDF should decrease as we move away from mean."""
        x = N(0, 1)
        pdf_0 = x.pdf(0.0)
        pdf_1 = x.pdf(1.0)
        pdf_2 = x.pdf(2.0)
        assert pdf_0 > pdf_1 > pdf_2

    def test_pdf_peak_at_mean_skewed(self):
        """Even skewed distribution should peak near mean."""
        x = uv([1, 1, 1.5, 8])
        pdf_at_mean = x.pdf(1.0)
        pdf_far = x.pdf(1.0 + 3 * x.std)
        assert pdf_at_mean > pdf_far
