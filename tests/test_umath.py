"""
Tests for soerp.umath functions — scalar pass-through and UF propagation.
"""

import math

import pytest

from soerp import N, umath


class TestUmath:
    """Verify that umath functions return the correct value at the mean
    and that uncertainty is propagated (variance > 0)."""

    def test_sqrt_scalar(self):
        assert umath.sqrt(4.0) == pytest.approx(2.0)

    def test_sqrt_uncertain(self):
        x = N(4.0, 0.1)
        r = umath.sqrt(x)
        assert r.mean == pytest.approx(2.0, rel=1e-3)
        assert r.var > 0

    def test_sin_scalar(self):
        assert umath.sin(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_sin_uncertain(self):
        x = N(math.pi / 6, 0.01)  # $\sin(\pi/6) = 0.5$
        r = umath.sin(x)
        assert r.mean == pytest.approx(0.5, rel=1e-3)
        assert r.var > 0

    def test_cos_scalar(self):
        assert umath.cos(0.0) == pytest.approx(1.0)

    def test_exp_scalar(self):
        assert umath.exp(1.0) == pytest.approx(math.e)

    def test_exp_uncertain(self):
        # E[exp(X)] for X~N(0, sigma) includes the second-order correction:
        # E[exp(X)] ≈ $\exp(\mu) \cdot (1 + \sigma^2/2) = \exp(0) \cdot (1 + 0.01/2) = 1.005$  # noqa: E501
        x = N(0.0, 0.1)
        r = umath.exp(x)
        assert r.mean == pytest.approx(1.005, rel=1e-3)
        assert r.var > 0

    def test_log_scalar(self):
        assert umath.log(math.e) == pytest.approx(1.0)

    def test_log_uncertain(self):
        x = N(1.0, 0.01)
        r = umath.log(x)
        assert r.mean == pytest.approx(0.0, abs=1e-4)
        assert r.var > 0

    def test_ln_equals_log_base_e(self):
        x = N(2.0, 0.1)
        assert umath.ln(x).mean == pytest.approx(umath.log(x).mean, rel=1e-10)

    def test_atan2_scalar(self):
        assert umath.atan2(1.0, 1.0) == pytest.approx(math.pi / 4)

    def test_hypot_scalar(self):
        assert umath.hypot(3.0, 4.0) == pytest.approx(5.0)

    def test_hypot_uncertain(self):
        x = N(3.0, 0.1)
        y = N(4.0, 0.1)
        r = umath.hypot(x, y)
        assert r.mean == pytest.approx(5.0, rel=1e-3)
        assert r.var > 0

    def test_erf_scalar(self):
        assert umath.erf(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_erfc_scalar(self):
        assert umath.erfc(0.0) == pytest.approx(1.0)

    def test_degrees_scalar(self):
        assert umath.degrees(math.pi) == pytest.approx(180.0)

    def test_radians_scalar(self):
        assert umath.radians(180.0) == pytest.approx(math.pi)

    def test_floor_uncertain(self):
        x = N(3.7, 0.01)
        r = umath.floor(x)
        assert r.mean == pytest.approx(3.0)

    def test_ceil_uncertain(self):
        x = N(3.2, 0.01)
        r = umath.ceil(x)
        assert r.mean == pytest.approx(4.0)
