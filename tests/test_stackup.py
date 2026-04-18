"""
Tests for Manufacturing Tolerance Stackup example.
  w = x + y + z,  each ~ Gamma(shape=9, scale=1/6)
"""

import pytest
import scipy.stats as ss

from soerp import Gamma, uv


LOOSE = dict(rel=2e-3)


def _moments(uf):
    return uf.mean, uf.var, uf.skew, uf.kurt


class TestToleranceStackup:
    MEAN = 4.5
    VAR = 0.75
    SKEW = 0.385
    KURT = 3.22

    def _gamma_moments(self):
        return [1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0]

    def test_moments_input(self):
        m = self._gamma_moments()
        x, y, z = uv(m), uv(m), uv(m)
        w = x + y + z
        mn, vr, sk, kt = _moments(w)
        assert mn == pytest.approx(self.MEAN, rel=1e-4)
        assert vr == pytest.approx(self.VAR, rel=1e-4)
        assert sk == pytest.approx(self.SKEW, rel=1e-3)
        assert kt == pytest.approx(self.KURT, rel=1e-3)

    def test_scipy_rv(self):
        mn_val, vr_val = 1.5, 0.25
        shape = mn_val**2 / vr_val
        scale = vr_val / mn_val
        rv = ss.gamma(shape, scale=scale)
        x, y, z = uv(rv=rv), uv(rv=rv), uv(rv=rv)
        w = x + y + z
        mn, vr, sk, kt = _moments(w)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, **LOOSE)
        assert sk == pytest.approx(self.SKEW, rel=5e-3)
        assert kt == pytest.approx(self.KURT, rel=5e-3)

    def test_constructors(self):
        mn_val, vr_val = 1.5, 0.25
        shape = mn_val**2 / vr_val
        scale = vr_val / mn_val
        x = Gamma(shape, scale)
        y = Gamma(shape, scale)
        z = Gamma(shape, scale)
        w = x + y + z
        mn, vr, sk, kt = _moments(w)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, **LOOSE)
        assert sk == pytest.approx(self.SKEW, rel=5e-3)
        assert kt == pytest.approx(self.KURT, rel=5e-3)
