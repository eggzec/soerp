"""
Tests for Scheduling Facilities example (six stations).
  T = s1 + s2 + s3 + s4 + s5 + s6
"""

import pytest
import scipy.stats as ss

from soerp import Chi2, Exp, Gamma, N, uv


TIGHT = dict(rel=1e-4)
LOOSE = dict(rel=2e-3)


def _moments(uf):
    return uf.mean, uf.var, uf.skew, uf.kurt


class TestSchedulingFacilities:
    MEAN = 51.7
    VAR = 33.3
    SKEW = 0.52
    KURT = 3.49

    def test_moments_input(self):
        s1 = uv([10, 1, 0, 3, 0, 0, 0, 0])
        s2 = uv([20, 2, 0, 3, 0, 0, 0, 0])
        s3 = uv([1.5, 0.25, 0.67, 3.67, 0, 0, 0, 0])
        s4 = uv([10, 10, 0.63, 3.6, 0, 0, 0, 0])
        s5 = uv([0.2, 0.04, 2, 9, 0, 0, 0, 0])
        s6 = uv([10, 20, 0.89, 4.2, 0, 0, 0, 0])
        T = s1 + s2 + s3 + s4 + s5 + s6
        mn, vr, sk, kt = _moments(T)
        assert mn == pytest.approx(self.MEAN, **TIGHT)
        assert vr == pytest.approx(self.VAR, rel=1e-3)
        assert sk == pytest.approx(self.SKEW, rel=5e-3)
        assert kt == pytest.approx(self.KURT, rel=5e-3)

    def test_scipy_rv(self):
        s1 = uv(rv=ss.norm(loc=10, scale=1))
        s2 = uv(rv=ss.norm(loc=20, scale=2**0.5))
        s3 = uv(rv=ss.gamma(9, scale=1 / 6.0))
        s4 = uv(rv=ss.gamma(10, scale=1))
        s5 = uv(rv=ss.expon(scale=0.2))
        s6 = uv(rv=ss.chi2(10))
        T = s1 + s2 + s3 + s4 + s5 + s6
        mn, vr, sk, kt = _moments(T)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, rel=5e-3)
        assert sk == pytest.approx(self.SKEW, rel=1e-2)
        assert kt == pytest.approx(self.KURT, rel=1e-2)

    def test_constructors(self):
        s1 = N(10, 1)
        s2 = N(20, 2**0.5)
        s3 = Gamma(9, 1 / 6.0)
        s4 = Gamma(10, 1)
        s5 = Exp(5)
        s6 = Chi2(10)
        T = s1 + s2 + s3 + s4 + s5 + s6
        mn, vr, sk, kt = _moments(T)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, rel=5e-3)
        assert sk == pytest.approx(self.SKEW, rel=1e-2)
        assert kt == pytest.approx(self.KURT, rel=1e-2)
