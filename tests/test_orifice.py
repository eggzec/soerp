"""
Tests for Orifice Flow Meter example.
  Q = C * sqrt(520*H*P / (M*(t+460)))
  H ~ N(64, 0.5),  M ~ N(16, 0.1),  P ~ N(361, 2),  t ~ N(165, 0.5)
"""

import pytest
import scipy.stats as ss

from soerp import N, umath, uv


TIGHT = dict(rel=1e-4)
LOOSE = dict(rel=2e-3)


def _moments(uf):
    return uf.mean, uf.var, uf.skew, uf.kurt


class TestOrificeFlowMeter:
    C = 38.4
    MEAN = 1330.9997
    VAR = 58.210763
    SKEW = 0.010942207
    KURT = 3.0003269

    def test_moments_input(self):
        H = uv([64, 0.25, 0, 3, 0, 15, 0, 105])
        M = uv([16, 0.01, 0, 3, 0, 15, 0, 105])
        P = uv([361, 4, 0, 3, 0, 15, 0, 105])
        t = uv([165, 0.25, 0, 3, 0, 15, 0, 105])
        Q = self.C * umath.sqrt((520 * H * P) / (M * (t + 460)))
        mn, vr, sk, kt = _moments(Q)
        assert mn == pytest.approx(self.MEAN, **TIGHT)
        assert vr == pytest.approx(self.VAR, **TIGHT)
        assert sk == pytest.approx(self.SKEW, **TIGHT)
        assert kt == pytest.approx(self.KURT, **TIGHT)

    def test_scipy_rv(self):
        H = uv(rv=ss.norm(loc=64, scale=0.5))
        M = uv(rv=ss.norm(loc=16, scale=0.1))
        P = uv(rv=ss.norm(loc=361, scale=2))
        t = uv(rv=ss.norm(loc=165, scale=0.5))
        Q = self.C * umath.sqrt((520 * H * P) / (M * (t + 460)))
        mn, vr, sk, kt = _moments(Q)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, **LOOSE)
        assert sk == pytest.approx(self.SKEW, **LOOSE)
        assert kt == pytest.approx(self.KURT, **LOOSE)

    def test_constructors(self):
        H = N(64, 0.5)
        M = N(16, 0.1)
        P = N(361, 2)
        t = N(165, 0.5)
        Q = self.C * umath.sqrt((520 * H * P) / (M * (t + 460)))
        mn, vr, sk, kt = _moments(Q)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, **LOOSE)
        assert sk == pytest.approx(self.SKEW, **LOOSE)
        assert kt == pytest.approx(self.KURT, **LOOSE)
