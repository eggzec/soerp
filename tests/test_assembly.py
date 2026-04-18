r"""
Tests for Three-Part Assembly example.

$Z = \frac{x_1 \cdot x_2^2}{15 \cdot (1.5 + x_3)}$

where:
$x_1 \sim \mathcal{N}(24, 1)$
$x_2 \sim \mathcal{N}(37, 4)$
$x_3 \sim \text{Exp}(\lambda=2)$
"""

import pytest
import scipy.stats as ss

from soerp import Exp, N, uv


TIGHT = dict(rel=1e-4)
LOOSE = dict(rel=2e-3)


def _moments(uf):
    return uf.mean, uf.var, uf.skew, uf.kurt


class TestThreePartAssembly:
    MEAN = 1176.45
    VAR = 99699.682
    SKEW = 0.70801305
    KURT = 6.1632855

    def test_moments_input(self):
        x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])
        x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])
        x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])
        Z = (x1 * x2**2) / (15 * (1.5 + x3))
        mn, vr, sk, kt = _moments(Z)
        assert mn == pytest.approx(self.MEAN, **TIGHT)
        assert vr == pytest.approx(self.VAR, **TIGHT)
        assert sk == pytest.approx(self.SKEW, **TIGHT)
        # kurtosis is a 4th-order quantity; ~0.4 % tolerance
        assert kt == pytest.approx(self.KURT, rel=5e-3)

    def test_scipy_rv(self):
        x1 = uv(rv=ss.norm(loc=24, scale=1))
        x2 = uv(rv=ss.norm(loc=37, scale=4))
        x3 = uv(rv=ss.expon(scale=0.5))
        Z = (x1 * x2**2) / (15 * (1.5 + x3))
        mn, vr, sk, kt = _moments(Z)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, **LOOSE)
        assert sk == pytest.approx(self.SKEW, **LOOSE)
        # kurtosis is a 4th-order quantity; ~0.4 % tolerance
        assert kt == pytest.approx(self.KURT, rel=5e-3)

    def test_constructors(self):
        x1 = N(24, 1)
        x2 = N(37, 4)
        x3 = Exp(2)
        Z = (x1 * x2**2) / (15 * (1.5 + x3))
        mn, vr, sk, kt = _moments(Z)
        assert mn == pytest.approx(self.MEAN, **LOOSE)
        assert vr == pytest.approx(self.VAR, **LOOSE)
        assert sk == pytest.approx(self.SKEW, **LOOSE)
        # kurtosis is a 4th-order quantity; ~0.4 % tolerance
        assert kt == pytest.approx(self.KURT, rel=5e-3)
