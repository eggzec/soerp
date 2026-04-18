"""
Tests for Two-Bar Truss example.
Tests that means match deterministic values and variances are positive.
"""

import math

import pytest

from soerp import N, umath


class TestTwoBarTruss:
    pi = math.pi

    def _build(self, cls):
        H = cls(30, 5 / 3.0, tag="H")
        B = cls(60, 0.5 / 3.0, tag="B")
        d = cls(3, 0.1 / 3, tag="d")
        t = cls(0.15, 0.01 / 3, tag="t")
        E = cls(30000, 1500 / 3.0, tag="E")
        rho = cls(0.3, 0.01 / 3.0, tag="rho")
        P = cls(66, 1.0, tag="P")
        return H, B, d, t, E, rho, P

    def _compute(self, H, B, d, t, E, rho, P):  # noqa: PLR0913, PLR0917
        pi = self.pi
        wght = 2 * pi * rho * d * t * umath.sqrt((B / 2) ** 2 + H**2)
        strs = (P * umath.sqrt((B / 2) ** 2 + H**2)) / (2 * pi * d * t * H)
        buck = (pi**2 * E * (d**2 + t**2)) / (8 * ((B / 2) ** 2 + H**2))
        defl = (P * ((B / 2) ** 2 + H**2) ** 1.5) / (2 * pi * d * t * H**2 * E)
        return wght, strs, buck, defl

    def test_means_match_deterministic(self):  # noqa: PLR0914
        H, B, d, t, E, rho, P = self._build(N)
        wght, strs, buck, defl = self._compute(H, B, d, t, E, rho, P)

        Hv, Bv, dv, tv, Ev, rv, Pv = 30, 60, 3, 0.15, 30000, 0.3, 66
        r2 = (Bv / 2) ** 2 + Hv**2
        wght_det = 2 * self.pi * rv * dv * tv * math.sqrt(r2)
        strs_det = (Pv * math.sqrt(r2)) / (2 * self.pi * dv * tv * Hv)
        buck_det = (self.pi**2 * Ev * (dv**2 + tv**2)) / (8 * r2)
        defl_det = (Pv * r2**1.5) / (2 * self.pi * dv * tv * Hv**2 * Ev)

        # wght is a product formula - small 2nd-order correction to the mean
        assert wght.mean == pytest.approx(wght_det, rel=1e-3)
        # strs/defl include 1/H terms so the 2nd-order mean correction is
        # larger (~0.3 %); use a 1 % tolerance for those
        assert strs.mean == pytest.approx(strs_det, rel=1e-2)
        assert buck.mean == pytest.approx(buck_det, rel=1e-2)
        assert defl.mean == pytest.approx(defl_det, rel=1e-2)

    def test_variances_positive(self):
        H, B, d, t, E, rho, P = self._build(N)
        wght, strs, buck, defl = self._compute(H, B, d, t, E, rho, P)
        assert wght.var > 0
        assert strs.var > 0
        assert buck.var > 0
        assert defl.var > 0

    def test_error_components_sum_to_variance(self):
        H, B, d, t, E, rho, P = self._build(N)
        wght, *_ = self._compute(H, B, d, t, E, rho, P)
        ec = wght.error_components()
        assert sum(ec.values()) == pytest.approx(wght.var, rel=1e-6)

    def test_tags_preserved(self):
        H, B, _d, _t, _E, rho, _P = self._build(N)
        assert repr(H) == "H"
        assert repr(B) == "B"
        assert repr(rho) == "rho"
