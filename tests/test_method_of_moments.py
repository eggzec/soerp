"""
Coverage of the Python layer in ``soerp.method_of_moments``: the
standardization helpers, the argument checks around the FORTRAN kernel,
the printed report and the raw-to-central conversion.
"""

from typing import ClassVar

import numpy as np
import pytest

import soerp.method_of_moments as mm
from soerp import raw2central
from soerp.method_of_moments import (
    centralmoment,
    rawmoment,
    soerp_numeric,
    standard_cp,
    standard_lc,
    standard_qc,
    standardize,
    variance_components,
    variance_contrib,
)


NORM = [1, 0, 1, 0, 3, 0, 15, 0, 105]

LC = np.array([-802.65, -430.5])
QC = np.array([205.54, 78.66])
CP = np.array([[0.0, -216.5], [-216.5, 0.0]])
VM = np.array([NORM, NORM], dtype=float)


class TestStandardization:
    STDEVS = np.array([2.0, 3.0])

    def test_standard_lc_scales_by_sigma(self):
        got = standard_lc(np.array([1.0, 2.0]), self.STDEVS)
        assert got == pytest.approx([2.0, 6.0])

    def test_standard_qc_scales_by_variance(self):
        got = standard_qc(np.array([1.0, 2.0]), self.STDEVS)
        assert got == pytest.approx([4.0, 18.0])

    def test_standard_cp_scales_by_both_sigmas(self):
        cp = np.array([[0.0, 5.0], [5.0, 0.0]])
        got = standard_cp(cp, self.STDEVS)
        assert got[0, 1] == pytest.approx(30.0)
        assert got[1, 0] == pytest.approx(30.0)

    def test_standard_cp_single_variable_is_untouched(self):
        cp = np.zeros((1, 1))
        assert standard_cp(cp, np.array([2.0])).shape == (1, 1)

    def test_standardize_returns_all_three(self):
        cp = np.array([[0.0, 5.0], [5.0, 0.0]])
        slc, sqc, scp = standardize(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), cp, self.STDEVS
        )
        assert slc == pytest.approx([2.0, 6.0])
        assert sqc == pytest.approx([4.0, 18.0])
        assert scp[0, 1] == pytest.approx(30.0)


class TestMomentArgumentChecks:
    @pytest.mark.parametrize("k", [-1, 5, 100])
    def test_rawmoment_rejects_unsupported_order(self, k):
        with pytest.raises(ValueError, match="raw moments"):
            rawmoment(LC, QC, CP, VM, k)

    @pytest.mark.parametrize("k", [-1, 5, 100])
    def test_centralmoment_rejects_unsupported_order(self, k):
        with pytest.raises(ValueError, match="central moments"):
            centralmoment([1.0, 0.0, 1.0, 0.0, 3.0], k)

    def test_rawmoment_accepts_plain_lists(self):
        from_lists = rawmoment(list(LC), list(QC), [list(r) for r in CP], VM, 2)
        assert from_lists == pytest.approx(rawmoment(LC, QC, CP, VM, 2))

    def test_zeroth_moment_is_one(self):
        assert rawmoment(LC, QC, CP, VM, 0) == pytest.approx(1.0)

    def test_centralmoment_accepts_short_input(self):
        # Only the first three raw moments are supplied.
        assert centralmoment([1.0, 2.0, 5.0], 2) == pytest.approx(1.0)


class TestCentralMoments:
    """mu2 = 1, mu3 = 0 and mu4 = 6 for these raw moments."""

    VY: ClassVar[list[float]] = [1.0, 2.0, 5.0, 14.0, 46.0]

    @pytest.mark.parametrize(
        ("k", "want"), [(0, 1.0), (1, 0.0), (2, 1.0), (3, 0.0), (4, 6.0)]
    )
    def test_known_values(self, k, want):
        assert centralmoment(self.VY, k) == pytest.approx(want)


class TestAssumeLinear:
    def test_second_order_terms_are_dropped(self, monkeypatch):
        """With assume_linear set, only the linear coefficients survive."""
        monkeypatch.setattr(mm, "assume_linear", True)
        got = rawmoment(LC, QC, CP, VM, 2)
        expected = LC[0] ** 2 + LC[1] ** 2  # both inputs standardized
        assert got == pytest.approx(expected)

    def test_first_moment_vanishes_without_quadratic_terms(self, monkeypatch):
        monkeypatch.setattr(mm, "assume_linear", True)
        assert rawmoment(LC, QC, CP, VM, 1) == pytest.approx(0.0)


class TestVarianceReporting:
    def test_components_and_contributions(self):
        vy = [rawmoment(LC, QC, CP, VM, k) for k in range(5)]
        vz = [centralmoment(vy, k) for k in range(5)]
        vc_lc, vc_qc, vc_cp = variance_components(LC, QC, CP, VM, vz)

        assert vc_lc.shape == (2,)
        assert vc_qc.shape == (2,)
        assert vc_cp.shape == (2, 2)

        clc, cqc, ccp = variance_contrib(vc_lc, vc_qc, vc_cp, vz)
        assert np.all(clc >= 0)
        assert np.all(cqc >= 0)
        assert np.all(ccp >= 0)

    def test_contributions_are_zero_when_there_is_no_variance(self):
        zeros = np.zeros(2)
        clc, cqc, ccp = variance_contrib(
            zeros, zeros, np.zeros((2, 2)), [1.0, 0.0, 0.0, 0.0, 0.0]
        )
        assert np.all(clc == 0.0)
        assert np.all(cqc == 0.0)
        assert np.all(ccp == 0.0)


class TestSoerpNumeric:
    """The published example from the original SOERP user guide."""

    F0 = 4152
    EXPECTED: ClassVar[list[float]] = [4436.2, 973317.70, 0.61068742, 4.1121529]

    def test_silent_returns_the_four_moments(self):
        got = soerp_numeric(LC, QC, CP, VM, self.F0, silent=True)
        assert got[0] == pytest.approx(self.EXPECTED[0], rel=1e-6)
        assert got[1] == pytest.approx(self.EXPECTED[1], rel=1e-6)
        assert got[2] == pytest.approx(self.EXPECTED[2], rel=1e-6)
        assert got[3] == pytest.approx(self.EXPECTED[3], rel=1e-6)

    def test_silent_prints_nothing(self, capsys):
        soerp_numeric(LC, QC, CP, VM, self.F0, silent=True)
        assert not capsys.readouterr().out

    def test_report_is_printed(self, capsys):
        soerp_numeric(LC, QC, CP, VM, self.F0, title="EXAMPLE")
        out = capsys.readouterr().out
        assert "SOERP: EXAMPLE" in out
        assert "MEAN-INTERCEPT (EDEL1)" in out
        assert "COEFFICIENT OF KURTOSIS (BETA2)" in out
        assert "Variance Contribution of lc[x0]" in out
        assert "Variance Contribution of cp[x0, x1]" in out

    def test_report_without_a_title(self, capsys):
        soerp_numeric(LC, QC, CP, VM, self.F0)
        assert "VARIANCE (VARDL)" in capsys.readouterr().out

    def test_debug_reports_every_moment(self, capsys):
        soerp_numeric(LC, QC, CP, VM, self.F0, debug=True)
        out = capsys.readouterr().out
        for k in range(5):
            assert f"Raw Moment {k}:" in out
            assert f"Central Moment {k}:" in out

    def test_debug_is_overridden_by_silent(self, capsys):
        soerp_numeric(LC, QC, CP, VM, self.F0, debug=True, silent=True)
        assert not capsys.readouterr().out

    def test_degenerate_input_has_no_skewness_or_kurtosis(self):
        zeros = np.zeros(2)
        got = soerp_numeric(
            zeros, zeros, np.zeros((2, 2)), VM, 1.0, silent=True
        )
        assert got[1] == pytest.approx(0.0)
        assert got[2] == pytest.approx(0.0)
        assert got[3] == pytest.approx(0.0)


class TestRaw2Central:
    def test_matches_the_textbook_relations(self):
        raw = [2.0, 5.0, 14.0, 46.0]
        got = raw2central(raw)
        assert got[0] == pytest.approx(0.0)
        assert got[1] == pytest.approx(1.0)
        assert got[2] == pytest.approx(0.0)
        assert got[3] == pytest.approx(6.0)

    def test_length_is_preserved(self):
        assert len(raw2central([1.0, 2.0, 3.0, 4.0, 5.0])) == 5


class TestNoInputVariables:
    """A constant has no input variables, so the kernel would be handed
    zero-length arrays. The wrappers must short-circuit instead."""

    EMPTY_LC = np.zeros(0)
    EMPTY_CP = np.zeros((0, 0))
    EMPTY_VM = np.zeros((0, 9))

    @pytest.mark.parametrize(
        ("k", "want"), [(0, 1.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.0)]
    )
    def test_rawmoment(self, k, want):
        got = rawmoment(
            self.EMPTY_LC, self.EMPTY_LC, self.EMPTY_CP, self.EMPTY_VM, k
        )
        assert got == pytest.approx(want)

    def test_variance_components_are_empty(self):
        vc_lc, vc_qc, vc_cp = variance_components(
            self.EMPTY_LC,
            self.EMPTY_LC,
            self.EMPTY_CP,
            self.EMPTY_VM,
            [1.0, 0.0, 0.0, 0.0, 0.0],
        )
        assert vc_lc.size == 0
        assert vc_qc.size == 0
        assert vc_cp.size == 0

    def test_soerp_numeric_returns_the_intercept(self):
        got = soerp_numeric(
            self.EMPTY_LC,
            self.EMPTY_LC,
            self.EMPTY_CP,
            self.EMPTY_VM,
            7.0,
            silent=True,
        )
        assert got == pytest.approx([7.0, 0.0, 0.0, 0.0])
