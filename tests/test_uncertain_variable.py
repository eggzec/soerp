"""
Tests for building UncertainVariable objects, especially from scipy.stats
distributions, where the eight standardized moments are derived by integration.
"""

import pytest
import scipy.stats as ss

from soerp import N, U, uv
from soerp.uncertain_variable import matplotlib_installed


class TestConstructionErrors:
    def test_requires_moments_or_rv(self):
        with pytest.raises(ValueError, match="Either the moments"):
            uv()

    # scipy validates shape arguments when a distribution is frozen, so an
    # inconsistent rv cannot be built directly. ``args`` is a public attribute
    # of the frozen object, so we set it afterwards to reach soerp's own guard
    # (which also protects against non-scipy, duck-typed rv objects).

    def test_shape_arg_rejected_for_shapeless_distribution(self):
        rv = ss.norm(loc=0, scale=1)  # norm takes no shape parameters
        rv.args = (0.5,)
        with pytest.raises(ValueError, match="doesn't support"):
            uv(rv=rv)

    def test_missing_shape_arg_rejected(self):
        rv = ss.gamma(3, loc=0, scale=1)  # gamma requires a shape parameter
        rv.args = ()
        with pytest.raises(ValueError, match="requires a third"):
            uv(rv=rv)


class TestStandardizedMoments:
    """Moments 3-8 are standardized, so they depend only on distribution
    shape - never on location or scale."""

    def test_uniform_moments_are_standardized(self):
        """Regression: the branch for distributions without a shape parameter
        ignored loc/scale and skipped standardization entirely, so U(2, 6)
        reported mu_4 = 0.0125 instead of 1.8.

        For a standardized uniform, mu_(2k) = 3**k / (2k + 1).
        """
        x = U(2, 6)
        mn, vr, *standardized = x.moments()
        assert mn == pytest.approx(4.0)
        assert vr == pytest.approx(4.0 / 3.0)
        assert standardized == pytest.approx(
            [0.0, 9 / 5, 0.0, 27 / 7, 0.0, 81 / 9], abs=1e-6
        )

    @pytest.mark.parametrize(
        ("loc", "scale"), [(0.0, 1.0), (24.0, 1.0), (-5.0, 4.0), (1000.0, 0.5)]
    )
    def test_normal_moments_invariant_to_loc_and_scale(self, loc, scale):
        x = uv(rv=ss.norm(loc=loc, scale=scale))
        mn, vr, *standardized = x.moments()
        assert mn == pytest.approx(loc)
        assert vr == pytest.approx(scale**2)
        assert standardized == pytest.approx(
            [0.0, 3.0, 0.0, 15.0, 0.0, 105.0], abs=1e-6
        )

    def test_exponential_moments(self):
        x = uv(rv=ss.expon(scale=0.5))
        _, vr, *standardized = x.moments()
        assert vr == pytest.approx(0.25)
        assert standardized == pytest.approx(
            [2.0, 9.0, 44.0, 265.0, 1854.0, 14833.0], rel=1e-6
        )


class TestMomentAccessors:
    def test_moments_returns_all_eight(self):
        x = uv([10, 4, 0, 3, 0, 15, 0, 105])
        assert x.moments() == [10, 4, 0, 3, 0, 15, 0, 105]

    @pytest.mark.parametrize(
        ("idx", "want"), [(0, 10), (1, 4), (2, 0), (3, 3), (7, 105)]
    )
    def test_moments_by_index(self, idx, want):
        x = uv([10, 4, 0, 3, 0, 15, 0, 105])
        assert x.moments(idx) == want

    def test_out_of_range_index_returns_the_whole_list(self):
        x = uv([10, 4, 0, 3, 0, 15, 0, 105])
        assert x.moments(99) == x.moments()

    def test_named_properties(self):
        x = uv([10, 4, 0.5, 3.2, 0, 15, 0, 105])
        assert x.mean == 10
        assert x.var == 4
        assert x.std == pytest.approx(2.0)
        assert x.skew == 0.5
        assert x.kurt == 3.2

    def test_hash_is_identity_based(self):
        x = N(1.0, 0.1)
        assert hash(x) == hash(x)
        assert hash(x) != hash(N(1.0, 0.1))


class TestSetters:
    @staticmethod
    def _variable():
        return uv([10, 4, 0, 3, 0, 15, 0, 105])

    def test_set_mean(self):
        x = self._variable()
        x.set_mean(42.0)
        assert x.mean == 42.0

    def test_set_std(self):
        x = self._variable()
        x.set_std(3.0)
        assert x.var == pytest.approx(9.0)

    def test_set_var(self):
        x = self._variable()
        x.set_var(25.0)
        assert x.var == 25.0
        assert x.std == pytest.approx(5.0)

    def test_set_skew(self):
        x = self._variable()
        x.set_skew(1.5)
        assert x.skew == 1.5

    def test_set_kurt(self):
        x = self._variable()
        x.set_kurt(4.5)
        assert x.kurt == 4.5

    def test_set_moments(self):
        x = self._variable()
        x.set_moments([1, 2, 3, 4, 5, 6, 7, 8])
        assert x.moments() == [1, 2, 3, 4, 5, 6, 7, 8]

    @pytest.mark.parametrize("bad", [[1, 2, 3], [1] * 9, []])
    def test_set_moments_requires_exactly_eight(self, bad):
        x = self._variable()
        with pytest.raises(ValueError, match="eight values"):
            x.set_moments(bad)


@pytest.mark.skipif(not matplotlib_installed, reason="matplotlib not installed")
class TestPlot:
    def test_plotting_needs_a_scipy_distribution(self):
        x = uv([10, 4, 0, 3, 0, 15, 0, 105])
        with pytest.raises(NotImplementedError, match="Cannot determine"):
            x.plot()

    def test_plot_from_a_scipy_distribution(self, monkeypatch):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        monkeypatch.setattr(plt, "show", lambda *a, **k: None)
        x = uv(rv=ss.norm(loc=0, scale=1))
        x.plot()
        x.plot(vals=[-2.0, 2.0])
        plt.close("all")
