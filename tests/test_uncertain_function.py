"""
Coverage of the UncertainFunction arithmetic, comparison and reporting
surface: the operator overloads, the derivative accessors and the
error-component reports.
"""

import math

import pytest

from soerp import N, UncertainFunction, to_uncertain_func, uv
from soerp.uncertain_function import make_uf_compatible_object


class TestToUncertainFunc:
    def test_passes_through_an_uncertain_function(self):
        x = N(1.0, 0.1)
        assert to_uncertain_func(x) is x

    @pytest.mark.parametrize("value", [3, 3.0])
    def test_wraps_plain_constants(self, value):
        wrapped = to_uncertain_func(value)
        assert isinstance(wrapped, UncertainFunction)
        assert wrapped.d() == {}

    def test_returns_none_for_unknown_types(self):
        assert to_uncertain_func("not a number") is None

    def test_compatibility_shim_is_identity(self):
        x = N(1.0, 0.1)
        assert make_uf_compatible_object(x) is x


class TestArithmetic:
    """Each operator is checked against the value it must produce and the
    first derivative it must carry."""

    def test_add(self):
        x, y = N(3.0, 0.1), N(4.0, 0.1)
        r = x + y
        assert r.x == pytest.approx(7.0)
        assert r.d(x) == pytest.approx(1.0)
        assert r.d(y) == pytest.approx(1.0)

    def test_radd(self):
        x = N(3.0, 0.1)
        assert (2 + x).x == pytest.approx(5.0)

    def test_sub(self):
        x, y = N(3.0, 0.1), N(4.0, 0.1)
        r = x - y
        assert r.x == pytest.approx(-1.0)
        assert r.d(y) == pytest.approx(-1.0)

    def test_rsub(self):
        x = N(3.0, 0.1)
        assert (10 - x).x == pytest.approx(7.0)

    def test_mul(self):
        x, y = N(3.0, 0.1), N(4.0, 0.1)
        r = x * y
        assert r.x == pytest.approx(12.0)
        assert r.d(x) == pytest.approx(4.0)
        assert r.d2c(x, y) == pytest.approx(1.0)

    def test_rmul(self):
        x = N(3.0, 0.1)
        assert (2 * x).x == pytest.approx(6.0)

    def test_mul_by_a_plain_number(self):
        x = N(3.0, 0.1)
        r = x * 2
        assert r.x == pytest.approx(6.0)
        assert r.d(x) == pytest.approx(2.0)

    def test_truediv(self):
        x, y = N(6.0, 0.1), N(3.0, 0.1)
        r = x / y
        assert r.x == pytest.approx(2.0)
        assert r.d(x) == pytest.approx(1 / 3)
        assert r.d(y) == pytest.approx(-6 / 9)

    def test_rtruediv(self):
        x = N(4.0, 0.1)
        assert (8 / x).x == pytest.approx(2.0)

    def test_pow_constant_exponent(self):
        x = N(3.0, 0.1)
        r = x**2
        assert r.x == pytest.approx(9.0)
        assert r.d(x) == pytest.approx(6.0)
        assert r.d2(x) == pytest.approx(2.0)

    def test_pow_uncertain_exponent(self):
        x, y = N(3.0, 0.01), N(2.0, 0.01)
        r = x**y
        assert r.x == pytest.approx(9.0)
        assert r.d(x) == pytest.approx(2 * 3.0)
        assert r.d(y) == pytest.approx(9.0 * math.log(3.0))

    def test_rpow(self):
        x = N(2.0, 0.01)
        assert (3**x).x == pytest.approx(9.0)

    @pytest.mark.parametrize("exponent", [0.5, 1.0, 2.0, 3.0])
    def test_pow_near_zero_base_stays_finite(self, exponent):
        r = N(0.0, 1e-3) ** exponent
        assert math.isfinite(r.x)

    def test_neg(self):
        x = N(3.0, 0.1)
        r = -x
        assert r.x == pytest.approx(-3.0)
        assert r.d(x) == pytest.approx(-1.0)

    def test_pos(self):
        x = N(3.0, 0.1)
        r = +x
        assert r.x == pytest.approx(3.0)
        assert r.d(x) == pytest.approx(1.0)

    def test_abs_of_negative(self):
        x = N(-3.0, 0.1)
        r = abs(x)
        assert r.x == pytest.approx(3.0)
        assert r.d(x) == pytest.approx(-1.0)

    def test_abs_of_positive(self):
        x = N(3.0, 0.1)
        assert abs(x).d(x) == pytest.approx(1.0)


class TestComparisons:
    def test_equal_and_not_equal(self):
        x = N(3.0, 0.1)
        assert x == x  # ruff: ignore[comparison-with-itself]
        assert x - x == 0
        assert x != N(4.0, 0.1)

    def test_ordering(self):
        small, large = N(1.0, 0.1), N(5.0, 0.1)
        assert small < large
        assert large > small
        assert small <= large
        assert large >= small
        assert small <= small  # ruff: ignore[comparison-with-itself]
        assert large >= large  # ruff: ignore[comparison-with-itself]

    def test_bool(self):
        x = N(3.0, 0.1)
        assert bool(x) is True
        # x - x is identically zero; two independent variables that merely
        # share a distribution are not.
        assert bool(x - x) is False
        assert bool(N(3.0, 0.1) - N(3.0, 0.1)) is True

    def test_hash_is_identity_based(self):
        x = N(3.0, 0.1)
        assert hash(x) == hash(x)
        assert hash(x) != hash(N(3.0, 0.1))

    def test_derived_functions_are_hashable(self):
        """UncertainFunction defines its own __hash__, separate from the
        one UncertainVariable overrides."""
        x = N(3.0, 0.1)
        derived = x * 2
        assert hash(derived) == hash(derived)
        assert hash(derived) != hash(x)
        assert len({derived, x}) == 2


class TestDerivativeAccessors:
    @staticmethod
    def _model():
        x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])
        x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])
        x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])
        return x1, x2, x3, (x1 * x2**2) / (15 * (1.5 + x3))

    def test_first_derivatives(self):
        x1, x2, x3, z = self._model()
        assert z.d(x1) == pytest.approx(1369.0 / 30.0)
        assert z.d(x2) == pytest.approx(59.2)
        assert z.d(x3) == pytest.approx(-547.6)

    def test_second_and_cross_derivatives(self):
        x1, x2, x3, z = self._model()
        assert z.d2(x2) == pytest.approx(1.6)
        assert z.d2c(x1, x3) == pytest.approx(-1369.0 / 60.0)
        # Order must not matter.
        assert z.d2c(x3, x1) == pytest.approx(z.d2c(x1, x3))

    def test_dict_forms(self):
        x1, _, _, z = self._model()
        assert isinstance(z.d(), dict)
        assert isinstance(z.d2(), dict)
        assert isinstance(z.d2c(), dict)
        assert x1 in z.d()

    def test_unknown_variable_has_zero_derivative(self):
        _, _, _, z = self._model()
        other = N(1.0, 0.1)
        assert z.d(other) == 0.0
        assert z.d2(other) == 0.0
        assert z.d2c(other, other) == 0.0

    def test_gradient_and_hessian(self):
        x1, x2, x3, z = self._model()
        grad = z.gradient([x1, x2, x3])
        assert grad == pytest.approx([1369.0 / 30.0, 59.2, -547.6])

        hess = z.hessian([x1, x2, x3])
        assert len(hess) == 3
        assert all(len(row) == 3 for row in hess)
        assert hess[1][1] == pytest.approx(1.6)
        assert hess[0][2] == pytest.approx(hess[2][0])


class TestReporting:
    @staticmethod
    def _model():
        x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])
        x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])
        x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])
        return x1, x2, x3, (x1 * x2**2) / (15 * (1.5 + x3))

    def test_moments_list_and_index(self):
        *_, z = self._model()
        assert z.moments() == pytest.approx(
            [z.mean, z.var, z.skew, z.kurt], rel=1e-12
        )
        for i in range(4):
            assert z.moments(i) == pytest.approx(z.moments()[i])

    @pytest.mark.parametrize("idx", [-1, 4, 99])
    def test_moments_rejects_bad_index(self, idx):
        *_, z = self._model()
        with pytest.raises(ValueError, match="idx must be"):
            z.moments(idx)

    def test_std_is_root_variance(self):
        *_, z = self._model()
        assert z.std == pytest.approx(math.sqrt(z.var))

    def test_str_and_repr(self):
        *_, z = self._model()
        assert str(z).startswith("uv(")
        assert repr(z) == str(z)

    def test_constant_renders_as_a_bare_number(self):
        """A constant has no input variables at all; the moment kernel must
        still cope rather than being handed zero-length arrays."""
        constant = UncertainFunction(4.0)
        assert "uv(" not in str(constant)
        assert constant.mean == pytest.approx(4.0)
        assert constant.var == pytest.approx(0.0)
        assert constant.skew == pytest.approx(0.0)
        assert constant.kurt == pytest.approx(0.0)

    def test_constant_from_to_uncertain_func_is_usable(self):
        assert str(to_uncertain_func(7)) == str(7.0)

    def test_describe(self, capsys):
        *_, z = self._model()
        z.describe()
        out = capsys.readouterr().out
        assert "SOERP Uncertain Value" in out
        assert "Skewness Coefficient" in out

    def test_error_components_by_variable(self):
        x1, x2, x3, z = self._model()
        comps = z.error_components()
        assert set(comps) == {x1, x2, x3}
        assert all(math.isfinite(v) for v in comps.values())

    def test_error_components_as_equation_terms(self):
        x1, _, _, z = self._model()
        lc, qc, cp = z.error_components(as_eq_terms=True)
        assert x1 in lc
        assert x1 in qc
        # Cross-product keys are stored in both orders.
        assert any(pair[::-1] in cp for pair in cp)

    def test_error_components_pprint_by_variable(self, capsys):
        *_, z = self._model()
        assert z.error_components(pprint=True) is None
        assert "COMPOSITE VARIABLE ERROR COMPONENTS" in capsys.readouterr().out

    def test_error_components_pprint_as_equation_terms(self, capsys):
        *_, z = self._model()
        assert z.error_components(pprint=True, as_eq_terms=True) is None
        out = capsys.readouterr().out
        assert "LINEAR ERROR COMPONENTS" in out
        assert "QUADRATIC ERROR COMPONENTS" in out
        assert "CROSS-PRODUCT ERROR COMPONENTS" in out
