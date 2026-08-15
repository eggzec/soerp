"""
Coverage of every function exported by ``soerp.umath``.

Each one is exercised twice: once with a plain float, which must pass
straight through to the underlying math function, and once with an
uncertain input, which must propagate derivatives. The reference values
come from the standard library rather than from soerp, so a wrong formula
shows up as a mismatch rather than agreeing with itself.
"""

import math

import pytest

from soerp import N, umath


def _sec(x):
    return 1.0 / math.cos(x)


def _csc(x):
    return 1.0 / math.sin(x)


def _cot(x):
    return 1.0 / math.tan(x)


def _sech(x):
    return 1.0 / math.cosh(x)


def _csch(x):
    return 1.0 / math.sinh(x)


def _coth(x):
    return 1.0 / math.tanh(x)


# (name, evaluation point, reference implementation)
UNARY = [
    ("sin", 0.7, math.sin),
    ("cos", 0.7, math.cos),
    ("tan", 0.7, math.tan),
    ("sec", 0.7, _sec),
    ("csc", 0.7, _csc),
    ("cot", 0.7, _cot),
    ("asin", 0.4, math.asin),
    ("acos", 0.4, math.acos),
    ("atan", 0.7, math.atan),
    ("acot", 1.7, lambda x: math.atan(1.0 / x)),
    ("asec", 1.7, lambda x: math.acos(1.0 / x)),
    ("acsc", 1.7, lambda x: math.asin(1.0 / x)),
    ("sinh", 0.7, math.sinh),
    ("cosh", 0.7, math.cosh),
    ("tanh", 0.7, math.tanh),
    ("sech", 0.7, _sech),
    ("csch", 0.7, _csch),
    ("coth", 0.7, _coth),
    ("asinh", 0.7, math.asinh),
    ("acosh", 1.7, math.acosh),
    ("atanh", 0.4, math.atanh),
    ("acoth", 1.7, lambda x: math.atanh(1.0 / x)),
    ("asech", 0.4, lambda x: math.acosh(1.0 / x)),
    ("acsch", 1.7, lambda x: math.asinh(1.0 / x)),
    ("exp", 0.7, math.exp),
    ("expm1", 0.7, math.expm1),
    ("ln", 1.7, math.log),
    ("log", 1.7, math.log),
    ("log10", 1.7, math.log10),
    ("log1p", 0.7, math.log1p),
    ("sqrt", 1.7, math.sqrt),
    ("erf", 0.7, math.erf),
    ("erfc", 0.7, math.erfc),
    ("gamma", 1.7, math.gamma),
    ("lgamma", 1.7, math.lgamma),
    ("degrees", 0.7, math.degrees),
    ("radians", 0.7, math.radians),
    ("abs_", -1.7, abs),
    ("fabs", -1.7, math.fabs),
]

# Piecewise-constant functions: value only, derivatives are zero.
STEPWISE = [("ceil", 3.2, 4.0), ("floor", 3.7, 3.0), ("trunc", 3.7, 3.0)]


@pytest.mark.parametrize(
    ("name", "x0", "ref"), UNARY, ids=[c[0] for c in UNARY]
)
class TestUnaryFunctions:
    def test_scalar_passthrough(self, name, x0, ref):
        assert getattr(umath, name)(x0) == pytest.approx(ref(x0))

    def test_uncertain_value_at_the_mean(self, name, x0, ref):
        x = N(x0, 1e-4)
        r = getattr(umath, name)(x)
        assert r.x == pytest.approx(ref(x0))

    def test_uncertain_propagates_a_derivative(self, name, x0, ref):
        x = N(x0, 1e-4)
        r = getattr(umath, name)(x)
        assert math.isfinite(r.d(x))
        assert r.d(x) != 0.0
        assert r.var > 0


@pytest.mark.parametrize(
    ("name", "x0", "want"), STEPWISE, ids=[c[0] for c in STEPWISE]
)
class TestStepwiseFunctions:
    def test_scalar(self, name, x0, want):
        assert getattr(umath, name)(x0) == pytest.approx(want)

    def test_uncertain_has_no_spread(self, name, x0, want):
        r = getattr(umath, name)(N(x0, 0.01))
        assert r.x == pytest.approx(want)
        assert r.var == pytest.approx(0.0)


class TestPow:
    def test_scalar(self):
        assert umath.pow_(1.7, 2.0) == pytest.approx(1.7**2)

    def test_uncertain(self):
        x = N(1.7, 0.01)
        r = umath.pow_(x, 3.0)
        assert r.x == pytest.approx(1.7**3)
        assert r.d(x) == pytest.approx(3 * 1.7**2)
        assert r.d2(x) == pytest.approx(6 * 1.7)

    def test_near_zero_base_is_finite(self):
        r = umath.pow_(N(0.0, 1e-3), 3.0)
        assert math.isfinite(r.x)


class TestFactorial:
    def test_scalar(self):
        assert umath.factorial(5) == pytest.approx(120.0)

    def test_uncertain_collapses_to_a_constant(self):
        assert umath.factorial(N(5.0, 0.01)) == pytest.approx(120.0)


class TestBinaryFunctions:
    """Covers every mix of uncertain and plain arguments."""

    def test_atan2_both_scalar(self):
        assert umath.atan2(1.0, 1.0) == pytest.approx(math.pi / 4)

    def test_atan2_both_uncertain(self):
        y, x = N(1.0, 0.01), N(2.0, 0.01)
        r = umath.atan2(y, x)
        assert r.x == pytest.approx(math.atan2(1.0, 2.0))
        assert r.d(y) == pytest.approx(2.0 / 5.0)
        assert r.d(x) == pytest.approx(-1.0 / 5.0)

    def test_atan2_uncertain_numerator(self):
        y = N(1.0, 0.01)
        r = umath.atan2(y, 2.0)
        assert r.x == pytest.approx(math.atan2(1.0, 2.0))
        assert r.var > 0

    def test_atan2_uncertain_denominator(self):
        x = N(2.0, 0.01)
        r = umath.atan2(1.0, x)
        assert r.x == pytest.approx(math.atan2(1.0, 2.0))
        assert r.var > 0

    def test_hypot_both_scalar(self):
        assert umath.hypot(3.0, 4.0) == pytest.approx(5.0)

    def test_hypot_both_uncertain(self):
        x, y = N(3.0, 0.01), N(4.0, 0.01)
        r = umath.hypot(x, y)
        assert r.x == pytest.approx(5.0)
        assert r.d(x) == pytest.approx(3.0 / 5.0)
        assert r.d(y) == pytest.approx(4.0 / 5.0)

    def test_hypot_mixed(self):
        x = N(3.0, 0.01)
        assert umath.hypot(x, 4.0).x == pytest.approx(5.0)
        assert umath.hypot(3.0, N(4.0, 0.01)).x == pytest.approx(5.0)


class TestExports:
    def test_every_exported_name_is_present(self):
        for name in umath.__all__:
            assert hasattr(umath, name), name

    def test_constants(self):
        assert umath.pi == pytest.approx(math.pi)
        assert umath.e == pytest.approx(math.e)
