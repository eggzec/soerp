"""
Generalizes mathematical operators that work on numeric objects (from the math
module) compatible with objects with uncertainty distributions.

All functions implement exact second-order derivatives via the chain rule so
that second-order error propagation remains accurate.
"""

import math

from scipy.special import digamma, polygamma

from .uncertain_function import UncertainFunction, _combine_op, _unary_op


__author__ = "Abraham Lee"

__all__ = []

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

e = math.e
pi = math.pi

__all__ += ["e", "pi"]

# ---------------------------------------------------------------------------
# Internal helper: build an UncertainFunction from a unary operation applied
# to a value that *may* be an UncertainFunction.
# ---------------------------------------------------------------------------


def _uf_unary(
    x: UncertainFunction | float, hx: float, dh: float, d2h: float
) -> UncertainFunction | float:
    """Return _unary_op(x, hx, dh, d2h) when x is UncertainFunction,
    otherwise return hx (a plain scalar).

    Returns
    -------
    result : UncertainFunction | float
        The result of the unary operation with uncertainty propagated.
    """
    if isinstance(x, UncertainFunction):
        return _unary_op(x, hx, dh, d2h)
    return hx


def _uf_binary(  # noqa: PLR0913, PLR0917
    x: UncertainFunction | float,
    y: UncertainFunction | float,
    hx: float,
    dh_dx: float,
    dh_dy: float,
    d2h_dx2: float,
    d2h_dy2: float,
    d2h_dxy: float,
) -> UncertainFunction | float:
    """Return _combine_op(...) when at least one arg is UncertainFunction,
    otherwise return hx.

    Returns
    -------
    result : UncertainFunction | float
        The result of the binary operation with uncertainty propagated.
    """
    x_uf = isinstance(x, UncertainFunction)
    y_uf = isinstance(y, UncertainFunction)
    if not x_uf and not y_uf:
        return hx
    if not x_uf:
        x = UncertainFunction(float(x))
    if not y_uf:
        y = UncertainFunction(float(y))
    return _combine_op(x, y, hx, dh_dx, dh_dy, d2h_dx2, d2h_dy2, d2h_dxy)


# ---------------------------------------------------------------------------
# Trigonometric
# ---------------------------------------------------------------------------


def sin(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, math.sin(fx), math.cos(fx), -math.sin(fx))


__all__.append("sin")


def cos(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, math.cos(fx), -math.sin(fx), -math.cos(fx))


__all__.append("cos")


def tan(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    c = math.cos(fx)
    t = math.tan(fx)
    d1 = 1.0 / (c * c)  # $\sec^2(x)$
    d2 = 2.0 * t / (c * c)  # $2 \tan(x) \sec^2(x)$
    return _uf_unary(x, t, d1, d2)


__all__.append("tan")


def sec(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    s = 1.0 / math.cos(fx)  # sec(x)
    t = math.tan(fx)
    d1 = s * t  # sec·tan
    d2 = s * (1.0 + 2.0 * t * t)  # $\sec(1 + 2 \tan^2)$
    return _uf_unary(x, s, d1, d2)


__all__.append("sec")


def csc(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    cs = 1.0 / math.sin(fx)  # csc(x)
    ct = math.cos(fx) / math.sin(fx)  # cot(x)
    d1 = -cs * ct
    d2 = cs * (1.0 + 2.0 * ct * ct)  # $\csc(1 + 2 \cot^2)$
    return _uf_unary(x, cs, d1, d2)


__all__.append("csc")


def cot(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    c = math.cos(fx)
    s = math.sin(fx)
    ct = c / s
    cs = s / c
    d1 = -(cs * cs)  # $-\csc^2$
    d2 = 2.0 * ct * cs * cs  # $2 \cot \csc^2$
    return _uf_unary(x, ct, d1, d2)


__all__.append("cot")

# ---------------------------------------------------------------------------
# Inverse trigonometric
# ---------------------------------------------------------------------------


def asin(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = math.sqrt(1.0 - fx * fx)
    d1 = 1.0 / r
    d2 = fx / (r * r * r)
    return _uf_unary(x, math.asin(fx), d1, d2)


__all__.append("asin")


def acos(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = math.sqrt(1.0 - fx * fx)
    d1 = -1.0 / r
    d2 = -fx / (r * r * r)
    return _uf_unary(x, math.acos(fx), d1, d2)


__all__.append("acos")


def atan(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    q = 1.0 + fx * fx
    d1 = 1.0 / q
    d2 = -2.0 * fx / (q * q)
    return _uf_unary(x, math.atan(fx), d1, d2)


__all__.append("atan")


def atan2(
    y: UncertainFunction | float, x: UncertainFunction | float
) -> UncertainFunction | float:
    yv = y.x if isinstance(y, UncertainFunction) else float(y)
    xv = x.x if isinstance(x, UncertainFunction) else float(x)
    r2 = xv * xv + yv * yv
    hx = math.atan2(yv, xv)
    # $\partial/\partial y = x/r^2$, $\partial/\partial x = -y/r^2$
    dh_dy = xv / r2
    dh_dx = -yv / r2
    # $\partial^2/\partial y^2 = -2xy/r^4$,  $\partial^2/\partial x^2 = 2xy/r^4$,  $\partial^2/(\partial x \partial y) = (y^2-x^2)/r^4$ # noqa: E501
    d2h_dy2 = -2.0 * xv * yv / (r2 * r2)
    d2h_dx2 = 2.0 * xv * yv / (r2 * r2)
    d2h_dxy = (yv * yv - xv * xv) / (r2 * r2)
    return _uf_binary(y, x, hx, dh_dy, dh_dx, d2h_dy2, d2h_dx2, d2h_dxy)


__all__.append("atan2")


def acot(x: UncertainFunction | float) -> UncertainFunction | float:
    r"""Inverse cotangent: acot(x) = atan(1/x)  ($= \pi/2 - \operatorname{atan}(x)$)

    Returns
    -------
    result : UncertainFunction | float
        acot(x) with uncertainty propagated.
    """  # noqa: E501
    fx = x.x if isinstance(x, UncertainFunction) else x
    q = 1.0 + fx * fx
    d1 = -1.0 / q
    d2 = 2.0 * fx / (q * q)
    return _uf_unary(x, math.atan(1.0 / fx) if fx != 0 else math.pi / 2, d1, d2)


__all__.append("acot")


def asec(x: UncertainFunction | float) -> UncertainFunction | float:
    """Inverse secant: asec(x) = acos(1/x)

    Returns
    -------
    result : UncertainFunction | float
        asec(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = abs_(fx) * math.sqrt(fx * fx - 1.0)
    d1 = 1.0 / r
    d2 = -(2.0 * fx * fx - 1.0) / (fx * fx * (fx * fx - 1.0) ** 1.5)
    return _uf_unary(x, math.acos(1.0 / fx), d1, d2)


__all__.append("asec")


def acsc(x: UncertainFunction | float) -> UncertainFunction | float:
    """Inverse cosecant: acsc(x) = asin(1/x)

    Returns
    -------
    result : UncertainFunction | float
        acsc(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = abs_(fx) * math.sqrt(fx * fx - 1.0)
    d1 = -1.0 / r
    d2 = (2.0 * fx * fx - 1.0) / (fx * fx * (fx * fx - 1.0) ** 1.5)
    return _uf_unary(x, math.asin(1.0 / fx), d1, d2)


__all__.append("acsc")

# ---------------------------------------------------------------------------
# Hyperbolic
# ---------------------------------------------------------------------------


def sinh(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, math.sinh(fx), math.cosh(fx), math.sinh(fx))


__all__.append("sinh")


def cosh(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, math.cosh(fx), math.sinh(fx), math.cosh(fx))


__all__.append("cosh")


def tanh(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    ch = math.cosh(fx)
    th = math.tanh(fx)
    d1 = 1.0 / (ch * ch)  # $\sech^2$
    d2 = -2.0 * th / (ch * ch)  # $-2 \tanh \sech^2$
    return _uf_unary(x, th, d1, d2)


__all__.append("tanh")


def sech(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    ch = math.cosh(fx)
    th = math.tanh(fx)
    s = 1.0 / ch  # sech
    d1 = -s * th
    d2 = s * (2.0 * th * th - 1.0)
    return _uf_unary(x, s, d1, d2)


__all__.append("sech")


def csch(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    sh = math.sinh(fx)
    ct = math.cosh(fx) / sh  # coth
    cs = 1.0 / sh  # csch
    d1 = -cs * ct
    d2 = cs * (2.0 * ct * ct - 1.0)  # $= \csch(1 + 2 \csch^2)$
    return _uf_unary(x, cs, d1, d2)


__all__.append("csch")


def coth(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    sh = math.sinh(fx)
    ct = math.cosh(fx) / sh
    cs = 1.0 / sh  # csch
    d1 = -(cs * cs)  # $-\csch^2$
    d2 = 2.0 * ct * cs * cs  # $2 \coth \csch^2$
    return _uf_unary(x, ct, d1, d2)


__all__.append("coth")

# ---------------------------------------------------------------------------
# Inverse hyperbolic
# ---------------------------------------------------------------------------


def asinh(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = math.sqrt(fx * fx + 1.0)
    d1 = 1.0 / r
    d2 = -fx / (r * r * r)
    return _uf_unary(x, math.asinh(fx), d1, d2)


__all__.append("asinh")


def acosh(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = math.sqrt(fx * fx - 1.0)
    d1 = 1.0 / r
    d2 = -fx / (r * r * r)
    return _uf_unary(x, math.acosh(fx), d1, d2)


__all__.append("acosh")


def atanh(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    q = 1.0 - fx * fx
    d1 = 1.0 / q
    d2 = 2.0 * fx / (q * q)
    return _uf_unary(x, math.atanh(fx), d1, d2)


__all__.append("atanh")


def acoth(x: UncertainFunction | float) -> UncertainFunction | float:
    """Inverse hyperbolic cotangent: acoth(x) = atanh(1/x), |x| > 1

    Returns
    -------
    result : UncertainFunction | float
        acoth(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    q = 1.0 - fx * fx
    d1 = 1.0 / q
    d2 = 2.0 * fx / (q * q)
    return _uf_unary(x, math.atanh(1.0 / fx), d1, d2)


__all__.append("acoth")


def asech(x: UncertainFunction | float) -> UncertainFunction | float:
    """Inverse hyperbolic secant: asech(x) = acosh(1/x), 0 < x <= 1

    Returns
    -------
    result : UncertainFunction | float
        asech(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = math.sqrt(1.0 - fx * fx)
    d1 = -1.0 / (fx * r)
    d2 = (1.0 - 2.0 * fx * fx) / (fx * fx * r * r * r)
    return _uf_unary(x, math.acosh(1.0 / fx), d1, d2)


__all__.append("asech")


def acsch(x: UncertainFunction | float) -> UncertainFunction | float:
    """Inverse hyperbolic cosecant: acsch(x) = asinh(1/x)

    Returns
    -------
    result : UncertainFunction | float
        acsch(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    r = math.sqrt(1.0 + fx * fx)
    d1 = -1.0 / (abs_(fx) * r)
    d2 = (1.0 + 2.0 * fx * fx) / (fx * fx * r * r * r)
    return _uf_unary(x, math.asinh(1.0 / fx), d1, d2)


__all__.append("acsch")

# ---------------------------------------------------------------------------
# Exponential and logarithmic
# ---------------------------------------------------------------------------


def exp(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    ex = math.exp(fx)
    return _uf_unary(x, ex, ex, ex)


__all__.append("exp")


def expm1(x: UncertainFunction | float) -> UncertainFunction | float:
    """exp(x) - 1, accurate for small x.

    Returns
    -------
    result : UncertainFunction | float
        expm1(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    ex = math.exp(fx)  # derivative same as exp
    return _uf_unary(x, math.expm1(fx), ex, ex)


__all__.append("expm1")


def ln(x: UncertainFunction | float) -> UncertainFunction | float:
    """Natural logarithm (alias for log base e).

    Returns
    -------
    result : UncertainFunction | float
        ln(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    d1 = 1.0 / fx
    d2 = -1.0 / (fx * fx)
    return _uf_unary(x, math.log(fx), d1, d2)


__all__.append("ln")


def log(
    x: UncertainFunction | float, base: float = math.e
) -> UncertainFunction | float:
    """Logarithm to given base (default: natural log).

    Returns
    -------
    result : UncertainFunction | float
        log(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    lb = math.log(base)
    d1 = 1.0 / (fx * lb)
    d2 = -1.0 / (fx * fx * lb)
    return _uf_unary(x, math.log(fx, base), d1, d2)


__all__.append("log")


def log10(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    lb = math.log(10.0)
    d1 = 1.0 / (fx * lb)
    d2 = -1.0 / (fx * fx * lb)
    return _uf_unary(x, math.log10(fx), d1, d2)


__all__.append("log10")


def log1p(x: UncertainFunction | float) -> UncertainFunction | float:
    """log(1 + x), accurate for small x.

    Returns
    -------
    result : UncertainFunction | float
        log1p(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    q = 1.0 + fx
    d1 = 1.0 / q
    d2 = -1.0 / (q * q)
    return _uf_unary(x, math.log1p(fx), d1, d2)


__all__.append("log1p")

# ---------------------------------------------------------------------------
# Power and root
# ---------------------------------------------------------------------------


def sqrt(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    s = math.sqrt(fx)
    d1 = 0.5 / s
    d2 = -0.25 / (fx * s)  # = -1/(4 x^{3/2})
    return _uf_unary(x, s, d1, d2)


__all__.append("sqrt")


def pow_(x: UncertainFunction | float, n: float) -> UncertainFunction | float:
    """x raised to the power n (n may be any real number).

    Returns
    -------
    result : UncertainFunction | float
        x**n with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    hx = fx**n
    if abs(fx) < 1e-10:
        d1 = 0.0 if n > 1 else (1.0 if n == 1 else float("inf"))
        d2 = 0.0 if n > 2 else (1.0 if n == 2 else float("inf"))
    else:
        d1 = n * fx ** (n - 1)
        d2 = n * (n - 1) * fx ** (n - 2)
    return _uf_unary(x, hx, d1, d2)


__all__.append("pow_")

# ---------------------------------------------------------------------------
# Absolute value and rounding (piecewise constant → zero 2nd derivative)
# ---------------------------------------------------------------------------


def abs_(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    s = 1.0 if fx >= 0 else -1.0
    return _uf_unary(x, math.fabs(fx), s, 0.0)


__all__.append("abs_")


def fabs(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    s = 1.0 if fx >= 0 else -1.0
    return _uf_unary(x, math.fabs(fx), s, 0.0)


__all__.append("fabs")


def ceil(x: UncertainFunction | float) -> UncertainFunction | float:
    """Ceiling: piecewise constant, treated as identity for uncertainty.

    Returns
    -------
    result : UncertainFunction | float
        ceil(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, float(math.ceil(fx)), 0.0, 0.0)


__all__.append("ceil")


def floor(x: UncertainFunction | float) -> UncertainFunction | float:
    """Floor: piecewise constant, treated as identity for uncertainty.

    Returns
    -------
    result : UncertainFunction | float
        floor(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, float(math.floor(fx)), 0.0, 0.0)


__all__.append("floor")


def trunc(x: UncertainFunction | float) -> UncertainFunction | float:
    """Truncate toward zero: treated as constant for uncertainty.

    Returns
    -------
    result : UncertainFunction | float
        trunc(x) with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, float(math.trunc(fx)), 0.0, 0.0)


__all__.append("trunc")


def factorial(x: UncertainFunction | float) -> float:
    """Integer factorial: returns a constant (no uncertainty propagation).

    Returns
    -------
    result : float
        The factorial of the integer value of x.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    return float(math.factorial(int(fx)))


__all__.append("factorial")

# ---------------------------------------------------------------------------
# Angle conversion (linear → constant derivative)
# ---------------------------------------------------------------------------


def degrees(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, math.degrees(fx), 180.0 / math.pi, 0.0)


__all__.append("degrees")


def radians(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    return _uf_unary(x, math.radians(fx), math.pi / 180.0, 0.0)


__all__.append("radians")

# ---------------------------------------------------------------------------
# Special functions (error, gamma)
# ---------------------------------------------------------------------------

_TWO_OVER_SQRT_PI = 2.0 / math.sqrt(math.pi)


def erf(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    e2 = math.exp(-fx * fx)
    d1 = _TWO_OVER_SQRT_PI * e2
    d2 = -2.0 * fx * d1
    return _uf_unary(x, math.erf(fx), d1, d2)


__all__.append("erf")


def erfc(x: UncertainFunction | float) -> UncertainFunction | float:
    fx = x.x if isinstance(x, UncertainFunction) else x
    e2 = math.exp(-fx * fx)
    d1 = -_TWO_OVER_SQRT_PI * e2
    d2 = -2.0 * fx * d1
    return _uf_unary(x, math.erfc(fx), d1, d2)


__all__.append("erfc")


def gamma(x: UncertainFunction | float) -> UncertainFunction | float:
    r"""Gamma function $\Gamma(x)$.  Derivatives use the digamma ($\psi$) function.

    Returns
    -------
    result : UncertainFunction | float
        $\Gamma(x)$ with uncertainty propagated.
    """  # noqa: E501
    fx = x.x if isinstance(x, UncertainFunction) else x
    gx = math.gamma(fx)
    psi = float(digamma(fx))
    psi1 = float(polygamma(1, fx))
    d1 = gx * psi
    d2 = gx * (psi * psi + psi1)
    return _uf_unary(x, gx, d1, d2)


__all__.append("gamma")


def lgamma(x: UncertainFunction | float) -> UncertainFunction | float:
    r"""Natural log of the Gamma function $\ln \Gamma(x)$.

    Returns
    -------
    result : UncertainFunction | float
        $\ln \Gamma(x)$ with uncertainty propagated.
    """
    fx = x.x if isinstance(x, UncertainFunction) else x
    d1 = float(digamma(fx))
    d2 = float(polygamma(1, fx))
    return _uf_unary(x, math.lgamma(fx), d1, d2)


__all__.append("lgamma")

# ---------------------------------------------------------------------------
# Binary helpers
# ---------------------------------------------------------------------------


def hypot(
    x: UncertainFunction | float, y: UncertainFunction | float
) -> UncertainFunction | float:
    r"""Euclidean distance $\sqrt{x^2 + y^2}$.

    Returns
    -------
    result : UncertainFunction | float
        hypot(x, y) with uncertainty propagated.
    """
    xv = x.x if isinstance(x, UncertainFunction) else float(x)
    yv = y.x if isinstance(y, UncertainFunction) else float(y)
    r = math.hypot(xv, yv)
    r3 = r * r * r
    dh_dx = xv / r
    dh_dy = yv / r
    d2h_dx2 = yv * yv / r3
    d2h_dy2 = xv * xv / r3
    d2h_dxy = -xv * yv / r3
    return _uf_binary(x, y, r, dh_dx, dh_dy, d2h_dx2, d2h_dy2, d2h_dxy)


__all__.append("hypot")
