import math
from collections.abc import Callable

import numpy as np

from .method_of_moments import (
    soerp_numeric,
    variance_components,
    variance_contrib,
)


CONSTANT_TYPES = (float, int, complex, np.number)


def to_uncertain_func(
    x: "UncertainFunction | float | int | complex",
) -> "UncertainFunction | None":
    """
    Transforms x into a constant automatically differentiated UncertainFunction
    (UF), unless it already is (in which case x is returned unchanged).

    Raises an exception unless 'x' belongs to some specific classes of
    objects that are known not to depend on UncertainFunction objects
    (which then cannot be considered as constants).

    Returns
    -------
    result : UncertainFunction or None
        The UncertainFunction wrapping x, or None if x is not a known type.
    """

    if isinstance(x, UncertainFunction):
        return x

    # ! In Python 2.6+, numbers.Number could be used instead, here:
    if isinstance(x, CONSTANT_TYPES):
        # No variable => no derivative to define:
        return UncertainFunction(x, {}, {}, {})


###############################################################################
# Second-order forward-mode automatic differentiation helpers.
#
# Each UncertainFunction stores:
#   x    - function value
#   _lc  - {var: $\partial f/\partial \text{var}$}              first-order partials                        # noqa: E501
#   _qc  - {var: $\partial^2 f/\partial \text{var}^2$}            pure second-order partials                  # noqa: E501
#   _cp  - {(var_i, var_j): $\partial^2 f/(\partial \text{var}_i \partial \text{var}_j)$}  cross second-order partials (i < j)   # noqa: E501
#
# The chain rules implemented here are exact for polynomial functions and give
# the second-order Taylor approximation for non-polynomial ones, which is
# exactly what SOERP requires.
###############################################################################


def _combine_op(  # noqa: PLR0913, PLR0914, PLR0917
    f: "UncertainFunction",
    g: "UncertainFunction",
    hx: float,
    dh_df: float,
    dh_dg: float,
    d2h_df2: float,
    d2h_dg2: float,
    d2h_dfg: float,
) -> "UncertainFunction":
    r"""
    Propagate derivatives through a binary operation h = h(f, g).

    Parameters
    ----------
    f : UncertainFunction
        First UncertainFunction input.
    g : UncertainFunction
        Second UncertainFunction input.
    hx : float
        Scalar value of h.
    dh_df : float
        $\partial h/\partial f$  (scalar, evaluated at f.x, g.x)
    dh_dg : float
        $\partial h/\partial g$
    d2h_df2 : float
        $\partial^2 h/\partial f^2$
    d2h_dg2 : float
        $\partial^2 h/\partial g^2$
    d2h_dfg : float
        $\partial^2 h/(\partial f \partial g)$

    Returns
    -------
    result : UncertainFunction
        New UncertainFunction with propagated derivatives.
    """
    all_vars = set(f._lc) | set(g._lc)
    lc, qc, cp = {}, {}, {}

    for v in all_vars:
        fi = f._lc.get(v, 0.0)
        gi = g._lc.get(v, 0.0)
        lc[v] = dh_df * fi + dh_dg * gi

    for v in all_vars:
        fi = f._lc.get(v, 0.0)
        gi = g._lc.get(v, 0.0)
        fi2 = f._qc.get(v, 0.0)
        gi2 = g._qc.get(v, 0.0)
        qc[v] = (
            d2h_df2 * fi * fi
            + dh_df * fi2
            + 2.0 * d2h_dfg * fi * gi
            + dh_dg * gi2
            + d2h_dg2 * gi * gi
        )

    vlist = list(all_vars)
    n = len(vlist)
    for i in range(n):
        vi = vlist[i]
        for j in range(i + 1, n):
            vj = vlist[j]
            fi = f._lc.get(vi, 0.0)
            fj = f._lc.get(vj, 0.0)
            gi = g._lc.get(vi, 0.0)
            gj = g._lc.get(vj, 0.0)
            fij = f._cp.get((vi, vj), f._cp.get((vj, vi), 0.0))
            gij = g._cp.get((vi, vj), g._cp.get((vj, vi), 0.0))
            cp[vi, vj] = (
                d2h_df2 * fi * fj
                + dh_df * fij
                + d2h_dfg * (fi * gj + fj * gi)
                + dh_dg * gij
                + d2h_dg2 * gi * gj
            )

    return UncertainFunction(hx, lc, qc, cp)


def _unary_op(
    f: "UncertainFunction", hx: float, dh_df: float, d2h_df2: float
) -> "UncertainFunction":
    """
    Propagate derivatives through a unary operation h = h(f).

    Parameters
    ----------
    f : UncertainFunction
        UncertainFunction input.
    hx : float
        Scalar value of h.
    dh_df : float
        dh/df  (evaluated at f.x)
    d2h_df2 : float
                $d^2h/df^2$

    Returns
    -------
    result : UncertainFunction
        New UncertainFunction with propagated derivatives.
    """
    lc = {v: dh_df * df for v, df in f._lc.items()}
    qc = {
        v: d2h_df2 * df * df + dh_df * f._qc.get(v, 0.0)
        for v, df in f._lc.items()
    }
    cp = {}
    vlist = list(f._lc.keys())
    n = len(vlist)
    for i in range(n):
        vi = vlist[i]
        for j in range(i + 1, n):
            vj = vlist[j]
            fi = f._lc[vi]
            fj = f._lc[vj]
            fij = f._cp.get((vi, vj), f._cp.get((vj, vi), 0.0))
            cp[vi, vj] = d2h_df2 * fi * fj + dh_df * fij
    return UncertainFunction(hx, lc, qc, cp)


###############################################################################


class UncertainFunction:  # noqa: PLR0904
    """
    UncertainFunction objects represent the uncertainty of a result of
    calculations with uncertain variables. Nearly all basic mathematical
    operations are supported.

    This class is mostly intended for internal use.
    """

    def __init__(
        self,
        x: float,
        lc: dict | None = None,
        qc: dict | None = None,
        cp: dict | None = None,
    ) -> None:
        r"""
        Parameters
        ----------
        x  : scalar - function value (evaluated at variable means)
        lc : dict   - {var: $\partial f/\partial \text{var}$}          first-order partials
        qc : dict   - {var: $\partial^2 f/\partial \text{var}^2$}         pure second-order partials
        cp : dict   - {(var_i, var_j): $\partial^2 f/(\partial \text{var}_i \partial \text{var}_j)$}   cross partials  (i < j ordering)
        """  # noqa: E501
        self.x = float(x)
        self._lc = lc if lc is not None else {}
        self._qc = qc if qc is not None else {}
        self._cp = cp if cp is not None else {}

    def __hash__(self) -> int:
        return id(self)

    # ------------------------------------------------------------------
    # Derivative accessors (mirror the interface previously from `ad`)
    # ------------------------------------------------------------------

    def d(self, var: "UncertainFunction | None" = None) -> "dict | float":
        r"""Return first-derivative dict, or the scalar $\partial f/\partial \text{var}$ for a given var.

        Returns
        -------
        derivative : dict or float
            First-derivative dict if var is None, else scalar $\partial f/\partial \text{var}$.
        """  # noqa: E501
        if var is None:
            return self._lc
        return self._lc.get(var, 0.0)

    def d2(self, var: "UncertainFunction | None" = None) -> "dict | float":
        r"""Return pure second-derivative dict, or scalar $\partial^2 f/\partial \text{var}^2$ for var.

        Returns
        -------
        derivative : dict or float
            Pure second-derivative dict if var is None, else $\partial^2 f/\partial \text{var}^2$.
        """  # noqa: E501
        if var is None:
            return self._qc
        return self._qc.get(var, 0.0)

    def d2c(
        self,
        var1: "UncertainFunction | None" = None,
        var2: "UncertainFunction | None" = None,
    ) -> "dict | float":
        r"""Return cross-derivative dict, or the scalar $\partial^2 f/(\partial \text{var}_1 \partial \text{var}_2)$.

        Returns
        -------
        derivative : dict or float
            Cross-derivative dict if var1 is None, else $\partial^2 f/(\partial \text{var}_1 \partial \text{var}_2)$.
        """  # noqa: E501
        if var1 is None:
            return self._cp
        return self._cp.get((var1, var2), self._cp.get((var2, var1), 0.0))

    def gradient(self, uvars: list) -> list:
        """
        Return the gradient (first derivatives) with
        respect to a list of variables.

        Parameters
        ----------
        uvars : list
            List of variables (UncertainFunction instances) to differentiate
            with respect to.

        Returns
        -------
        grad : list
            List of first derivatives [d(self)/dvar for var in uvars].
        """
        return [self.d(var) for var in uvars]

    def hessian(self, uvars: list) -> list:
        """
        Return the Hessian matrix (second derivatives) with
        respect to a list of variables.

        Parameters
        ----------
        uvars : list
            List of variables (UncertainFunction instances) to differentiate
            with respect to.

        Returns
        -------
        hess : list of lists
            Hessian matrix [[d2(self)/dvar_i dvar_j for j] for i].
        """
        n = len(uvars)
        return [
            [
                self.d2(uvars[i]) if i == j else self.d2c(uvars[i], uvars[j])
                for j in range(n)
            ]
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Statistical moment properties (via method-of-moments)
    # ------------------------------------------------------------------

    _dist = None
    _moments = None

    @property
    def mean(self) -> float:
        """Mean value as a result of an uncertainty calculation"""
        return self.moments(0)

    @property
    def var(self) -> float:
        """Variance value as a result of an uncertainty calculation"""
        return self.moments(1)

    @property
    def std(self) -> float:
        r"""
        Standard deviation value as a result of an uncertainty calculation,
        defined as::

                        ________
                std = \/variance

        """
        return self.var**0.5

    @property
    def skew(self) -> float:
        r"""
        Skewness coefficient value as a result of an uncertainty calculation,
        defined as::

              _____     m3
            \/beta1 = ------
                      std**3

        where m3 is the third central moment and std is the standard deviation
        """
        return self.moments(2)

    @property
    def kurt(self) -> float:
        """
        Kurtosis coefficient value as a result of an uncertainty calculation,
        defined as::

                          m4
                beta2 = ------
                        std**4

        where m4 is the fourth central moment and std is the standard deviation
        """
        return self.moments(3)

    def moments(self, idx: int | None = None) -> "list | float":
        """
        The first four standard moments of a distribution: mean, variance, and
        standardized skewness and kurtosis coefficients.

        Returns
        -------
        moments : list or float
            All four moments as a list, or a single moment if idx is given.

        Raises
        ------
        ValueError
            If idx is not in the range [0, 3].
        """
        slc, sqc, scp, var_moments, f0 = self._get_inputs_for_soerp()
        m = soerp_numeric(slc, sqc, scp, var_moments, f0, silent=True)
        if idx is not None:
            if not (0 <= idx <= 3):
                raise ValueError(
                    "idx must be 0, 1, 2, or 3 since only the first "
                    "four moments can be calculated"
                )
            return m[idx]
        else:
            return m

    def _to_general_representation(
        self, str_func: Callable[[float], str]
    ) -> str:
        m = self.moments()
        mn, vr, sk, kt = m[:4]
        return (
            f"uv({str_func(mn)}, {str_func(vr)}, "
            f"{str_func(sk)}, {str_func(kt)})"
            if any([vr, sk, kt])
            else str_func(mn)
        )

    def __str__(self) -> str:
        return self._to_general_representation(str)

    def __repr__(self) -> str:
        return str(self)

    def describe(self) -> None:
        """
        Cleanly show what the distribution moments are:
            - Mean, Variance, Skewness and Kurtosis Coefficients
        """
        mn, vr, sk, kt = [self.moments(i) for i in [0, 1, 2, 3]]
        s = "SOERP Uncertain Value:\n"
        s += f" > Mean................... {mn: }\n"
        s += f" > Variance............... {vr: }\n"
        s += f" > Skewness Coefficient... {sk: }\n"
        s += f" > Kurtosis Coefficient... {kt: }\n"
        print(s)

    def _get_inputs_for_soerp(self) -> tuple:
        """
        Prepare variable moments and derivatives for method-of-moments
        calculations by standardizing the derivatives (moving the distribution
        to the origin and normalising with the standard deviation).

        Returns
        -------
        inputs : tuple
            Tuple of (slc, sqc, scp, var_moments, f0).
        """
        variables = self.d().keys()
        nvar = len(variables)

        # standardize the input derivatives
        # - slc: linear terms
        # - sqc: pure quadratic terms
        # - scp: cross quadratic terms
        slc = np.array([self.d(v) * v.std for v in variables])
        sqc = np.array([0.5 * self.d2(v) * v.var for v in variables])
        scp = np.zeros((nvar, nvar))
        for i, v1 in enumerate(variables):
            for j, v2 in enumerate(variables):
                if hash(v1) != hash(v2):
                    scp[i, j] = self.d2c(v1, v2) * v1.std * v2.std
                else:
                    scp[i, j] = 0.0

        var_moments = np.array([[1, 0, 1, *v._moments[2:]] for v in variables])

        f0 = self.x  # from evaluation at input means

        return (slc, sqc, scp, var_moments, f0)

    def error_components(  # noqa: PLR0912, PLR0914, PLR0915
        self, *, pprint: bool = False, as_eq_terms: bool = False
    ) -> "dict | tuple | None":
        """
        The parts of the second order approximation of the variance function,
        returned in three pieces if ``as_eq_terms`` = True, first-order
        components, pure-quadratic components, and cross-product components,
        otherwise the error components from the linear terms are added to the
        corresponding error components from the quadratic terms. Any
        cross-product term components are divided equally between the two
        factors of the cross-product.

        Optional
        --------
        pprint : bool, default is False,
            Pretty-print the error components, showing both the component and
            the percent contribution of the component
        as_eq_terms : bool, default is False,
            True to return the error components in the form of the equation
            terms (pure linear, pure quadratic, and cross-product), where both
            orders are available for the cross-product terms (i.e., (x, y) and
            (y, x) will be returned in the cross-product terms), otherwise in
            terms of the contributing UncertainVariables.

        Returns
        -------
        err_comp : dict
            A dictionary that maps the error components to the contributing
            UncertainVariables. If ``as_eq_terms=True``, then a tuple of three
            dictionaries is returned containing the 1) linear, 2) pure
            quadratic, and 3) cross-product term contributions).

        Example
        --------
        If we had a function of two variables ended up with the linear terms
        (``as_eq_terms`` = False here), ::

            >>> lc = {x:0.5, y:0.25}

        the quadratic terms::

            >>> qc = {x:0.2, y:0.1}

        and the cross-product term::

            >>> cp = {(x, y}:0.14}

        then the variables would be given the error components like this::

            >>> lc[x] + qc[x] + 0.5*cp[(x, y)] # first variable, x
            0.77
            >>> lc[y] + qc[y] + 0.5*cp[(x, y)] # second variable, y
            0.42

        """
        variables = self.d().keys()
        slc, sqc, scp, var_moments, _ = self._get_inputs_for_soerp()
        vz = self.moments()

        # convert standardized moments back to central moments
        vz[2] *= vz[1] ** 1.5
        vz[3] *= vz[1] ** 2
        vz = [1, *vz]  # the leading 1 is required by method-of-moments
        vlc, vqc, vcp = variance_components(slc, sqc, scp, var_moments, vz)

        vc_lc = {}
        vc_qc = {}
        vc_cp = {}

        for i, v1 in enumerate(variables):
            vc_lc[v1] = vlc[i]
            vc_qc[v1] = vqc[i]
            for j, v2 in enumerate(variables):
                if i < j:
                    vc_cp[v1, v2] = vcp[i, j]
                    vc_cp[v2, v1] = vcp[i, j]

        if not as_eq_terms:
            error_wrt_var = {v: 0.0 for v in variables}
            for i, v1 in enumerate(variables):
                if v1 in vc_lc:
                    error_wrt_var[v1] += vc_lc[v1]
                    error_wrt_var[v1] += vc_qc[v1]
                    for j, v2 in enumerate(variables):
                        if i < j and (v1, v2) in vc_cp:
                            error_wrt_var[v1] += 0.5 * vc_cp[v1, v2]
                            error_wrt_var[v2] += 0.5 * vc_cp[v1, v2]
            if pprint:
                print("COMPOSITE VARIABLE ERROR COMPONENTS")
                for v in variables:
                    pct = np.abs(error_wrt_var[v] / vz[2])
                    print(f"{v} = {error_wrt_var[v]} or {pct:%}")
                print(" ")
            else:
                return error_wrt_var
        elif pprint:
            vcont_lc, vcont_qc, vcont_cp = variance_contrib(vlc, vqc, vcp, vz)
            print("*" * 65)
            print("LINEAR ERROR COMPONENTS:")
            for i, v1 in enumerate(variables):
                if v1 in vc_lc:
                    print(f"{v1} = {vc_lc[v1]} or {vcont_lc[i]:%}")
                else:
                    print(f"{v1} = {0.0} or {0.0:%}")

            print("*" * 65)
            print("QUADRATIC ERROR COMPONENTS:")
            for i, v1 in enumerate(variables):
                if v1 in vc_qc:
                    print(f"{v1} = {vc_qc[v1]} or {vcont_qc[i]:%}")
                else:
                    print(f"{v1} = {0.0} or {0.0:%}")

            print("*" * 65)
            print("CROSS-PRODUCT ERROR COMPONENTS:")
            for i, v1 in enumerate(variables):
                for j, v2 in enumerate(variables):
                    if i < j:
                        if (v1, v2) in vc_cp:
                            print(
                                f"({v1}, {v2}) = {vc_cp[v1, v2]}"
                                f" or {vcont_cp[i, j]:%}"
                            )
                        elif (v2, v1) in vc_cp:
                            print(
                                f"({v2}, {v1}) = {vc_cp[v2, v1]}"
                                f" or {vcont_cp[j, i]:%}"
                            )
                        else:
                            print(f"({v1}, {v2}) = {0.0} or {0.0:%}")
            print(" ")
        else:
            return (vc_lc, vc_qc, vc_cp)

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __add__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return _combine_op(self, val, self.x + val.x, 1.0, 1.0, 0.0, 0.0, 0.0)

    def __radd__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return val.__add__(self)

    def __mul__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        # h = f*g  →  $dh/df = g$, $dh/dg = f$, $d^2h/(df dg) = 1$, rest 0
        return _combine_op(
            self, val, self.x * val.x, val.x, self.x, 0.0, 0.0, 1.0
        )

    def __rmul__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return val.__mul__(self)

    def __sub__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return _combine_op(self, val, self.x - val.x, 1.0, -1.0, 0.0, 0.0, 0.0)

    def __rsub__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return val.__sub__(self)

    def __truediv__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        fx, gx = self.x, val.x
        hx = fx / gx
        dh_df = 1.0 / gx
        dh_dg = -fx / (gx * gx)
        d2h_dg2 = 2.0 * fx / (gx**3)
        d2h_dfg = -1.0 / (gx * gx)
        return _combine_op(self, val, hx, dh_df, dh_dg, 0.0, d2h_dg2, d2h_dfg)

    def __rtruediv__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return val.__truediv__(self)

    def __pow__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            n = float(val)
            fx = self.x
            hx = fx**n
            if abs(fx) < 1e-10:
                dh_df = 0.0 if n > 1 else (1.0 if n == 1 else float("inf"))
                d2h_df2 = 0.0 if n > 2 else (1.0 if n == 2 else float("inf"))
            else:
                dh_df = n * fx ** (n - 1)
                d2h_df2 = n * (n - 1) * fx ** (n - 2)
            return _unary_op(self, hx, dh_df, d2h_df2)
        else:
            # Both uncertain: h = f^g
            fx, gx = self.x, val.x
            hx = fx**gx
            log_fx = math.log(fx) if fx > 0 else 0.0
            dh_df = gx * fx ** (gx - 1) if fx != 0 else 0.0
            dh_dg = hx * log_fx
            d2h_df2 = gx * (gx - 1) * fx ** (gx - 2) if fx != 0 else 0.0
            d2h_dg2 = hx * log_fx**2
            d2h_dfg = (fx ** (gx - 1) * (1.0 + gx * log_fx)) if fx != 0 else 0.0
            return _combine_op(
                self, val, hx, dh_df, dh_dg, d2h_df2, d2h_dg2, d2h_dfg
            )

    def __rpow__(
        self, val: "UncertainFunction | float | int | complex"
    ) -> "UncertainFunction":
        if not isinstance(val, UncertainFunction):
            val = UncertainFunction(float(val))
        return val.__pow__(self)

    def __neg__(self) -> "UncertainFunction":
        return _unary_op(self, -self.x, -1.0, 0.0)

    def __pos__(self) -> "UncertainFunction":
        return UncertainFunction(
            self.x, dict(self._lc), dict(self._qc), dict(self._cp)
        )

    def __abs__(self) -> "UncertainFunction":
        s = 1.0 if self.x >= 0 else -1.0
        return _unary_op(self, abs(self.x), s, 0.0)

    def __eq__(self, val: object) -> bool:
        diff = self - val
        return not (diff.mean or diff.var or diff.skew or diff.kurt)

    def __ne__(self, val: object) -> bool:
        return not self.__eq__(val)

    def __lt__(self, val: "UncertainFunction | float | int | complex") -> bool:
        val = to_uncertain_func(val)
        return float(self.mean - val.mean) < 0

    def __le__(self, val: "UncertainFunction | float | int | complex") -> bool:
        return (self == val) or self < val

    def __gt__(self, val: "UncertainFunction | float | int | complex") -> bool:
        val = to_uncertain_func(val)
        return float(self.mean - val.mean) > 0

    def __ge__(self, val: "UncertainFunction | float | int | complex") -> bool:
        return (self == val) or self > val

    def __bool__(self) -> bool:
        return self != 0


def make_uf_compatible_object(tmp: "UncertainFunction") -> "UncertainFunction":
    """
    Backward-compatible shim: all arithmetic on UncertainFunction objects
    now returns UncertainFunction directly, so this is effectively a no-op.
    Kept so that external code (e.g. older umath wrappers) continues to work.

    Returns
    -------
    tmp : UncertainFunction
        The input unchanged.
    """
    return tmp
