"""
Method-of-moments driver for SOERP.

The moment equations themselves - equations (A-6) through (A-9) of
N. D. Cox, "Tolerance Analysis by Computer", Journal of Quality
Technology, Vol. 11, No. 2, April 1979 - are implemented as FORTRAN 77
subroutines in ``_soerp.f`` and reached through the f2py-generated
``soerp._soerp`` extension. This module is the thin Python layer that
marshals arrays across that boundary and formats the results.
"""

import math

import numpy as np
from numpy.typing import NDArray

from . import _soerp


assume_linear = False  # if True, the sqc and scp parts are ignored

MAX_MOMENT = 4  # highest moment order the Cox equations provide

###############################################################################


def standard_lc(lc: NDArray, stdevs: NDArray) -> NDArray:
    """
    Standardizes the first derivatives in preparation for moment calculation.

    Parameters
    ----------
    lc : array
        The first partial derivatives of a single output.
    stdevs : array
        The standard deviations for each input

    Returns
    -------
    slc : array
        The standardized first-order derivatives
    """
    return np.array([
        coef * stdev for coef, stdev in zip(lc, stdevs, strict=False)
    ])


###############################################################################


def standard_qc(qc: NDArray, stdevs: NDArray) -> NDArray:
    """
    Standardizes the pure second derivatives in preparation for moment
    calculation.

    Parameters
    ----------
    qc : array
        The pure second partial derivatives of a single output.
    stdevs : array
        The standard deviations for each input

    Returns
    -------
    sqc : array
        The standardized pure second-order derivatives
    """
    return np.array([
        coef * stdev**2 for coef, stdev in zip(qc, stdevs, strict=False)
    ])


###############################################################################


def standard_cp(cp: NDArray, stdevs: NDArray) -> NDArray:
    """
    Standardizes the cross-product second derivatives in preparation for
    moment calculation.

    Parameters
    ----------
    cp : 2d-array
        The cross-product second-order partial derivatives matrix of a single
        output.
    stdevs : array
        The standard deviations for each input

    Returns
    -------
    scp : 2d-array
        The standardized cross-product second-order derivatives matrix
    """
    nvars = cp.shape[1]
    scp = np.empty_like(cp)
    if nvars >= 2:
        for i in range(nvars):
            for j in range(i + 1, nvars):
                scp[i, j] = cp[i, j] * stdevs[i] * stdevs[j]
                scp[j, i] = scp[i, j]
    return scp


###############################################################################


def standardize(
    lc: NDArray, qc: NDArray, cp: NDArray, stdevs: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """
    A helper function to convert normal first and second-order partial
    derivatives to "standardized" partial derivatives, in preparation for
    moment calculations.

    Parameters
    ----------
    lc : array
        The first partial derivatives of a single output.
    qc : array
        The pure second partial derivatives of a single output.
    cp : 2d-array
        The cross-product second-order partial derivatives matrix of a single
        output.
    stdevs : array
        The standard deviations for each input

    Returns
    -------
    slc : array
        The standardized first-order derivatives
    sqc : array
        The standardized pure second-order derivatives
    scp : 2d-array
        The standardized cross-product second-order derivatives matrix
    """
    slc = standard_lc(lc, stdevs)
    sqc = standard_qc(qc, stdevs)
    scp = standard_cp(cp, stdevs)
    return slc, sqc, scp


###############################################################################


def _kernel_inputs(
    slc: NDArray, sqc: NDArray, scp: NDArray, vm: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Coerce the coefficient and moment arrays into the contiguous float64
    layout the FORTRAN kernels expect, honouring ``assume_linear``.

    Returns
    -------
    inputs : tuple
        ``(lc, qc, cp, vm)``, ready to hand to ``soerp._soerp``.
    """
    lc = np.ascontiguousarray(slc, dtype=np.float64)
    qc = np.ascontiguousarray(sqc, dtype=np.float64)
    cp = np.ascontiguousarray(scp, dtype=np.float64)
    moments = np.ascontiguousarray(vm, dtype=np.float64)

    if assume_linear:
        qc = np.zeros_like(qc)
        cp = np.zeros_like(cp)

    return lc, qc, cp, moments


###############################################################################


def rawmoment(
    slc: NDArray, sqc: NDArray, scp: NDArray, vm: NDArray, k: int
) -> float:
    """
    The kth moment of the output distribution about the origin.

    The arithmetic is performed by the FORTRAN 77 routine ``RAWMOM`` in
    ``_soerp.f``, which implements equations (A-6) to (A-9) of N. D. Cox
    (1979). MODIFY THAT CODE AT YOUR OWN RISK.

    Each of the derivative components needs to be standardized prior to input
    to this function. This means multiplying them by their respective standard
    deviations, depending on the order of the derivative. Helper functions have
    been defined for this purpose (standard_lc, standard_qc, and standard_cp).
    However, this is only necessary if manually calling this function rather
    than using soerp_numeric below.

    Parameters
    ----------
    slc : array
        The standardized first derivative terms.
    sqc : array
        The standardized pure second derivative terms
    scp : 2d-array
        The standardized cross-product second derivative terms
    vm : 2d-array
        The first 9 (starting at 0) standardized distribution moments (one
        row for each input variable, corresponding to the derivative array
        order). See the documentation for ``soerp_numeric`` for more details.
    k : int
        The kth distribution moment to calculate.

    Returns
    -------
    rm : scalar
        The kth raw distribution moment

    Raises
    ------
    ValueError
        If k lies outside the range [0, 4].
    """
    if not 0 <= k <= MAX_MOMENT:
        raise ValueError(
            f"Can only calculate raw moments k = 0 to {MAX_MOMENT}"
        )
    lc, qc, cp, moments = _kernel_inputs(slc, sqc, scp, vm)
    if lc.size == 0:
        # A constant carries no variables, so y is identically zero.
        return 1.0 if k == 0 else 0.0
    return float(_soerp.rawmom(lc, qc, cp, moments, k))


###############################################################################


def centralmoment(vi: NDArray, k: int) -> float:
    """
    Converts raw distribution moments to central moments

    Parameters
    ----------
    vi : array
        The first four raw distribution moments
    k : int
        The central moment (0 to 4) to calculate (i.e., k=2 is the variance)

    Returns
    -------
    cm : scalar
        The central moment itself

    Raises
    ------
    ValueError
        If k lies outside the range [0, 4].
    """
    if not 0 <= k <= MAX_MOMENT:
        raise ValueError(
            f"Can only calculate central moments k = 0 to {MAX_MOMENT}"
        )
    supplied = np.asarray(vi, dtype=np.float64).ravel()
    vy = np.zeros(MAX_MOMENT + 1, dtype=np.float64)
    vy[: min(supplied.size, MAX_MOMENT + 1)] = supplied[: MAX_MOMENT + 1]
    return float(_soerp.cenmom(vy, k))


###############################################################################


def variance_components(
    slc: NDArray, sqc: NDArray, scp: NDArray, var_moments: NDArray, vz: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Calculate the 1st and 2nd-order output variance components for each input
    variable.

    Parameters
    ----------
    slc : array
        The standardized first derivative terms.
    sqc : array
        The standardized pure second derivative terms
    scp : 2d-array
        The standardized cross-product second derivative terms
    var_moments : 2d-array
        The first 9 (starting at 0) standardized distribution moments (one
        row for each input variable, corresponding to the derivative array
        order). See the documentation for ``soerp_numeric`` for more details.
    vz : array
        The 1st-4th central output distribution moments

    Returns
    -------
    var_comp_lc : array
        The actual variance components from the 1st-order terms
    var_comp_qc : array
        The actual variance components from the pure 2nd-order terms
    var_comp_cp : 2d-array
        The actual variance components from the cross-product 2nd-order terms
    """
    lc, qc, cp, moments = _kernel_inputs(slc, sqc, scp, var_moments)
    if lc.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros((0, 0))
    return _soerp.varcmp(lc, qc, cp, moments, vz[2])


###############################################################################


def variance_contrib(
    var_comp_lc: NDArray,
    var_comp_qc: NDArray,
    var_comp_cp: NDArray,
    vz: NDArray,
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Convert actual variance components to percent contributions (best if used
    in conjunction with ``variance_components`` function).

    Parameters
    ----------
    var_comp_lc : array
        The actual variance components from the 1st-order terms
    var_comp_qc : array
        The actual variance components from the pure 2nd-order terms
    var_comp_cp : 2d-array
        The actual variance components from the cross-product 2nd-order terms
    vz : array
        The 1st-4th central output distribution moments

    Returns
    -------
    var_contrib_lc : array
        The contribution percentage of the variance components from the
        1st-order terms
    var_contrib_qc : array
        The contribution percentage of the variance components from the pure
        2nd-order terms
    var_contrib_cp : 2d-array
        The contribution percentage of the variance components from the
        cross-product 2nd-order terms

    """
    if vz[2]:
        return (
            np.abs(var_comp_lc / vz[2]),
            np.abs(var_comp_qc / vz[2]),
            np.abs(var_comp_cp / vz[2]),
        )

    return (
        np.zeros_like(var_comp_lc),
        np.zeros_like(var_comp_qc),
        np.zeros_like(var_comp_cp),
    )


###############################################################################


def soerp_numeric(  # ruff: ignore[too-many-arguments, too-many-positional-arguments, too-many-branches, too-many-locals, too-many-statements]
    slc: NDArray,
    sqc: NDArray,
    scp: NDArray,
    var_moments: NDArray,
    func0: float,
    title: str | None = None,
    *,
    debug: bool = False,
    silent: bool = False,
) -> list:
    """
    This performs the same moment calculations, but expects that all input
    derivatives and moments have been put in standardized form. It can also
    describe the variance contributions and print out any output distribution
    information, both raw and central moments.

    Parameters
    ----------
    slc : array
        1st-order standardized derivatives (i.e., multiplied by the standard
        deviation of the related input)
    sqc : array
        2nd-order derivatives (i.e., multiplied by the standard
        deviation squared, or variance, of the related input)
    scp : 2d-array
        2nd-order cross-derivatives (i.e., multiplied by the two standard
        deviations of the related inputs)
    var_moments : 2-d array
        Standardized moments where row[i] contains the first 9 moments of
        variable x[i]. FYI: the first 3 values should always be [1, 0, 1]
    func0 : scalar
        System mean (i.e. value of the system evaluated at the means of all
        the input variables)

    Optional
    --------
    title : str
        Identifier for results that get printed to the screen
    debug : bool, false by default
        If true, all intermediate calculation results get printed to the screen
    silent : bool, false by default
        If true, nothing gets printed to the screen (overrides debug).

    Returns
    -------
    moments : list
        The first four standard moments (mean, variance, skewness and kurtosis
        coefficients)

    Example
    -------
    Example taken from the original SOERP user guide by N. D. Cox:
        >>> norm_moments = [1, 0, 1, 0, 3, 0, 15, 0, 105]
        >>> lc = [-802.65, -430.5]
        >>> qc = [205.54, 78.66]
        >>> cp = np.array([[0, -216.5], [-216.5, 0]])
        >>> vm = np.array([norm_moments, norm_moments])
        >>> f0 = 4152
        >>> soerp_numeric(
        ...     lc,
        ...     qc,
        ...     cp,
        ...     vm,
        ...     f0,
        ...     title="EXAMPLE FROM ORIGINAL SOERP USER GUIDE",
        ... )
        ********************************************************************************
        **************** SOERP: EXAMPLE FROM ORIGINAL SOERP USER GUIDE *****************
        ********************************************************************************
        Variance Contribution of lc[x0]: 66.19083%
        Variance Contribution of lc[x1]: 19.04109%
        Variance Contribution of qc[x0]: 8.68097%
        Variance Contribution of qc[x1]: 1.27140%
        Variance Contribution of cp[x0, x1]: 4.81572%
        ********************************************************************************
        MEAN-INTERCEPT (EDEL1)....................  2.8420000E+02
        MEAN......................................  4.4362000E+03
        SECOND MOMENT (EDEL2).....................  1.0540873E+06
        VARIANCE (VARDL)..........................  9.7331770E+05
        STANDARD DEVIATION (RTVAR)................  9.8656865E+02
        THIRD MOMENT (EDEL3)......................  1.4392148E+09
        THIRD CENTRAL MOMENT (MU3DL)..............  5.8640938E+08
        COEFFICIENT OF SKEWNESS SQUARED (BETA1)...  3.7293913E-01
        COEFFICIENT OF SKEWNESS (RTBT1)...........  6.1068742E-01
        FOURTH MOMENT (EDEL4).....................  5.0404781E+12
        FOURTH CENTRAL MOMENT (MU4DL).............  3.8956371E+12
        COEFFICIENT OF KURTOSIS (BETA2)...........  4.1121529E+00
        ********************************************************************************
    """  # ruff: ignore[line-too-long]
    if not silent:
        print("\n", "*" * 80)
        if title:
            print("{:*^80}".format(" SOERP: " + title + " "))

    ############################

    lc, qc, cp, moments = _kernel_inputs(slc, sqc, scp, var_moments)

    if lc.size == 0:
        # A constant: y is identically zero, so only the 0th moment is 1
        # and there are no variance components to report.
        vy = np.zeros(MAX_MOMENT + 1)
        vz = np.zeros(MAX_MOMENT + 1)
        vy[0] = vz[0] = 1.0
        vc_lc = np.zeros(0)
        vc_qc = np.zeros(0)
        vc_cp = np.zeros((0, 0))
    else:
        # A single trip across the language boundary: the raw moments, the
        # central moments and every variance component come back at once.
        vy, vz, vc_lc, vc_qc, vc_cp = _soerp.soerpm(lc, qc, cp, moments)

    if debug and not silent:
        print("*" * 80)
        for k in range(MAX_MOMENT + 1):
            print(f"Raw Moment {k}: {vy[k]}")
        print("*" * 80)
        for k in range(MAX_MOMENT + 1):
            print(f"Central Moment {k}: {vz[k]}")

    sysmean = float(vy[1] + func0)

    ############################

    vlc, vqc, vcp = variance_contrib(vc_lc, vc_qc, vc_cp, vz)
    n = len(lc)

    if not silent:
        print("*" * 80)
        for i in range(n):
            print(f"Variance Contribution of lc[x{i:d}]: {vlc[i]:7.5%}")

        for i in range(n):
            print(f"Variance Contribution of qc[x{i:d}]: {vqc[i]:7.5%}")

        for i in range(n - 1):
            for j in range(i + 1, n):
                print(
                    f"Variance Contribution of cp[x{i:d}, x{j:d}]: {vcp[i, j]:7.5%}"  # ruff: ignore[line-too-long]
                )

    ############################

    stdev = vz[2] ** (0.5)
    if stdev:
        rtbt1 = vz[3] / vz[2] ** (1.5)
        beta2 = vz[4] / vz[2] ** 2
    else:
        rtbt1 = 0.0
        beta2 = 0.0
    beta1 = rtbt1**2
    if not silent:
        print("*" * 80)
        print("MEAN-INTERCEPT (EDEL1)....................", f"{vy[1]: 8.7E}")
        print("MEAN......................................", f"{sysmean: 8.7E}")
        print("SECOND MOMENT (EDEL2).....................", f"{vy[2]: 8.7E}")
        print("VARIANCE (VARDL)..........................", f"{vz[2]: 8.7E}")
        print("STANDARD DEVIATION (RTVAR)................", f"{stdev: 8.7E}")
        print("THIRD MOMENT (EDEL3)......................", f"{vy[3]: 8.7E}")
        print("THIRD CENTRAL MOMENT (MU3DL)..............", f"{vz[3]: 8.7E}")
        print("COEFFICIENT OF SKEWNESS SQUARED (BETA1)...", f"{beta1: 8.7E}")
        print("COEFFICIENT OF SKEWNESS (RTBT1)...........", f"{rtbt1: 8.7E}")
        print("FOURTH MOMENT (EDEL4).....................", f"{vy[4]: 8.7E}")
        print("FOURTH CENTRAL MOMENT (MU4DL).............", f"{vz[4]: 8.7E}")
        print("COEFFICIENT OF KURTOSIS (BETA2)...........", f"{beta2: 8.7E}")
        print("*" * 80)

    return [sysmean, vz[2], rtbt1, beta2]


###############################################################################


def raw2central(v: list) -> list:
    """Convert raw moments (1 to len(v)) to central moments

    Returns
    -------
    central_moments : list
        The central moments corresponding to the input raw moments.
    """

    def nci(n: int, i: int) -> float:
        return math.factorial(n) / (math.factorial(i) * math.factorial(n - i))

    v = [1, *v]
    central_moments = []
    for k in range(len(v)):
        val = 0.0
        for j in range(k + 1):
            val += (-1) ** j * nci(k, j) * v[k - j] * v[1] ** j
        central_moments.append(val)

    return central_moments[1:]
