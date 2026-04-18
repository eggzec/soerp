import scipy.stats as ss

from .uncertain_variable import UncertainVariable, uv


def normal(
    mu: float, sigma: float, tag: str | None = None
) -> UncertainVariable:
    """
    A Normal (or Gaussian) random variate

    Parameters
    ----------
    mu : scalar
        The mean value of the distribution
    sigma : scalar
        The standard deviation (must be positive and non-zero)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Normal distribution.

    Raises
    ------
    ValueError
        If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    return uv(rv=ss.norm(loc=mu, scale=sigma), tag=tag)


###############################################################################


def uniform(a: float, b: float, tag: str | None = None) -> UncertainVariable:
    """
    A Uniform random variate

    Parameters
    ----------
    a : scalar
        Lower bound of the distribution support.
    b : scalar
        Upper bound of the distribution support.

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Uniform distribution.

    Raises
    ------
    ValueError
        If a >= b.
    """
    if a >= b:
        raise ValueError("Lower bound must be less than the upper bound")
    return uv(rv=ss.uniform(loc=a, scale=b - a), tag=tag)


###############################################################################


def exponential(lamda: float, tag: str | None = None) -> UncertainVariable:
    """
    An Exponential random variate

    Parameters
    ----------
    lamda : scalar
        The inverse scale (as shown on Wikipedia), FYI: mu = 1/lamda.

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with an Exponential distribution.
    """
    return uv(rv=ss.expon(scale=1.0 / lamda), tag=tag)


###############################################################################


def gamma(k: float, theta: float, tag: str | None = None) -> UncertainVariable:
    """
    A Gamma random variate

    Parameters
    ----------
    k : scalar
        The shape parameter (must be positive and non-zero)
    theta : scalar
        The scale parameter (must be positive and non-zero)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Gamma distribution.

    Raises
    ------
    ValueError
        If k <= 0 or theta <= 0.
    """
    if not (k > 0 and theta > 0):
        raise ValueError("Gamma parameters must be greater than zero")
    return uv(rv=ss.gamma(k, scale=theta), tag=tag)


###############################################################################


def beta(
    alpha: float,
    beta: float,
    a: float = 0,
    b: float = 1,
    tag: str | None = None,
) -> UncertainVariable:
    """
    A Beta random variate

    Parameters
    ----------
    alpha : scalar
        The first shape parameter
    beta : scalar
        The second shape parameter

    Optional
    --------
    a : scalar
        Lower bound of the distribution support (default=0)
    b : scalar
        Upper bound of the distribution support (default=1)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Beta distribution.

    Raises
    ------
    ValueError
        If alpha <= 0 or beta <= 0.
    """
    if not (alpha > 0 and beta > 0):
        raise ValueError("Shape parameters must be greater than zero")
    return uv(rv=ss.beta(alpha, beta, loc=a, scale=b - a), tag=tag)


###############################################################################


def log_normal(
    mu: float, sigma: float, tag: str | None = None
) -> UncertainVariable:
    """
    A Log-Normal random variate

    Parameters
    ----------
    mu : scalar
        The location parameter
    sigma : scalar
        The scale parameter (must be positive and non-zero)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Log-Normal distribution.

    Raises
    ------
    ValueError
        If sigma <= 0.
    """
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    return uv(rv=ss.lognorm(sigma, loc=mu), tag=tag)


###############################################################################


def chi_squared(df: int, tag: str | None = None) -> UncertainVariable:
    """
    A Chi-Squared random variate

    Parameters
    ----------
    df : int
        The degrees of freedom of the distribution (must be greater than one)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Chi-Squared distribution.

    Raises
    ------
    ValueError
        If df is not an int greater than 1.
    """
    if not (isinstance(df, int) and df > 1):
        raise ValueError("DF must be an int greater than 1")
    return uv(rv=ss.chi2(df), tag=tag)


###############################################################################


def f_distribution(
    d1: int, d2: int, tag: str | None = None
) -> UncertainVariable:
    """
    An F (fisher) random variate

    Parameters
    ----------
    d1 : int
        Numerator degrees of freedom
    d2 : int
        Denominator degrees of freedom

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with an F distribution.

    Raises
    ------
    ValueError
        If d1 or d2 is not an int greater than 1.
    """
    if not (isinstance(d1, int) and d1 > 1):
        raise ValueError("d1 must be an int greater than 1")
    if not (isinstance(d2, int) and d2 > 1):
        raise ValueError("d2 must be an int greater than 1")
    return uv(rv=ss.f(d1, d2), tag=tag)


###############################################################################


def triangular(
    a: float, b: float, c: float, tag: str | None = None
) -> UncertainVariable:
    """
    A triangular random variate

    Parameters
    ----------
    a : scalar
        Lower bound of the distribution support (default=0)
    b : scalar
        Upper bound of the distribution support (default=1)
    c : scalar
        The location of the triangle's peak (a <= c <= b)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Triangular distribution.

    Raises
    ------
    ValueError
        If c is not between a and b (inclusive).
    """
    if not (a <= c <= b):
        raise ValueError("peak must lie in between low and high")
    return uv(rv=ss.triang(c, loc=a, scale=b - a), tag=tag)


###############################################################################


def student_t(v: int, tag: str | None = None) -> UncertainVariable:
    """
    A Student-T random variate

    Parameters
    ----------
    v : int
        The degrees of freedom of the distribution (must be greater than one)

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Student-T distribution.

    Raises
    ------
    ValueError
        If v is not an int greater than 1.
    """
    if not (isinstance(v, int) and v > 1):
        raise ValueError("v must be an int greater than 1")
    return uv(rv=ss.t(v), tag=tag)


###############################################################################


def weibull(
    lamda: float, k: float, tag: str | None = None
) -> UncertainVariable:
    """
    A Weibull random variate

    Parameters
    ----------
    lamda : scalar
        The scale parameter
    k : scalar
        The shape parameter

    Returns
    -------
    rv : UncertainVariable
        An UncertainVariable with a Weibull distribution.

    Raises
    ------
    ValueError
        If lamda <= 0 or k <= 0.
    """
    if not (lamda > 0 and k > 0):
        raise ValueError(
            "Weibull scale and shape parameters must be greater than zero"
        )
    return uv(rv=ss.exponweib(lamda, k), tag=tag)


N = normal
U = uniform
Exp = exponential
Gamma = gamma
Beta = beta
LogN = log_normal
Chi2 = chi_squared
F = f_distribution
Tri = triangular
T = student_t
Weib = weibull
