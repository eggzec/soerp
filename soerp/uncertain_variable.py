import numpy as np
import scipy.stats as ss

from . import pdf_estimator
from .method_of_moments import raw2central
from .uncertain_function import UncertainFunction


try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib_installed = False
else:
    matplotlib_installed = True


class UncertainVariable(UncertainFunction):
    """
    UncertainVariable objects track the effects of uncertainty, characterized
    in terms of the first four standard moments of statistical distributions
    (mean, variance, skewness and kurtosis coefficients). Most texts
    only deal with first-order models, but this class uses a full second
    order model, which requires a knowledge of the first eight central moments
    of a distribution.

    Parameters
    ----------
    moments : array-like, optional
        The first eight moments (standardized) of the uncertain variable's
        underlying statistical distribution (the first two values should be the
        mean and variance)

    rv : scipy.stats.rv_continous, optional
        If supplied, the ``moments`` kwarg is ignored and the first eight
        standardized moments are calculated internally

    tag : str, optional
        A string identifier when information about this variable is printed to
        the screen

    Notes
    -----

    For a full report on the methods behind this class, see:

        N. D. Cox, "Tolerance Analysis by Computer," Journal of Quality
        Technology, Vol. 11, No. 2, 1979.

    Here are the first eight moments of some standard distributions:

        - Normal Distribution: [0, 1, 0, 3, 0, 15, 0, 105]
        - Uniform Distribution: [0, 1, 0, 1.8, 0, 3.857, 0, 9]
        - Exponential Distribution: [0, 1, 2, 9, 44, 265, 1854, 14833]

    A distribution's raw moment (moment about the origin) is defined as::

                oo
                 /
                |
           k    |   k
        E(x ) = |  x *f(x) dx
                |
               /
               -oo

    where E(...) is the expectation operator, k is the order of the moment, and
    f(x) is the probability density function (pdf) of x.

    To convert these to central moments (moment about the mean), we can simply
    use the helper function::

        >>> moments = raw2central(raw_moments)

    or we can use the mathematical definition to calculate the kth moment as::

                     oo
                      /
                     |
                k    |        k
        E((x-mu) ) = |  (x-mu) *f(x) dx
                     |
                    /
                    -oo

    This then needs to be standardized by normalizing each of the moments
    (starting with the third moment) using the standard deviation::

        >>> sd = moment[1]**0.5
        >>> moment[k] = [moment[k]/sd**(k + 1) for k in range(2, 9)]

    The ``scipy.stats`` module contains many distributions from which we can
    easily generate these moments for any distribution. Currently, only
    ``rv_continuous`` distributions are supported. It is important to follow
    the initialization syntax for creating any kind of rv_continuous object:

        - *Location* and *Scale* values must use the kwargs ``loc`` and
          ``scale``
        - *Shape* values are passed in as arguments before the location and
          scale

    The mathematical operations that can be performed on Uncertain... objects
    will work for any moments or distribution supplied, but may not be
    misleading if the supplied moments or distribution is not accurately
    defined. Here are some guidelines for creating UncertainVariable objects
    using some of the most common statistical distributions:

    +---------------------------+-----------+-----------------+-----+--------+
    | Distribution              | stats cls | args            | loc | scale  |
    |                           |           | (shape params)  |     |        |
    +===========================+===========+=================+=====+========+
    | Normal(mu, sigma)         | norm      |                 | mu  | sigma  |
    +---------------------------+-----------+-----------------+-----+--------+
    | Uniform(a, b)             | uniform   |                 | a   | b-a    |
    +---------------------------+-----------+-----------------+-----+--------+
    | Exponential(lamda)        | expon     |                 |     | 1/lam  |
    +---------------------------+-----------+-----------------+-----+--------+
    | Gamma(k, theta)           | gamma     | k               |     | theta  |
    +---------------------------+-----------+-----------------+-----+--------+
    | Beta(alpha, beta, [a, b]) | beta      | alpha, beta     | a   | b-a    |
    +---------------------------+-----------+-----------------+-----+--------+
    | Log-Normal(mu, sigma)     | lognorm   | sigma           | mu  |        |
    +---------------------------+-----------+-----------------+-----+--------+
    | Chi-Square(k)             | chi2      | k               |     |        |
    +---------------------------+-----------+-----------------+-----+--------+
    | F(d1, d2)                 | f         | d1, d2          |     |        |
    +---------------------------+-----------+-----------------+-----+--------+
    | Triangular(a, b, c)       | triang    | c               | a   | b-a    |
    +---------------------------+-----------+-----------------+-----+--------+
    | Student-T(v)              | t         | v               |     |        |
    +---------------------------+-----------+-----------------+-----+--------+
    | Weibull(lamda, k)         | exponweib | lamda, k        |     |        |
    +---------------------------+-----------+-----------------+-----+--------+

    Thus, each distribution above would have the same call signature::

        >>> import scipy.stats as ss
        >>> ss.your_dist_here(args,loc=loc,scale=scale)

    Convenient constructors have been created to make assigning these
    distributions easier. They follow the parameter notation found in the
    respective Wikipedia articles:

    +---------------------------+--------------------------------------------+
    | MCERP Distibution         | Wikipedia page                             |
    +===========================+============================================+
    | N(mu, sigma)              | wikipedia.org/wiki/Normal_distribution     |
    +---------------------------+--------------------------------------------+
    | U(a, b)                   | wikipedia.org/wiki/Uniform_distribution_   |
    |                           | (continuous)                               |
    +---------------------------+--------------------------------------------+
    | Exp(lamda, [mu])          | wikipedia.org/wiki/Exponential_distribution|
    +---------------------------+--------------------------------------------+
    | Gamma(k, theta)           | wikipedia.org/wiki/Gamma_distribution      |
    +---------------------------+--------------------------------------------+
    | Beta(alpha, beta, [a, b]) | wikipedia.org/wiki/Beta_distribution       |
    +---------------------------+--------------------------------------------+
    | LogN(mu, sigma)           | wikipedia.org/wiki/Log-normal_distribution |
    +---------------------------+--------------------------------------------+
    | X2(df)                    | wikipedia.org/wiki/Chi-squared_distribution|
    +---------------------------+--------------------------------------------+
    | F(dfn, dfd)               | wikipedia.org/wiki/F-distribution          |
    +---------------------------+--------------------------------------------+
    | Tri(a, b, c)              | wikipedia.org/wiki/Triangular_distribution |
    +---------------------------+--------------------------------------------+
    | T(df)                     | wikipedia.org/wiki/Student's_t-distribution|
    +---------------------------+--------------------------------------------+
    | Weib(lamda, k)            | wikipedia.org/wiki/Weibull_distribution    |
    +---------------------------+--------------------------------------------+


    Thus, the following are equivalent::

        >>> x = uv([10, 1, 0, 3, 0, 15, 0, 105])
        >>> x = uv(rv=ss.norm(loc=10, scale=1))
        >>> x = N(10, 1)

    Examples
    --------
    Using the first eight distribution moments::

        >>> x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])  # normally distributed
        >>> x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])  # normally distributed
        >>> x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833]) # exp. distributed
        >>> Z = (x1*x2**2)/(15*(1.5 + x3))
        >>> Z
        uv(1176.45, 99699.6822919, 0.708013052954, 6.16324345122)

    The result shows the mean, variance, and standardized skewness and kurtosis
    of the output variable Z.

    Same example, but now using ``scipy.stats`` objects::

        >>> import scipy.stats as ss
        >>> x1 = uv(rv=ss.norm(loc=24, scale=1))  # normally distributed
        >>> x2 = uv(rv=ss.norm(loc=37, scale=4))  # normally distributed
        >>> x3 = uv(rv=ss.expon(scale=0.5))  # exponentially distributed

    Or using the convenient distribution constructors::

        >>> x1 = N(24, 1)
        >>> x2 = N(37, 4)
        >>> x3 = Exp(2)

    The results may be slightly different from using the moments manually since
    moment calculations can suffer from numerical errors during the integration
    of the expectation equations above, but they will be close enough.

    Basic math operations may be applied to distributions, where all
    statistical calculations are performed using method of moments. Built-in
    trig-, logarithm-, etc. functions should be used when possible since they
    support both scalar values and uncertain objects.

    At any time, the 8 standardized moments of variables (or the 4 that result
    from calculations) can be retrieved using::

        >>> x1.moments()
        [24.0, 1.0, 0.0, 3.0, 0.0, 15.0, 0.0, 105.0]

    Or any moment can be accessed directly by specifying its index::

        >>> Z.moments(1)  # variance
        99699.6822919

    Important
    ---------

    One final thing to note is that some answers suffer from the use of a
    second-order approximation to the method of moment equations. For example,
    the equation f(x) = x*sin(x) has this issue::

        >>> x = N(0, 1)  # standard normal distribution
        >>> x*sin(x)
        uv(1.0, 2.0, 2.82842712475, 15.0)

    This is the precise answer for f(x) = x**2, which just so happens to be the
    second-order Taylor series approximation of x*sin(x). The correct answer
    for [mean,variance,skewness,kurtosis] here can be calculated by::

        >>> mu = 0.0
        >>> sigma = 1.0
        >>> n = ss.norm(loc=mu, scale=sigma)
        >>> rm = [n.dist.expect(lambda x: (x*math.sin(x))**k, loc=mu,
        ...       scale=sigma) for k in (1, 2, 3, 4)]
        >>> cm = raw2central(rm)
        >>> mean = rm[0]
        >>> var = cm[1]
        >>> std = var**0.5
        >>> skew = cm[2]/std**3
        >>> kurt = cm[3]/std**4
        >>> [mean, var, skew, kurt]
        [0.6065306597, 0.3351234837, 0.6539519888, 2.5584134397]

    Thus, care should be taken to make sure that the equations used are
    effectively quadratic within the respective input variable distribution
    ranges or you will see approximation errors like the example above.

    """

    def __init__(
        self,
        moments: list | None = None,
        rv: "ss.rv_continuous | None" = None,
        tag: str | None = None,
    ) -> None:
        if not moments and not rv:
            raise ValueError(
                "Either the moments must be put in manually or a "
                '"rv_continuous" object from the "scipy.stats" '
                "module must be supplied"
            )

        if rv is not None:
            loc = rv.kwds.get("loc", 0.0)
            scale = rv.kwds.get("scale", 1.0)
            shape = rv.args

            mn = rv.mean()
            sd = rv.std()

            if shape:
                if rv.dist.numargs < 1:
                    raise ValueError(
                        "The distribution provided doesn't support"
                        " a 'shape' parameter"
                    )

                def expect(k: int) -> float:
                    return rv.dist.expect(
                        lambda x: x**k, args=shape, loc=loc, scale=scale
                    )

                raw_moments = [expect(k) for k in range(1, 9)]
                moments = raw2central(list(raw_moments))
                for k in range(2, 8):
                    moments[k] /= sd ** (k + 1)

            else:
                if rv.dist.numargs != 0:
                    raise ValueError(
                        "The distribution provided requires a third"
                        " 'shape' parameter"
                    )

                def expect(k: int) -> float:
                    return rv.dist.expect(lambda x: x**k)

                raw_moments = [expect(k) for k in range(1, 9)]
                moments = raw2central(list(raw_moments))

            moments[0] = mn  # mean
            moments[1] = sd**2  # variance

            self._dist = rv

        else:
            self._dist = None

        # Initialise as a leaf node of the AD graph.
        # The derivative of this variable with respect to itself is 1;
        # all higher derivatives are zero.
        UncertainFunction.__init__(self, moments[0], lc={}, qc={}, cp={})
        self._lc[self] = 1.0  # $\partial x/\partial x$ = 1
        self._qc[self] = 0.0  # $\partial^2 x/\partial x^2$ = 0

        self._tag = tag
        self._moments = moments

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        if self._tag is not None:
            return self._tag
        return UncertainFunction.__repr__(self)

    @property
    def mean(self) -> float:
        return self.moments(0)

    @property
    def var(self) -> float:
        return self.moments(1)

    @property
    def std(self) -> float:
        return self.var**0.5

    @property
    def skew(self) -> float:
        return self.moments(2)

    @property
    def kurt(self) -> float:
        return self.moments(3)

    def moments(self, idx: int | None = None) -> "list | float":
        if idx is not None and idx < len(self._moments):
            return self._moments[idx]
        else:
            return self._moments

    def set_mean(self, mn: float) -> None:
        """Modify the first moment via the mean"""
        self._moments[0] = mn

    def set_std(self, sd: float) -> None:
        """Modify the second moment via the standard deviation"""
        self._moments[1] = sd**2

    def set_var(self, vr: float) -> None:
        """Modify the second moment via the variance"""
        self._moments[1] = vr

    def set_skew(self, sk: float) -> None:
        """Modify the third moment via the standardized skewness coefficient"""
        self._moments[2] = sk

    def set_kurt(self, kt: float) -> None:
        """Modify the fourth moment via the standardized kurtosis coefficient"""
        self._moments[3] = kt

    def set_moments(self, m: list) -> None:
        """Modify the first eight moments of the UncertainVariable's
        distribution.

        Raises
        ------
        ValueError
            If m does not contain exactly eight values.
        """
        if len(m) != 8:
            raise ValueError("Input moments must include eight values")
        self._moments = m

    def pdf(
        self, x: "float | np.ndarray", method: str = "cornish_fisher"
    ) -> "float | np.ndarray":
        """Estimate the probability density function at given points.

        Parameters
        ----------
        x : float or array-like
            Points at which to evaluate the PDF
        method : str, optional
            Estimation method: "cornish_fisher" (default),
            "edgeworth", "pearson", or "monte_carlo".

        Raises
        ------
        ValueError
            If method unknown.

        Returns
        -------
        float or ndarray
            PDF values at the given points

        Examples
        --------
        >>> x = N(0, 1)
        >>> y = N(1, 2)
        >>> z = x + y
        >>> z.pdf(0.5)
        0.20...
        >>> z.pdf([0, 1, 2], method="edgeworth")
        array([0.24..., 0.30..., 0.17...])
        """
        mn = self.moments(0)
        vr = self.moments(1)
        sk = self.moments(2)
        kt = self.moments(3)

        if self._dist is not None:
            return self._dist.pdf(x)

        method_lower = method.lower()
        if method_lower == "cornish_fisher":
            return pdf_estimator.cornish_fisher_pdf(x, mn, vr, sk, kt)
        elif method_lower == "edgeworth":
            return pdf_estimator.edgeworth_pdf(x, mn, vr, sk, kt)
        elif method_lower == "pearson":
            return pdf_estimator.pearson_pdf(x, mn, vr, sk, kt)
        elif method_lower == "monte_carlo":
            return pdf_estimator.monte_carlo_pdf(x, mn, vr, sk, kt)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                "Use 'cornish_fisher', 'edgeworth', "
                "'pearson', or 'monte_carlo'"
            )

    if matplotlib_installed:

        def plot(
            self, vals: "np.ndarray | list | None" = None, **kwargs: float
        ) -> None:
            """Plot the distribution of the input variable.

            NOTE: If a scipy distribution was provided, use its exact PDF.
            Otherwise, use the Cornish-Fisher approximation from moments.
            """
            mn = self.moments(0)
            vr = self.moments(1)
            sk = self.moments(2)
            kt = self.moments(3)
            std = np.sqrt(vr)

            if self._dist is not None:
                if vals is None:
                    low = self._dist.ppf(0.0001)
                    high = self._dist.ppf(0.9999)
                else:
                    low = min(vals)
                    high = max(vals)
                vals = np.linspace(low, high, 500)
                plt.plot(vals, self._dist.pdf(vals), **kwargs)
            else:
                if vals is None:
                    quantiles = np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
                    x_vals = np.array([
                        _cornish_fisher_quantile(q, mn, std, sk, kt)
                        for q in quantiles
                    ])
                    low = min(x_vals)
                    high = max(x_vals)
                else:
                    low = min(vals)
                    high = max(vals)

                x_plot = np.linspace(
                    low - (high - low) * 0.1, high + (high - low) * 0.1, 500
                )
                y = pdf(x_plot, mn, vr, sk, kt)
                plt.plot(x_plot, y, **kwargs)

            plt.xlim(low - (high - low) * 0.1, high + (high - low) * 0.1)
            plt.show()


def _cornish_fisher_quantile(
    z: float, mean: float, std: float, skew: float, kurt: float
) -> float:
    r"""Apply Cornish-Fisher expansion to approximate quantiles from moments.

    Parameters
    ----------
    z : float
        Standard normal quantile
    mean : float
        Distribution mean
    std : float
        Distribution standard deviation
    skew : float
        Distribution skewness
    kurt : float
        Distribution kurtosis

    Returns
    -------
    float
        Approximate quantile from the distribution
    """
    excess_kurt = kurt - 3
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * excess_kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    return mean + std * z_cf


def pdf(
    x: float | np.ndarray,
    mean: float,
    var: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float | np.ndarray:
    r"""Estimate probability density function from moments using Cornish-Fisher.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the PDF
    mean : float
        First moment (mean)
    var : float
        Second moment (variance)
    skew : float, optional
        Third standardized moment (skewness). Default is 0.
    kurt : float, optional
        Fourth standardized moment (kurtosis). Default is 3.

    Returns
    -------
    float or ndarray
        PDF values at the given points

    Raises
    ------
    ValueError
        If variance is not positive
    """
    if var <= 0:
        raise ValueError("Variance must be positive")

    std = np.sqrt(var)
    excess_kurt = kurt - 3

    if isinstance(x, (int, float)):
        x = np.array([x])
    else:
        x = np.asarray(x)

    z_values = (x - mean) / std

    adjusted_z = (
        z_values
        + (z_values**2 - 1) * skew / 6
        + (z_values**3 - 3 * z_values) * excess_kurt / 24
        - (2 * z_values**3 - 5 * z_values) * skew**2 / 36
    )

    adjusted_x = mean + std * adjusted_z

    pdf_values = np.exp(-0.5 * ((adjusted_x - mean) / std) ** 2) / (
        std * np.sqrt(2 * np.pi)
    )

    if isinstance(x, np.ndarray) and len(pdf_values) == 1:
        return float(pdf_values[0])
    return pdf_values


uv = UncertainVariable  # a nicer form for the user
