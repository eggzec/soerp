"""
Created on Tue Apr  9 15:48:17 2013

Overview
--------
The ``soerp`` package is the python equivalent of N. D. Cox's original SOERP
code written in Fortran. See the documentation in UncertainVariable for more
details and a reference to his work.

Credits
-------
A lot of code here was inspired/evolved from the `uncertainties`_ package by
`Eric O. LEBIGOT`_. I'm grateful to him for his support and good work.

.. _uncertainties: http://pypi.python.org/pypi/uncertainties
.. _Eric O. LEBIGOT: http://www.linkedin.com/pub/eric-lebigot/22/293/277

"""

from importlib.metadata import PackageNotFoundError, version

from .distributions import (
    Beta,
    Chi2,
    Exp,
    F,
    Gamma,
    LogN,
    N,
    T,
    Tri,
    U,
    Weib,
    beta,
    chi_squared,
    exponential,
    f_distribution,
    gamma,
    log_normal,
    normal,
    student_t,
    triangular,
    uniform,
    weibull,
)
from .method_of_moments import raw2central
from .pdf_estimator import (
    PEARSON_TYPE_I,
    PEARSON_TYPE_II,
    PEARSON_TYPE_III,
    PEARSON_TYPE_IV,
    PEARSON_TYPE_V,
    PEARSON_TYPE_VI,
    PEARSON_TYPE_VII,
    compare_methods,
    cornish_fisher_distribution,
    cornish_fisher_pdf,
    cornish_fisher_quantile,
    cornish_fisher_sample,
    edgeworth_cdf,
    edgeworth_pdf,
    edgeworth_sample,
    fit_pearson,
    monte_carlo_cdf,
    monte_carlo_pdf,
    monte_carlo_quantile,
    monte_carlo_sample,
    pearson_cdf,
    pearson_pdf,
    pearson_sample,
    pearson_type_from_moments,
    plot_comparison,
)
from .statistics import correlation_matrix, covariance_matrix
from .uncertain_function import (
    CONSTANT_TYPES,
    UncertainFunction,
    _combine_op,
    _unary_op,
    make_uf_compatible_object,
    to_uncertain_func,
)
from .uncertain_variable import UncertainVariable, pdf, uv


__author__ = "Abraham Lee"

try:  # noqa: RUF067, RUF100
    __version__ = version(__name__)
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = [
    "CONSTANT_TYPES",
    "PEARSON_TYPE_I",
    "PEARSON_TYPE_II",
    "PEARSON_TYPE_III",
    "PEARSON_TYPE_IV",
    "PEARSON_TYPE_V",
    "PEARSON_TYPE_VI",
    "PEARSON_TYPE_VII",
    "Beta",
    "Chi2",
    "Exp",
    "F",
    "Gamma",
    "LogN",
    "N",
    "T",
    "Tri",
    "U",
    "UncertainFunction",
    "UncertainVariable",
    "Weib",
    "_combine_op",
    "_unary_op",
    "beta",
    "chi_squared",
    "compare_methods",
    "cornish_fisher_distribution",
    "cornish_fisher_pdf",
    "cornish_fisher_quantile",
    "cornish_fisher_sample",
    "correlation_matrix",
    "covariance_matrix",
    "edgeworth_cdf",
    "edgeworth_pdf",
    "edgeworth_sample",
    "exponential",
    "f_distribution",
    "fit_pearson",
    "gamma",
    "log_normal",
    "make_uf_compatible_object",
    "monte_carlo_cdf",
    "monte_carlo_pdf",
    "monte_carlo_quantile",
    "monte_carlo_sample",
    "normal",
    "pdf",
    "pearson_cdf",
    "pearson_pdf",
    "pearson_sample",
    "pearson_type_from_moments",
    "plot_comparison",
    "raw2central",
    "student_t",
    "to_uncertain_func",
    "triangular",
    "uniform",
    "uv",
    "weibull",
]
