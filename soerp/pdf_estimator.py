"""PDF Estimation Module.

This module provides multiple methods for estimating probability
density functions from statistical moments.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.stats as ss


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class DistributionMoments:
    """Moments for PDF estimation."""

    mean: float
    var: float
    skew: float = 0.0
    kurt: float = 3.0


@dataclass
class DistributionParams:
    """Parameters for distribution sampling."""

    mean: float
    std: float
    skew: float = 0.0
    kurt: float = 3.0


def _cf_quantile(z: float, dm: DistributionMoments) -> float:
    """Apply Cornish-Fisher expansion to approximate quantile.

    Parameters
    ----------
    z : float
        Standard normal quantile.
    dm : DistributionMoments
        Distribution moments.

    Returns
    -------
    float
        Approximate quantile.
    """
    excess_kurt = dm.kurt - 3
    z_cf = (
        z
        + (z**2 - 1) * dm.skew / 6
        + (z**3 - 3 * z) * excess_kurt / 24
        - (2 * z**3 - 5 * z) * dm.skew**2 / 36
    )
    return dm.mean + np.sqrt(dm.var) * z_cf


def cornish_fisher_quantile(
    z: float, mean: float, std: float, skew: float, kurt: float
) -> float:
    """Apply Cornish-Fisher expansion to approximate quantiles.

    Parameters
    ----------
    z : float
        Standard normal quantile.
    mean : float
        Distribution mean.
    std : float
        Distribution standard deviation.
    skew : float
        Distribution skewness.
    kurt : float
        Distribution kurtosis.

    Returns
    -------
    float
        Approximate quantile.

    Raises
    ------
    ValueError
        If std <= 0.

    Examples
    --------
    >>> cornish_fisher_quantile(-1.96, 10.0, 2.0, 0.5, 4.0)
    6.189...
    """
    if std <= 0:
        raise ValueError("Standard deviation must be positive")
    dm = DistributionMoments(mean, std**2, skew, kurt)
    return _cf_quantile(z, dm)


def cornish_fisher_inverse(
    x: float, params: DistributionParams, max_iter: int = 50, tol: float = 1e-8
) -> float:
    """Find quantile z for x using inverse Cornish-Fisher.

    Parameters
    ----------
    x : float
        The value to find quantile for.
    params : DistributionParams
        Distribution parameters.
    max_iter : int, optional
        Maximum iterations. Default 50.
    tol : float, optional
        Convergence tolerance. Default 1e-8.

    Returns
    -------
    float
        Approximate standard normal quantile.

    Raises
    ------
    ValueError
        If std <= 0.
    """
    if params.std <= 0:
        raise ValueError("Standard deviation must be positive")
    dm = DistributionMoments(
        params.mean, params.std**2, params.skew, params.kurt
    )
    z = (x - params.mean) / params.std
    for _ in range(max_iter):
        x_est = _cf_quantile(z, dm)
        if abs(x_est - x) < tol:
            break
        z = (x - params.mean) / params.std + (x - x_est) / params.std * 0.5
    return z


def cornish_fisher_pdf(
    x: float | np.ndarray,
    mean: float,
    var: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float | np.ndarray:
    """Estimate PDF using Cornish-Fisher expansion.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate.
    mean : float
        First moment (mean).
    var : float
        Second moment (variance).
    skew : float, optional
        Third standardized moment. Default is 0.
    kurt : float, optional
        Fourth standardized moment. Default is 3.

    Returns
    -------
    float or ndarray
        PDF values.

    Raises
    ------
    ValueError
        If var <= 0.
    """
    if var <= 0:
        raise ValueError("Variance must be positive")
    std = np.sqrt(var)
    excess_kurt = kurt - 3
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    z_values = (x_arr - mean) / std
    adjusted_z = (
        z_values
        + (z_values**2 - 1) * skew / 6
        + (z_values**3 - 3 * z_values) * excess_kurt / 24
        - (2 * z_values**3 - 5 * z_values) * skew**2 / 36
    )
    pdf_vals = np.exp(-0.5 * adjusted_z**2) / (std * np.sqrt(2 * np.pi))
    if np.isscalar(x):
        return float(pdf_vals[0])
    return pdf_vals


def cornish_fisher_sample(
    n: int, mean: float, std: float, skew: float, kurt: float
) -> np.ndarray:
    """Generate samples using Cornish-Fisher expansion.

    Parameters
    ----------
    n : int
        Number of samples.
    mean : float
        Distribution mean.
    std : float
        Distribution standard deviation.
    skew : float
        Distribution skewness.
    kurt : float
        Distribution kurtosis.

    Returns
    -------
    ndarray
        Array of n samples.
    """
    z_normal = np.random.standard_normal(n)
    dm = DistributionMoments(mean, std**2, skew, kurt)
    return np.array([_cf_quantile(z, dm) for z in z_normal])


def cornish_fisher_distribution(
    mean: float, std: float, skew: float, kurt: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Create PDF function using Cornish-Fisher.

    Parameters
    ----------
    mean : float
        Distribution mean.
    std : float
        Distribution standard deviation.
    skew : float
        Distribution skewness.
    kurt : float
        Distribution kurtosis.

    Returns
    -------
    callable
        PDF function.
    """
    return lambda x_vals: cornish_fisher_pdf(x_vals, mean, std**2, skew, kurt)


def edgeworth_pdf(  # noqa PLR0913, PLR0917
    x: float | np.ndarray,
    mean: float,
    var: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    terms: int = 4,
) -> float | np.ndarray:
    """Estimate PDF using Edgeworth series (backward-compatible version).

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate.
    mean : float
        Distribution mean.
    var : float
        Distribution variance.
    skew : float, optional
        Distribution skewness. Default is 0.
    kurt : float, optional
        Distribution kurtosis. Default is 3.
    terms : int, optional
        Number of terms (1-4). Default is 4.

    Returns
    -------
    float or ndarray
        PDF values.

    Raises
    ------
    ValueError
        If var <= 0 or terms invalid.
    """
    if var <= 0:
        raise ValueError("Variance must be positive")
    if not 1 <= terms <= 4:
        raise ValueError("Terms must be between 1 and 4")
    std = np.sqrt(var)
    excess_kurt = kurt - 3
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    z_values = (x_arr - mean) / std
    phi_z = np.exp(-0.5 * z_values**2) / np.sqrt(2 * np.pi)
    result = phi_z.copy()
    if terms >= 2 and skew != 0:
        H3 = z_values**3 - 3 * z_values
        result += (skew / 6) * H3 * phi_z
    if terms >= 3 and excess_kurt != 0:
        H4 = z_values**4 - 6 * z_values**2 + 3
        result += (excess_kurt / 24) * H4 * phi_z
    if terms >= 4 and skew != 0:
        H6 = z_values**6 - 15 * z_values**4 + 45 * z_values**2 - 15
        result += (skew**2 / 72) * H6 * phi_z
    result /= std
    if np.isscalar(x):
        return float(result[0])
    return result


def edgeworth_cdf(
    x: float | np.ndarray, moments: DistributionMoments, terms: int = 4
) -> float | np.ndarray:
    """Estimate CDF using Edgeworth series.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate.
    moments : DistributionMoments
        Distribution moments.
    terms : int, optional
        Number of terms (1-4). Default is 4.

    Returns
    -------
    float or ndarray
        CDF values.

    Raises
    ------
    ValueError
        If var <= 0.
    """
    if moments.var <= 0:
        raise ValueError("Variance must be positive")
    std = np.sqrt(moments.var)
    excess_kurt = moments.kurt - 3
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    z_values = (x_arr - moments.mean) / std
    result = ss.norm.cdf(z_values)
    phi_z = np.exp(-0.5 * z_values**2) / np.sqrt(2 * np.pi)
    if terms >= 2 and moments.skew != 0:
        H2 = z_values**2 - 1
        result += (moments.skew / 6) * H2 * phi_z
    if terms >= 3 and excess_kurt != 0:
        H3 = z_values**3 - 3 * z_values
        result += (excess_kurt / 24) * H3 * phi_z
    if terms >= 4 and moments.skew != 0:
        H5 = z_values**5 - 10 * z_values**3 + 15 * z_values
        result += (moments.skew**2 / 72) * H5 * phi_z
    if np.isscalar(x):
        return float(result[0])
    return result


def edgeworth_sample(
    n: int, params: DistributionParams, terms: int = 4
) -> np.ndarray:
    """Generate samples using Edgeworth series.

    Parameters
    ----------
    n : int
        Number of samples.
    params : DistributionParams
        Distribution parameters.
    terms : int, optional
        Number of terms. Default is 4.

    Returns
    -------
    ndarray
        Array of n samples.
    """
    moments = DistributionMoments(
        params.mean, params.std**2, params.skew, params.kurt
    )
    samples = []
    for _ in range(n):
        u = np.random.uniform()
        z = ss.norm.ppf(u)
        for _ in range(50):
            cdf = edgeworth_cdf(params.mean + params.std * z, moments, terms)
            if abs(cdf - u) < 1e-8:
                break
            pdf = edgeworth_pdf(params.mean + params.std * z, moments, terms)
            if pdf > 1e-10:
                z -= (cdf - u) / pdf
        samples.append(params.mean + params.std * z)
    return np.array(samples)


PEARSON_TYPE_I = 1
PEARSON_TYPE_II = 2
PEARSON_TYPE_III = 3
PEARSON_TYPE_IV = 4
PEARSON_TYPE_V = 5
PEARSON_TYPE_VI = 6
PEARSON_TYPE_VII = 7


def pearson_type_from_moments(moments: DistributionMoments) -> int:
    """Determine Pearson type from moments.

    Parameters
    ----------
    moments : DistributionMoments
        Distribution moments.

    Returns
    -------
    int
        Pearson type (1-7).

    Raises
    ------
    ValueError
        If var <= 0.
    """
    if moments.var <= 0:
        raise ValueError("Variance must be positive")
    excess_kurt = moments.kurt - 3
    beta1 = moments.skew**2
    beta2 = excess_kurt
    kappa = beta2 - beta1 - 1

    if beta1 == 0 and beta2 < 0:
        return PEARSON_TYPE_VII
    if beta1 == 0:
        if beta2 == 0:
            return 0
        return PEARSON_TYPE_IV if moments.skew > 0 else PEARSON_TYPE_VII
    if kappa < 0:
        return PEARSON_TYPE_VI if beta1 >= 1 else PEARSON_TYPE_IV
    return 0


def fit_pearson(moments: DistributionMoments) -> ss.rv_continuous:
    """Fit Pearson distribution to moments.

    Parameters
    ----------
    moments : DistributionMoments
        Distribution moments.

    Returns
    -------
    rv_continuous
        Fitted distribution.

    Raises
    ------
    ValueError
        If var <= 0.
    """
    if moments.var <= 0:
        raise ValueError("Variance must be positive")
    excess_kurt = moments.kurt - 3
    if abs(moments.skew) < 1e-6 and abs(excess_kurt) < 1e-6:
        return ss.norm(loc=moments.mean, scale=np.sqrt(moments.var))
    ptype = pearson_type_from_moments(
        moments.mean, moments.var, moments.skew, moments.kurt
    )
    std = np.sqrt(moments.var)
    result = ss.norm(loc=moments.mean, scale=std)
    if ptype in {PEARSON_TYPE_I, PEARSON_TYPE_II}:
        a_param = std * np.sqrt(np.pi / 2) * abs(moments.skew) / np.sqrt(2)
        b_param = std**2 * (1 - np.pi / 4)
        if a_param > 0 and b_param > 0:
            result = ss.beta(
                a_param / b_param,
                (1 - a_param / b_param),
                loc=moments.mean - std * np.sqrt(np.pi / 2) * moments.skew,
                scale=2 * std,
            )
    elif ptype == PEARSON_TYPE_III:
        scale = (
            std * moments.skew if moments.skew > 0 else std * (-moments.skew)
        )
        result = ss.gamma(moments.skew**-2, scale=scale)
    elif ptype == PEARSON_TYPE_IV and excess_kurt != 0:
        m = (moments.skew**2 + 1) ** 1.5 / excess_kurt
        if m > 0:
            result = ss.chi2(1 / m)
    elif ptype == PEARSON_TYPE_VI:
        df = 1 + 2 * (1 - moments.skew**2) / moments.skew**2
        if df > 0:
            result = ss.chi2(df)
    elif ptype == PEARSON_TYPE_VII:
        df = 3 + 6 / excess_kurt if excess_kurt != 0 else 5
        if df > 0:
            result = ss.t(df, loc=moments.mean, scale=std)
    return result


def pearson_pdf(
    x: float | np.ndarray,
    mean: float,
    var: float,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float | np.ndarray:
    """Evaluate Pearson PDF fitted to moments (backward-compatible version).

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate.
    mean : float
        Distribution mean.
    var : float
        Distribution variance.
    skew : float, optional
        Distribution skewness. Default is 0.
    kurt : float, optional
        Distribution kurtosis. Default is 3.

    Returns
    -------
    float or ndarray
        PDF values.
    """
    moments = DistributionMoments(mean, var, skew, kurt)
    dist = fit_pearson(moments)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = dist.pdf(x_arr)
    if np.isscalar(x):
        return float(result[0])
    return result


def pearson_cdf(
    x: float | np.ndarray, moments: DistributionMoments
) -> float | np.ndarray:
    """Evaluate Pearson CDF fitted to moments.

    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate.
    moments : DistributionMoments
        Distribution moments.

    Returns
    -------
    float or ndarray
        CDF values.
    """
    dist = fit_pearson(moments)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = dist.cdf(x_arr)
    if np.isscalar(x):
        return float(result[0])
    return result


def pearson_sample(n: int, params: DistributionParams) -> np.ndarray:
    """Generate samples from Pearson distribution.

    Parameters
    ----------
    n : int
        Number of samples.
    params : DistributionParams
        Distribution parameters.

    Returns
    -------
    ndarray
        Array of n samples.
    """
    moments = DistributionMoments(
        params.mean, params.std**2, params.skew, params.kurt
    )
    return fit_pearson(moments).rvs(n)


def monte_carlo_sample(
    n: int, params: DistributionParams, method: str = "cornish_fisher"
) -> np.ndarray:
    """Generate Monte Carlo samples.

    Parameters
    ----------
    n : int
        Number of samples.
    params : DistributionParams
        Distribution parameters.
    method : str, optional
        Method: "cornish_fisher", "edgeworth", "pearson", "normal".
        Default is "cornish_fisher".

    Returns
    -------
    ndarray
        Samples.

    Raises
    ------
    ValueError
        If method unknown.
    """
    method_lower = method.lower()
    if method_lower == "normal":
        return np.random.normal(params.mean, params.std, n)
    elif method_lower == "cornish_fisher":
        return cornish_fisher_sample(
            n, params.mean, params.std, params.skew, params.kurt
        )
    elif method_lower == "edgeworth":
        return edgeworth_sample(n, params)
    elif method_lower == "pearson":
        return pearson_sample(
            n, params.mean, params.std, params.skew, params.kurt
        )
    else:
        raise ValueError(f"Unknown method: {method}")


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo methods."""

    moments: DistributionMoments
    n_samples: int = 10000
    method: str = "cornish_fisher"
    bins: int = 50


def monte_carlo_pdf(  # noqa PLR0913, PLR0917
    x: float | np.ndarray,
    mean: float,
    var: float,
    skew: float = 0.0,
    kurt: float = 3.0,
    n_samples: int = 10000,
    method: str = "cornish_fisher",
    bins: int = 50,
) -> float | np.ndarray:
    """Estimate PDF using Monte Carlo sampling (backward-compatible version).

    Parameters
    ----------
    x : float or array-like
        Points.
    mean : float
        Distribution mean.
    var : float
        Distribution variance.
    skew : float, optional
        Distribution skewness. Default is 0.
    kurt : float, optional
        Distribution kurtosis. Default is 3.
    n_samples : int, optional
        Number of samples. Default is 10000.
    method : str, optional
        Sampling method. Default is "cornish_fisher".
    bins : int, optional
        Histogram bins. Default is 50.

    Returns
    -------
    float or ndarray
        PDF values.

    Raises
    ------
    ValueError
        If var <= 0.
    """
    if var <= 0:
        raise ValueError("Variance must be positive")
    std = np.sqrt(var)
    params = DistributionParams(mean, std, skew, kurt)
    samples = monte_carlo_sample(n_samples, params, method)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    hist, bin_edges = np.histogram(samples, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    result = np.interp(x_arr, bin_centers, hist, left=0.0, right=0.0)
    if np.isscalar(x):
        return float(result[0])
    return result
    return result


@dataclass
class MonteCarloCDFConfig:
    """Configuration for Monte Carlo CDF methods."""

    moments: DistributionMoments
    n_samples: int = 10000
    method: str = "cornish_fisher"


def monte_carlo_cdf(
    x: float | np.ndarray, config: MonteCarloCDFConfig
) -> float | np.ndarray:
    """Estimate CDF using Monte Carlo sampling.

    Parameters
    ----------
    x : float or array-like
        Points.
    config : MonteCarloCDFConfig
        Monte Carlo CDF configuration.

    Returns
    -------
    float or ndarray
        CDF values.

    Raises
    ------
    ValueError
        If var <= 0.
    """
    if config.moments.var <= 0:
        raise ValueError("Variance must be positive")
    std = np.sqrt(config.moments.var)
    params = DistributionParams(
        config.moments.mean, std, config.moments.skew, config.moments.kurt
    )
    samples = monte_carlo_sample(config.n_samples, params, config.method)
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    result = np.array([np.mean(samples <= xi) for xi in x_arr])
    if np.isscalar(x):
        return float(result[0])
    return result


@dataclass
class MonteCarloQuantileConfig:
    """Configuration for Monte Carlo quantile methods."""

    moments: DistributionMoments
    n_samples: int = 10000
    method: str = "cornish_fisher"


def monte_carlo_quantile(
    p: float | np.ndarray, config: MonteCarloQuantileConfig
) -> float | np.ndarray:
    """Estimate quantiles using Monte Carlo sampling.

    Parameters
    ----------
    p : float or array-like
        Probability levels.
    config : MonteCarloQuantileConfig
        Monte Carlo quantile configuration.

    Returns
    -------
    float or ndarray
        Quantile values.
    """
    std = np.sqrt(config.moments.var)
    params = DistributionParams(
        config.moments.mean, std, config.moments.skew, config.moments.kurt
    )
    samples = monte_carlo_sample(config.n_samples, params, config.method)
    samples = np.sort(samples)
    p_arr = np.atleast_1d(np.asarray(p, dtype=float))
    result = np.quantile(samples, p_arr)
    if np.isscalar(p):
        return float(result[0])
    return result


def compare_methods(x: np.ndarray, moments: DistributionMoments) -> dict:
    """Compare all PDF methods.

    Parameters
    ----------
    x : ndarray
        Points at which to evaluate.
    moments : DistributionMoments
        Distribution moments.

    Returns
    -------
    dict
        Dictionary with PDF estimates from each method.
    """
    mc_config = MonteCarloConfig(moments)
    return {
        "cornish_fisher": cornish_fisher_pdf(
            x, moments.mean, moments.var, moments.skew, moments.kurt
        ),
        "edgeworth": edgeworth_pdf(x, moments),
        "pearson": pearson_pdf(x, moments),
        "monte_carlo": monte_carlo_pdf(x, mc_config),
    }


@dataclass
class PlotConfig:
    """Configuration for plotting."""

    moments: DistributionMoments
    x_range: tuple = (-5, 5)
    n_points: int = 500


def plot_comparison(config: PlotConfig) -> None:
    """Plot comparison of PDF methods.

    Parameters
    ----------
    config : PlotConfig
        Plot configuration.

    Raises
    ------
    ImportError
        If matplotlib unavailable.
    """
    if plt is None:
        raise ImportError("matplotlib required")
    std = np.sqrt(config.moments.var)
    x_pts = np.linspace(
        config.x_range[0] * std + config.moments.mean,
        config.x_range[1] * std + config.moments.mean,
        config.n_points,
    )
    results = compare_methods(x_pts, config.moments)
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_pts, results["cornish_fisher"], label="Cornish-Fisher", linewidth=2
    )
    plt.plot(x_pts, results["edgeworth"], label="Edgeworth", linewidth=2)
    plt.plot(x_pts, results["pearson"], label="Pearson", linewidth=2)
    plt.plot(
        x_pts,
        results["monte_carlo"],
        label="Monte Carlo",
        linewidth=2,
        alpha=0.7,
    )
    plt.xlabel("x")
    plt.ylabel("PDF")
    mean_str = "mu={}, sig={:.2f}, skew={:.2f}, kurt={:.2f}"
    plt.title(
        mean_str.format(
            config.moments.mean, std, config.moments.skew, config.moments.kurt
        )
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


__all__ = [
    "PEARSON_TYPE_I",
    "PEARSON_TYPE_II",
    "PEARSON_TYPE_III",
    "PEARSON_TYPE_IV",
    "PEARSON_TYPE_V",
    "PEARSON_TYPE_VI",
    "PEARSON_TYPE_VII",
]
