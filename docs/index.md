# soerp

**Second Order Error Propagation for Python**

---

## Overview

`soerp` is the Python implementation of the original Fortran code SOERP by N. D. Cox to apply a second-order analysis to [error propagation](http://en.wikipedia.org/wiki/Propagation_of_uncertainty) (or uncertainty analysis). The package allows you to **easily** and **transparently** track the effects of uncertainty through mathematical calculations. Advanced mathematical functions, similar to those in the standard [math](http://docs.python.org/library/math.html) module, can also be evaluated directly.

To use `soerp`, the **first eight statistical moments** of the underlying distribution are required: *mean*, *variance*, and the *standardized third through eighth moments*. These can be input manually as an array, or generated using the provided constructors or directly from `scipy.stats` distributions. The result of all calculations generates a *mean*, *variance*, and *standardized skewness and kurtosis* coefficients.

## Requirements

- [NumPy](http://www.numpy.org/)
- [SciPy](http://scipy.org)
- [Matplotlib](http://matplotlib.org/) (optional, for plotting)

## Example Usage

```python
from soerp import uv, normal, uniform, exponential, umath

# Normal distribution, mean=10, std=1
x = uv([10, 1, 0, 3, 0, 15, 0, 105])
# x = uv(rv=ss.norm(loc=10, scale=1))
x = normal(10, 1)

# Three-part assembly
x1 = normal(24, 1)
x2 = normal(37, 4)
x3 = exponential(2)
Z = (x1 * x2**2) / (15 * (1.5 + x3))
print(Z)
Z.describe()

# Moments
print(x1.moments())
print(Z.moments())

# Correlations
print(x1 - x1)
square = x1**2
print(square - x1 * x1)

# Derivatives
print(Z.d(x1))
print(Z.d2(x2))
print(Z.d2c(x1, x3))
print(Z.gradient([x1, x2, x3]))
print(Z.hessian([x1, x2, x3]))
Z.error_components(pprint=True)

# Orifice flow example
H = normal(64, 0.5)
M = normal(16, 0.1)
P = normal(361, 2)
t = normal(165, 0.5)
C = 38.4
Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
Q.describe()
```

## Main Features

1. **Transparent calculations** with automatic derivatives.
2. Basic NumPy support.
3. Nearly all standard math module functions supported via `soerp.umath`.
4. Analytical derivatives up to second order.
5. **Easy continuous distribution constructors:**
	- `normal(mu, sigma)` or `N(mu, sigma)` : Normal
	- `uniform(a, b)` or `U(a, b)` : Uniform
	- `exponential(lamda, [mu])` or `Exp(lamda, [mu])` : Exponential
	- `gamma(k, theta)` or `Gamma(k, theta)` : Gamma
	- `beta(alpha, beta, [a, b])` or `Beta(alpha, beta, [a, b])` : Beta
	- `log_normal(mu, sigma)` or `LogN(mu, sigma)` : Log-normal
	- `chi_squared(k)` or `Chi2(k)` : Chi-squared
	- `f_distribution(d1, d2)` or `F(d1, d2)` : F-distribution
	- `triangular(a, b, c)` or `Tri(a, b, c)` : Triangular
	- `student_t(v)` or `T(v)` : T-distribution
	- `weibull(lamda, k)` or `Weib(lamda, k)` : Weibull

## See Also

- [uncertainties](http://pypi.python.org/pypi/uncertainties): First-Order Error Propagation
- [mcerp](http://pypi.python.org/pypi/mcerp): Real-time Latin-Hypercube Sampling-based Monte Carlo Error Propagation

## Acknowledgements

The author thanks [Eric O. LEBIGOT](http://www.linkedin.com/pub/eric-lebigot/22/293/277) for the original [uncertainties](http://pypi.python.org/pypi/uncertainties) package, which inspired many ideas reused or evolved here. If you only need first-order uncertainty analysis, his package is an excellent alternative.

## References

- N.D. Cox, 1979, *Tolerance Analysis by Computer*, Journal of Quality Technology, Vol. 11, No. 2, pp. 80-87
