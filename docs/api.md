# API Reference

soerp provides a Python implementation of the SOERP method (Cox 1979) for second-order error propagation. See the [Theory](theory.md) and [Quickstart](quickstart.md) for mathematical background and usage.

## Main Features

- Transparent calculations with automatic derivatives
- Basic NumPy support
- Nearly all standard math module functions supported via `soerp.umath` (e.g., `sin`, `exp`, `sqrt`, etc.)
- Analytical derivatives up to second order
- Easy continuous distribution constructors:
	- `normal(mu, sigma)` or `N(mu, sigma)`: Normal
	- `uniform(a, b)` or `U(a, b)`: Uniform
	- `exponential(lamda, [mu])` or `Exp(lamda, [mu])`: Exponential
	- `gamma(k, theta)` or `Gamma(k, theta)`: Gamma
	- `beta(alpha, beta, [a, b])` or `Beta(alpha, beta, [a, b])`: Beta
	- `log_normal(mu, sigma)` or `LogN(mu, sigma)`: Log-normal
	- `chi_squared(k)` or `Chi2(k)`: Chi-squared
	- `f_distribution(d1, d2)` or `F(d1, d2)`: F-distribution
	- `triangular(a, b, c)` or `Tri(a, b, c)`: Triangular
	- `student_t(v)` or `T(v)`: T-distribution
	- `weibull(lamda, k)` or `Weib(lamda, k)`: Weibull

## Core Classes and Functions

- `uv`: Uncertain variable constructor (accepts moments or a scipy.stats distribution)
- `normal`, `uniform`, `exponential`, `gamma`, `chi_squared`, ...: Distribution functions
- `N`, `U`, `Exp`, `Gamma`, `Chi2`, ...: Same functions with uppercase names
- `umath`: Math functions for uncertain variables
- `describe()`: Print mean, variance, skewness, kurtosis
- `moments()`: Return moments of a variable
- `d()`, `d2()`, `d2c()`: First and second derivatives, mixed derivatives
- `gradient()`, `hessian()`: Vector/matrix of derivatives
- `error_components(pprint=True/False)`: Variance decomposition and error component breakdown

## Example Workflows

soerp supports both direct moment input and distribution-based construction. You can:

- Create uncertain variables from moments, scipy.stats distributions, or constructors
- Combine variables using arithmetic and math functions
- Compute and print all moments, derivatives, and error components

See the [Quickstart](quickstart.md) for full code examples, including:

- Assembly stack-up
- Orifice flow
- Manufacturing tolerance stackup
- Scheduling facilities
- Two-bar truss

All examples demonstrate both moment-based and distribution-based usage, as well as advanced features like error decomposition.
