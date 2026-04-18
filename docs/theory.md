# Theory

## Overview

`soerp` implements **second-order error propagation (SOERP)**, a method for estimating the
statistical moments of a function of random variables. Given a function
$z = h(x_1, x_2, \ldots, x_n)$ where each $x_i$ is an independent random variable with
known moments, SOERP computes the first four moments of $z$ using a second-order Taylor
series approximation. These moments characterize the output distribution — its mean,
variance, skewness, and kurtosis — without requiring Monte Carlo simulation.

The method is particularly useful for:

- **Tolerance analysis**: predicting how part-to-part variation propagates to system output
- **Uncertainty analysis**: quantifying output uncertainty from imprecisely known input parameters
- **Sensitivity analysis**: identifying which input variables contribute most to output variance

---

## Second-Order Taylor Series Approximation

### System Output Polynomial

Let $h(x_1, x_2, \ldots, x_n)$ be the function of interest and let $\nu_{i1}$ denote the
expected value (mean) of the $i$-th random variable $x_i$. The function is expanded in a
second-order Taylor series about the point $(\nu_{11}, \nu_{21}, \ldots, \nu_{n1})$:

$$
h(x_1, \ldots, x_n) \approx z[\nu_{11}, \ldots, \nu_{n1}]
+ \sum_{i=1}^n \frac{\partial h}{\partial x_i}(x_i - \nu_{i1})
+ \frac{1}{2}\sum_{i=1}^n \frac{\partial^2 h}{\partial x_i^2}(x_i - \nu_{i1})^2
+ \sum_{\substack{i,j \\ i < j}} \frac{\partial^2 h}{\partial x_i \partial x_j}(x_i - \nu_{i1})(x_j - \nu_{j1})
$$

where all partial derivatives are evaluated at the mean values $\nu_{i1}$.

This expansion is written compactly as a second-order polynomial $z$ in the centered
variables $(x_i - \nu_{i1})$:

$$
z = b_0 + \sum_{i=1}^n b_i(x_i - \nu_{i1})
      + \sum_{i=1}^n b_{ii}(x_i - \nu_{i1})^2
      + \sum_{\substack{i,j \\ i < j}} b_{ij}(x_i - \nu_{i1})(x_j - \nu_{j1})
$$

### Polynomial Coefficients

The coefficients are determined directly from the partial derivatives of $h$ evaluated
at the mean values:

| Coefficient | Expression | Description |
|---|---|---|
| $b_0$ | $h(\nu_{11}, \nu_{21}, \ldots, \nu_{n1})$ | Intercept (nominal output) |
| $b_i$ | $\displaystyle\left.\frac{\partial h}{\partial x_i}\right\|_{\boldsymbol{\nu}}$ | Linear (first-order) coefficient |
| $b_{ii}$ | $\displaystyle\frac{1}{2}\left.\frac{\partial^2 h}{\partial x_i^2}\right\|_{\boldsymbol{\nu}}$ | Quadratic (second-order) coefficient |
| $b_{ij}$ | $\displaystyle\left.\frac{\partial^2 h}{\partial x_i \partial x_j}\right\|_{\boldsymbol{\nu}}$ | Cross-product (interaction) coefficient |

### Centered Output Variable

For the purpose of moment calculations, a centered output variable $y$ is defined by
subtracting the nominal output from $z$:

$$
y = z(x_1, \ldots, x_n) - z(\nu_{11}, \ldots, \nu_{n1})
$$

The central moments of $y$ (order 2 and above) are identical to those of $z$. Only
the mean differs; the mean of $z$ is recovered as:

$$
\nu_{z1} = \nu_{y1} + z(\nu_{11}, \nu_{21}, \ldots, \nu_{n1})
$$

---

## Input Variable Moments

Each input variable $x_i$ is independently distributed with central moments
$\mu_{ij}$ for $2 \le j \le 8$. The central moments are related to moments about the
origin by:

$$
\begin{align}
\mu_{i0} &= 1 \\
\mu_{i1} &= 0 \\
\mu_{i2} &= \nu_{i2} - \nu_{i1}^2 \qquad\text{(variance)} \\
\mu_{i3} &= \nu_{i3} - 3\nu_{i2}\nu_{i1} + 2\nu_{i1}^3 \\
\mu_{i4} &= \nu_{i4} - 4\nu_{i3}\nu_{i1} + 6\nu_{i2}\nu_{i1}^2 - 3\nu_{i1}^4
\end{align}
$$

A rigorous second-order propagation requires moments up through **eighth order**
($\mu_{i2}$ through $\mu_{i8}$). Truncating at fourth-order central moments produces
an approximation equivalent to the simplified formulas found in the earlier literature.

### Standardized Input Variables

It is convenient to work with standardized (transformed) variables:

$$
w_i = \frac{x_i - \nu_{i1}}{\sigma_i}
$$

where $\sigma_i$ is the standard deviation of $x_i$. Each $w_i$ then has zero mean and
unit variance, and its central moments depend only on the shape of the distribution,
not its scale. For a **normally distributed** variable, the standardized central moments are:

$$
\mu_1 = 0, \quad
\mu_2 = 1, \quad
\mu_3 = 0, \quad
\mu_4 = 3, \quad
\mu_5 = 0, \quad
\mu_6 = 15, \quad
\mu_7 = 0, \quad
\mu_8 = 105
$$

When using standardized variables, the polynomial coefficients are also rescaled.
The quadratic coefficient for $w_i$ becomes:

$$
b_{ii}' = b_{ii}\,\sigma_i^2
$$

and the cross-product coefficient for $w_i w_j$ becomes:

$$
b_{ij}' = b_{ij}\,\sigma_i\,\sigma_j
$$

---

## Moment Equations

The $k$-th moment of $y$ about the origin is:

$$
\nu_{yk} = \int \cdots \int y^k \left(\prod_{i=1}^n f_i\right) dx_1 \cdots dx_n
$$

where $f_i$ is the probability density function of $x_i$. Substituting the second-order
polynomial approximation and exploiting the independence of the $x_i$, this integral
reduces to an algebraic function of the polynomial coefficients and the central moments
of the input variables.

The first four moments are given below. These are the equations solved by SOERP.

### First Moment (Mean of $y$)

$$
\nu_{y1} = \sum_{i=1}^n b_{ii}\,\mu_{i2}
\tag{A-6}
$$

Note that the linear terms $b_i$ do not contribute to the mean because the $x_i$ are
centered at their expected values. Only the quadratic terms $b_{ii}$ shift the mean.

### Second Moment

$$
\nu_{y2} = \sum_{i=1}^n \left[b_i^2\,\mu_{i2} + 2b_i b_{ii}\,\mu_{i3} + b_{ii}^2\,\mu_{i4}\right]
         + \sum_{\substack{i,j \\ i < j}} \left[2b_{ii}b_{jj} + b_{ij}^2\right]\mu_{i2}\,\mu_{j2}
\tag{A-7}
$$

### Third Moment

$$
\begin{align}
\nu_{y3} &= \sum_{i=1}^n \left[b_i^3\,\mu_{i3} + b_{ii}^3\,\mu_{i6}
            + 3b_i^2 b_{ii}\,\mu_{i4} + 3b_i b_{ii}^2\,\mu_{i5}\right] \\
&\quad + \sum_{\substack{i,j \\ i < j}} \left[b_{ij}^2\,\mu_{i3}\,\mu_{j3}
            + 6b_i b_j b_{ij}\,\mu_{i2}\,\mu_{j2}
            + 6b_{ii}b_{jj}b_{ij}\,\mu_{i3}\,\mu_{j3}\right] \\
&\quad + \sum_{i=1}^n \sum_{\substack{j=1 \\ j \ne i}}^n
            \left[3b_{ii}^2\,\mu_{i4}\,b_{jj}\,\mu_{j2}
            + 6b_i b_{ii} b_{ij}\,\mu_{i2}\,\mu_{j3}
            + 3b_{ii}b_j^2\,\mu_{i2}\,\mu_{j2}
            + 6b_i b_{ii} b_{jj}\,\mu_{i3}\,\mu_{j2}
            + 3b_i b_{ij}^2\,\mu_{i3}\,\mu_{j2}
            + 3b_{ii}b_{ij}^2\,\mu_{i4}\,\mu_{j2}\right] \\
&\quad + \sum_{i=1}^{n-2}\sum_{j=i+1}^{n-1}\sum_{k=j+1}^{n}
            \left\{6b_{ii}b_{jj}b_{kk}
            + 6b_{ij}b_{ik}b_{jk}
            + 3\left[b_{ii}b_{jk}^2 + b_{jj}b_{ik}^2 + b_{kk}b_{ij}^2\right]
            \right\}\mu_{i2}\,\mu_{j2}\,\mu_{k2}
\tag{A-8}
\end{align}
$$

### Fourth Moment

The fourth moment involves single-variable sums, double sums over pairs $(i,j)$, triple
sums over triplets $(i,j,k)$, and quadruple sums over quadruplets $(i,j,k,m)$.

**Single-variable terms:**

$$
\sum_{i=1}^n \left[b_i^4\,\mu_{i4} + b_{ii}^4\,\mu_{i8}
+ 4b_i^3 b_{ii}\,\mu_{i5} + 4b_i b_{ii}^3\,\mu_{i7}
+ 6b_i^2 b_{ii}^2\,\mu_{i6}\right]
$$

**Pair terms** ($i < j$):

$$
\sum_{\substack{i,j \\ i<j}} \left\{
  6b_i^2 b_j^2\,\mu_{i2}\,\mu_{j2}
  + 6b_{ii}^2 b_{jj}^2\,\mu_{i4}\,\mu_{j4}
  + b_{ij}^4\,\mu_{i4}\,\mu_{j4}
  + 12b_{ij}\left[b_i^2 b_j\,\mu_{i3}\,\mu_{j2} + b_i b_j^2\,\mu_{i2}\,\mu_{j3}\right]
  + 12b_{ij}b_{ii}b_{jj}\left[b_{ii}\,\mu_{i5}\,\mu_{j3} + b_{jj}\,\mu_{i3}\,\mu_{j5}\right]
  + 12b_{ii}b_{jj}\left[b_i^2\,\mu_{i4}\,\mu_{j2} + b_j^2\,\mu_{i2}\,\mu_{j4}
                      + 2b_i b_j\,\mu_{i3}\,\mu_{j3}\right]
  + 6b_{ij}^2\left[b_i^2\,\mu_{i4}\,\mu_{j2} + b_j^2\,\mu_{i2}\,\mu_{j4}
                  + 2b_i b_j\,\mu_{i3}\,\mu_{j3}\right]
  + 6b_{ij}^2\left[b_{ii}^2\,\mu_{i6}\,\mu_{j2} + b_{jj}^2\,\mu_{i2}\,\mu_{j6}
                  + 2b_{ii}b_{jj}\,\mu_{i4}\,\mu_{j4}\right]
  + 12b_{ij}\left[b_j b_{ii}(b_j\,\mu_{i3}\,\mu_{j3} + 2b_i\,\mu_{i4}\,\mu_{j2})
               + b_i b_{jj}(b_i\,\mu_{i3}\,\mu_{j3} + 2b_j\,\mu_{i2}\,\mu_{j4})\right]
  + 12b_{ij}\left[b_i b_{jj}(b_{jj}\,\mu_{i2}\,\mu_{j5} + 2b_{ii}\,\mu_{i4}\,\mu_{j3})
               + b_j b_{ii}(b_{ii}\,\mu_{i5}\,\mu_{j2} + 2b_{jj}\,\mu_{i3}\,\mu_{j4})\right]
  + 12b_{ij}^2\left[b_{ii}(b_i\,\mu_{i5}\,\mu_{j2} + b_j\,\mu_{i4}\,\mu_{j3})
                  + b_{jj}(b_i\,\mu_{i3}\,\mu_{j4} + b_j\,\mu_{i2}\,\mu_{j5})\right]
\right\}
$$

**Cross-diagonal pair terms** ($j \ne i$):

$$
\sum_{i=1}^n \sum_{\substack{j=1 \\ j \ne i}}^n
\left\{4b_{ii}^3 b_{jj}\,\mu_{i6}\,\mu_{j2}
+ \cdots
+ 6b_i^2 b_{ij}^2\,\mu_{i4}\,\mu_{j2}\right\}
$$

**Triplet terms** ($i < j < k$):

$$
\sum_{i=1}^{n-2}\sum_{j=i+1}^{n-1}\sum_{k=j+1}^{n}
\Bigl\{
  \left[12b_{ii}^2 b_{jj}b_{kk} + 6b_{ij}^2 b_{ik}^2
        + 12b_{ii}(b_{kk}b_{ij}^2 + b_{jj}b_{ik}^2) + 6b_{ii}^2 b_{jk}^2\right]\mu_{i4}\,\mu_{j2}\,\mu_{k2}
  + \left[\text{similar terms for } (j,i,k)\text{ and }(k,i,j)\right]
  + \left[12b_{ij}^2 b_{ik}b_{jk} + 24b_{ii}b_{jj}b_{kk}b_{ij}
          + 4b_{kk}b_{ij}^3 + 24b_{ii}b_{jj}b_{ik}b_{jk}\right]\mu_{i3}\,\mu_{j3}\,\mu_{k2}
  + \left[\text{similar terms for remaining } \mu_{i3}\mu_{j2}\mu_{k3}\text{ and }\mu_{i2}\mu_{j3}\mu_{k3}\right]
  + 24\left[b_{ii}b_{jj}b_{kk} + b_{ij}b_{ik}b_{jk}\right]
     \cdot\left[b_i\,\mu_{i3}\,\mu_{j2}\,\mu_{k2}
               + b_j\,\mu_{i2}\,\mu_{j3}\,\mu_{k2}
               + b_k\,\mu_{i2}\,\mu_{j2}\,\mu_{k3}\right]
  + \mu_{i2}\,\mu_{j2}\,\mu_{k2}\Bigl[
      12\left(b_{ii}b_{jj}b_{kk}^2 + b_{ii}b_{kk}b_{jk}^2 + b_{jj}b_{kk}b_{ik}^2\right) \\
      + 6\left(b_{ii}^2 b_{jk}^2 + b_{jj}^2 b_{ik}^2 + b_{kk}^2 b_{ij}^2\right)
      + 24\left(b_{ij}b_{ik}b_j b_k + b_{ij}b_{jk}b_i b_k + b_{ik}b_{jk}b_i b_j\right)
      + 24\left(b_i b_j b_{kk}b_{ij} + b_i b_k b_{jj}b_{ik} + b_j b_k b_{ii}b_{jk}\right)
    \Bigr]
\Bigr\}
$$

**Quadruplet terms** ($i < j < k < m$):

$$
\sum_{i=1}^{n-3}\sum_{j=i+1}^{n-2}\sum_{k=j+1}^{n-1}\sum_{m=k+1}^{n}
\mu_{i2}\,\mu_{j2}\,\mu_{k2}\,\mu_{m2}
\cdot\Bigl[
  24\left(b_{ii}b_{jj}b_{kk}b_{mm}
         + b_{ij}b_{ik}b_{jm}b_{km}
         + b_{ij}b_{im}b_{jk}b_{km}
         + b_{ik}b_{im}b_{jk}b_{jm}
         + b_{ii}b_{jk}b_{jm}b_{km}
         + b_{jj}b_{ik}b_{im}b_{km}
         + b_{kk}b_{ij}b_{im}b_{jm}
         + b_{mm}b_{ij}b_{ik}b_{jk}\right) \\
  + 12\left(b_{ii}b_{jj}b_{km}^2
           + b_{ii}b_{kk}b_{jm}^2
           + b_{ii}b_{mm}b_{jk}^2
           + b_{jj}b_{kk}b_{im}^2
           + b_{jj}b_{mm}b_{ik}^2
           + b_{kk}b_{mm}b_{ij}^2\right)
  + 6\left(b_{ij}^2 b_{km}^2
          + b_{ik}^2 b_{jm}^2
          + b_{im}^2 b_{jk}^2\right)
\Bigr]
\tag{A-9}
$$

---

## Output Distribution Characterization

### Central Moments of the Output

The central moments of $y$ (order 2 and above) are computed from the moments about the
origin using the standard relations:

$$
\begin{align}
\mu_{y2} &= \nu_{y2} - \nu_{y1}^2 \qquad\text{(variance)} \\
\mu_{y3} &= \nu_{y3} - 3\nu_{y2}\nu_{y1} + 2\nu_{y1}^3 \\
\mu_{y4} &= \nu_{y4} - 4\nu_{y3}\nu_{y1} + 6\nu_{y2}\nu_{y1}^2 - 3\nu_{y1}^4
\end{align}
$$

### Skewness and Kurtosis

The shape of the output distribution is characterized by two dimensionless coefficients.

**Skewness coefficient:**

$$
\sqrt{\beta_1} = \frac{\mu_{y3}}{\mu_{y2}^{3/2}}
$$

A symmetric distribution has $\sqrt{\beta_1} = 0$. Negative values indicate left-skew;
positive values indicate right-skew.

**Kurtosis coefficient:**

$$
\beta_2 = \frac{\mu_{y4}}{\mu_{y2}^2}
$$

For the normal distribution, $\beta_2 = 3$. Values greater than 3 indicate heavier tails
(leptokurtic); values less than 3 indicate lighter tails (platykurtic). For reference,
the uniform distribution has $\beta_2 = 1.8$.

### Selecting a Probability Distribution

Given the four computed moments, an appropriate parametric distribution can be fitted to
the output. Two common families are:

- **Pearson family**: covers a wide range of shapes indexed by $\beta_1$ and $\beta_2$,
  including Types I through VII. The Pearson Type IV distribution is appropriate for
  asymmetric, heavy-tailed outputs.
- **Johnson family**: provides an alternative set of transformations to normality.

If the computed $\sqrt{\beta_1}$ is close to 0 and $\beta_2$ is close to 3, the output
is approximately normal and standard Gaussian confidence intervals apply directly.

---

## Variance Sensitivity

By running SOERP with only a single input variable at a time (setting all other $b_i$,
$b_{ii}$, and $b_{ij}$ to zero), the contribution of each variable to the total output
variance can be isolated. This identifies which inputs drive the most uncertainty and
guides where precision improvements are most cost-effective.

For the linear approximation, the variance contribution of variable $i$ is simply
$b_i^2\,\mu_{i2}$. The second-order terms add corrections from $b_{ii}$ and cross-products
$b_{ij}$, which can be significant when inputs are far from symmetric or when the function
is strongly nonlinear.
