"""
Coverage of the convenience distribution constructors: each one must
produce the mean and variance of the distribution it names, and each one
must reject parameters outside its valid domain.
"""

import math

import pytest

from soerp import distributions as dist


class TestConstructors:
    """Mean and variance are checked against the closed-form values."""

    def test_normal(self):
        x = dist.normal(10.0, 2.0)
        assert x.mean == pytest.approx(10.0)
        assert x.var == pytest.approx(4.0)

    def test_uniform(self):
        x = dist.uniform(2.0, 6.0)
        assert x.mean == pytest.approx(4.0)
        assert x.var == pytest.approx((6.0 - 2.0) ** 2 / 12)

    def test_exponential(self):
        x = dist.exponential(2.0)  # rate lambda = 2 -> mean 1/2
        assert x.mean == pytest.approx(0.5)
        assert x.var == pytest.approx(0.25)

    def test_gamma(self):
        k, theta = 3.0, 2.0
        x = dist.gamma(k, theta)
        assert x.mean == pytest.approx(k * theta)
        assert x.var == pytest.approx(k * theta**2)

    def test_beta(self):
        a, b = 2.0, 3.0
        x = dist.beta(a, b)
        assert x.mean == pytest.approx(a / (a + b))
        assert x.var == pytest.approx(a * b / ((a + b) ** 2 * (a + b + 1)))

    def test_beta_on_a_shifted_support(self):
        x = dist.beta(2.0, 3.0, 10.0, 20.0)
        assert x.mean == pytest.approx(10.0 + 10.0 * (2.0 / 5.0))

    def test_log_normal(self):
        mu, sigma = 0.0, 0.25
        x = dist.log_normal(mu, sigma)
        assert x.mean == pytest.approx(math.exp(mu + sigma**2 / 2))

    def test_chi_squared(self):
        x = dist.chi_squared(9)
        assert x.mean == pytest.approx(9.0)
        assert x.var == pytest.approx(18.0)

    def test_f_distribution(self):
        d1, d2 = 10, 20  # d2 > 16 so the 8th moment converges
        x = dist.f_distribution(d1, d2)
        assert x.mean == pytest.approx(d2 / (d2 - 2))

    def test_triangular(self):
        a, b, c = 0.0, 4.0, 1.0
        x = dist.triangular(a, b, c)
        assert x.mean == pytest.approx((a + b + c) / 3)

    def test_student_t(self):
        x = dist.student_t(12)
        assert x.mean == pytest.approx(0.0, abs=1e-9)
        assert x.var == pytest.approx(12 / (12 - 2))

    def test_weibull(self):
        lamda, k = 1.0, 3.0
        x = dist.weibull(lamda, k)
        assert x.mean == pytest.approx(lamda * math.gamma(1 + 1 / k))


class TestAliases:
    @pytest.mark.parametrize(
        ("alias", "full"),
        [
            (dist.N, dist.normal),
            (dist.U, dist.uniform),
            (dist.Exp, dist.exponential),
            (dist.Gamma, dist.gamma),
            (dist.Beta, dist.beta),
            (dist.LogN, dist.log_normal),
            (dist.Chi2, dist.chi_squared),
            (dist.F, dist.f_distribution),
            (dist.Tri, dist.triangular),
            (dist.T, dist.student_t),
            (dist.Weib, dist.weibull),
        ],
    )
    def test_short_names_point_at_the_full_constructors(self, alias, full):
        assert alias is full


class TestParameterValidation:
    @pytest.mark.parametrize("sigma", [0.0, -1.0])
    def test_normal_rejects_non_positive_sigma(self, sigma):
        with pytest.raises(ValueError, match="Sigma must be positive"):
            dist.normal(0.0, sigma)

    @pytest.mark.parametrize(("a", "b"), [(5.0, 5.0), (6.0, 2.0)])
    def test_uniform_rejects_an_empty_support(self, a, b):
        with pytest.raises(ValueError, match="Lower bound"):
            dist.uniform(a, b)

    @pytest.mark.parametrize(
        ("k", "theta"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)]
    )
    def test_gamma_rejects_non_positive_parameters(self, k, theta):
        with pytest.raises(ValueError, match="greater than zero"):
            dist.gamma(k, theta)

    @pytest.mark.parametrize(("a", "b"), [(0.0, 1.0), (1.0, 0.0), (-2.0, 1.0)])
    def test_beta_rejects_non_positive_shapes(self, a, b):
        with pytest.raises(ValueError, match="greater than zero"):
            dist.beta(a, b)

    @pytest.mark.parametrize("sigma", [0.0, -0.5])
    def test_log_normal_rejects_non_positive_sigma(self, sigma):
        with pytest.raises(ValueError, match="Sigma must be positive"):
            dist.log_normal(0.0, sigma)

    @pytest.mark.parametrize("df", [1, 0, -3, 2.5])
    def test_chi_squared_rejects_bad_degrees_of_freedom(self, df):
        with pytest.raises(ValueError, match="DF must be an int"):
            dist.chi_squared(df)

    def test_f_distribution_rejects_bad_numerator_df(self):
        with pytest.raises(ValueError, match="d1 must be an int"):
            dist.f_distribution(1, 12)

    def test_f_distribution_rejects_bad_denominator_df(self):
        with pytest.raises(ValueError, match="d2 must be an int"):
            dist.f_distribution(20, 1)

    @pytest.mark.parametrize(
        ("a", "b", "c"), [(0.0, 4.0, 5.0), (0.0, 4.0, -1.0)]
    )
    def test_triangular_rejects_a_peak_outside_the_support(self, a, b, c):
        with pytest.raises(ValueError, match="peak must lie"):
            dist.triangular(a, b, c)

    @pytest.mark.parametrize("v", [1, 0, 2.5])
    def test_student_t_rejects_bad_degrees_of_freedom(self, v):
        with pytest.raises(ValueError, match="v must be an int"):
            dist.student_t(v)

    @pytest.mark.parametrize(
        ("lamda", "k"), [(0.0, 1.0), (1.0, 0.0), (-1.0, 2.0)]
    )
    def test_weibull_rejects_non_positive_parameters(self, lamda, k):
        with pytest.raises(ValueError, match="greater than zero"):
            dist.weibull(lamda, k)


class TestTagging:
    def test_tag_is_used_as_the_representation(self):
        x = dist.normal(1.0, 0.1, tag="width")
        assert repr(x) == "width"

    def test_without_a_tag_the_moments_are_shown(self):
        x = dist.normal(1.0, 0.1)
        assert repr(x).startswith("uv(")
