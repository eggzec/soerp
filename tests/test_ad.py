"""
Tests for basic AD (automatic differentiation) correctness.
Verifies derivative tracking for simple hand-checkable cases.
"""

import pytest

from soerp import N, uv


class TestADCorrectness:
    def test_linear_mean_and_var(self):
        # z = a*x + b*y  →  var(z) = $a^2 \cdot \text{var}(x) + b^2 \cdot \text{var}(y)$ for independent x, y  # noqa: E501
        x = uv([2.0, 1.0, 0, 3, 0, 15, 0, 105])
        y = uv([3.0, 4.0, 0, 3, 0, 15, 0, 105])
        z = 2 * x + 3 * y
        assert z.mean == pytest.approx(2 * 2.0 + 3 * 3.0)
        assert z.var == pytest.approx(4 * 1.0 + 9 * 4.0)

    def test_product_mean(self):
        # z = x*y  →  E[z] ≈ E[x]*E[y]  (first-order)
        x = uv([3.0, 0.01, 0, 3, 0, 15, 0, 105])
        y = uv([4.0, 0.01, 0, 3, 0, 15, 0, 105])
        z = x * y
        assert z.mean == pytest.approx(12.0, rel=1e-3)

    def test_constant_propagation(self):
        x = uv([5.0, 1.0, 0, 3, 0, 15, 0, 105])
        z = x + 0  # adding zero constant should not change var
        assert z.var == pytest.approx(x.var, rel=1e-10)

    def test_subtraction_cancels(self):
        x = uv([2.0, 1.0, 0, 3, 0, 15, 0, 105])
        z = x - x  # same variable → variance = 0
        assert z.var == pytest.approx(0.0, abs=1e-10)
        assert z.mean == pytest.approx(0.0, abs=1e-10)

    def test_power_two(self):
        # z = x^2  →  mean ≈ E[x]^2 + var(x),  var ≈ 4·E[x]^2·var(x) for normal
        mn, vr = 3.0, 0.01
        x = uv([mn, vr, 0, 3, 0, 15, 0, 105])
        z = x**2
        # second-order mean correction: E[x^2] = E[x]^2 + var(x)
        assert z.mean == pytest.approx(mn**2 + vr, rel=1e-3)

    def test_first_derivative_stored(self):
        x = N(2.0, 1.0)
        y = 3 * x
        # $\partial y/\partial x$ should be 3
        assert y.d(x) == pytest.approx(3.0)

    def test_second_derivative_stored(self):
        x = N(2.0, 1.0)
        y = x**2
        # $d^2y/dx^2$ = 2
        assert y.d2(x) == pytest.approx(2.0)

    def test_cross_derivative_stored(self):
        x = N(2.0, 0.5)
        y = N(3.0, 0.5)
        z = x * y
        # $\partial^2 (xy)/(\partial x \partial y)$ = 1
        assert z.d2c(x, y) == pytest.approx(1.0)

    def test_uv_variable_moments_passthrough(self):
        moments = [5.0, 2.0, 0.5, 3.5, 0, 0, 0, 0]
        x = uv(moments)
        assert x.mean == pytest.approx(5.0)
        assert x.var == pytest.approx(2.0)
        assert x.skew == pytest.approx(0.5)
        assert x.kurt == pytest.approx(3.5)

    def test_std_property(self):
        x = uv([0.0, 4.0, 0, 3, 0, 15, 0, 105])
        assert x.std == pytest.approx(2.0)

    def test_gradient_and_hessian(self):
        # Z = (x1 * x2**2) / (15 * (1.5 + x3))
        x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])
        x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])
        x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])
        Z = (x1 * x2**2) / (15 * (1.5 + x3))

        # Test first and second derivatives
        # dZ/dx1 = 45.63333333333333
        assert Z.d(x1) == pytest.approx(45.63333333333333, rel=1e-10)
        # d^2Z/dx2^2 = 1.6
        assert Z.d2(x2) == pytest.approx(1.6, rel=1e-10)
        # d^2Z/dx1dx3 = -22.816666666666666
        assert Z.d2c(x1, x3) == pytest.approx(-22.816666666666666, rel=1e-10)

        # Test gradient
        grad = Z.gradient([x1, x2, x3])
        expected_grad = [45.63333333333333, 59.199999999999996, -547.6]
        assert all(
            pytest.approx(g, rel=1e-10) == e
            for g, e in zip(grad, expected_grad, strict=True)
        )

        # Test hessian
        hess = Z.hessian([x1, x2, x3])
        expected_hess = [
            [0.0, 2.466666666666667, -22.816666666666666],
            [2.466666666666667, 1.6, -29.6],
            [-22.816666666666666, -29.6, 547.6],
        ]
        for row, expected_row in zip(hess, expected_hess, strict=True):
            assert all(
                pytest.approx(v, rel=1e-10) == e
                for v, e in zip(row, expected_row, strict=True)
            )
