from .uncertain_function import to_uncertain_func


def covariance_matrix(nums_with_uncert: list) -> list:
    """
    Calculate the covariance matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    cov_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2 * y
        >>> covariance_matrix([x, y, z])
        [[ 0.01  0.    0.01]
         [ 0.    0.01  0.02]
         [ 0.01  0.02  0.05]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = []
    for i1, expr1 in enumerate(ufuncs):
        derivatives1 = expr1._lc  # Optimization
        vars1 = set(derivatives1)
        coefs_expr1 = []
        for _, expr2 in enumerate(ufuncs[: i1 + 1]):
            derivatives2 = expr2._lc  # Optimization
            coef = 0.0
            for v in vars1.intersection(derivatives2):
                coef += derivatives1[v] * derivatives2[v] * v.var
            coefs_expr1.append(coef)
        cov_matrix.append(coefs_expr1)

    # Symmetrize the matrix:
    for i, covariance_coefs in enumerate(cov_matrix):
        covariance_coefs.extend(
            cov_matrix[j][i] for j in range(i + 1, len(cov_matrix))
        )

    return cov_matrix


def correlation_matrix(nums_with_uncert: list) -> list:
    """
    Calculate the correlation matrix of uncertain variables, oriented by the
    order of the inputs

    Parameters
    ----------
    nums_with_uncert : array-like
        A list of variables that have an associated uncertainty

    Returns
    -------
    corr_matrix : 2d-array-like
        A nested list containing covariance values

    Example
    -------

        >>> x = N(1, 0.1)
        >>> y = N(10, 0.1)
        >>> z = x + 2 * y
        >>> correlation_matrix([x, y, z])
        [[ 1.          0.          0.4472136 ]
         [ 0.          1.          0.89442719]
         [ 0.4472136   0.89442719  1.        ]]

    """
    ufuncs = list(map(to_uncertain_func, nums_with_uncert))
    cov_matrix = covariance_matrix(ufuncs)
    corr_matrix = []
    for i1, expr1 in enumerate(ufuncs):
        row_data = []
        for i2, expr2 in enumerate(ufuncs):
            row_data.append(cov_matrix[i1][i2] / expr1.std / expr2.std)
        corr_matrix.append(row_data)
    return corr_matrix
