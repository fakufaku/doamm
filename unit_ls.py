import numpy as np


def eval_secular(x, w, b):
    """
    Evaluate the value of the secular equation and its first derivative at x

    Parameters
    ----------
    x: float, shape (...)
        variable
    w: array_like, shape (..., D)
        eigenvalues in ascending order
    b: array_like, shape (..., D)
        offset
    """

    b_sq = b ** 2
    wpx = 1.0 / (w + x[..., None])
    wpx_sq = wpx ** 2

    f = np.sum(b_sq * wpx_sq, axis=-1) - 1
    df = -2.0 * np.sum(b_sq * wpx_sq * wpx, axis=-1)

    return f, df


def root_secular(w, b, n_iter=10, tol=None):
    """
    The Newton-Raphson root finding applied to secular equation
    """

    # apply normalization
    w_max = w[..., -1]
    w_ = w / w_max[..., None]
    b_ = b / w_max[..., None]

    # initialization as zero of second order polynomial around the smallest eigenvalue
    x = np.abs(b_[..., 0]) - w_[..., 0]

    # the lower bound is -min w
    lower_bound = -w_[..., 0]

    for epoch in range(n_iter):

        # compute the new estimate with Newton-Raphson
        f, df = eval_secular(x, w_, b_)
        new_x = x - f / df

        # protect with bisection
        I = new_x < lower_bound
        nI = np.logical_not(I)
        x[I] = 0.5 * (x[I] + lower_bound[I])
        x[nI] = new_x[nI]

        if np.max(np.abs(f)) < tol:
            break

    return (
        w_max * x,  # rescale before returning
        epoch,  # number of iterations for info
    )


def unit_ls(A, b, weights=None, tol=1e-8, max_iter=1000):
    """
    Solves a least-squares problem with unit norm constraint

    Parameters
    ----------
    A: array_like shape (..., N, M)
        The data matrix
    b: array_like shape (..., N)
        The right-hand side
    weights: array_like shape (..., N), optional
        Optional weights
    tol: float, optional
        The tolerance to which the constraint must apply
    max_iter: int, optional
        The constraint is enforced via an iterative root finding procedure, if
        the root finding does not convergence, it is stopped after a maximum
        number of iterations

    Returns
    -------
    x: ndarray
        The unit norm vectors ``x`` minimizing ``np.sum(weights * (b - A @ x) ** 2)``
    """

    def T(A):
        """ transpose on last two axes """
        return A.swapaxes(-2, -1)

    def matvec(mat, vec):
        return (mat @ vec[..., None])[..., 0]

    shape, N, M = A.shape[:-2], A.shape[-2], A.shape[-1]
    assert b.shape[:-1] == shape
    assert b.shape[-1] == N

    if len(shape) == 0:
        A = A[None, ...]
        b = b[None, ...]
        dim_added_flag = True
    else:
        dim_added_flag = False

    A = A.reshape((-1, N, M))
    b = b.reshape((-1, N))

    if weights is not None:
        assert weights.shape[:-1] == shape
        assert weights.shape[-1] == N
        weights = weights.reshape((-1, N))

        if dim_added_flag:
            weights = weights[None, :]

        # apply the weighting
        Aw = A * weights[..., None]

        ATA = T(Aw) @ A  # shape (n_dim, n_dim)
        ATb = matvec(T(Aw), b)  # shape (n_dim,)

    else:
        ATA = T(A) @ A
        ATb = matvec(T(A), b)

    # make sure ATA is PSD
    ATA = 0.5 * (ATA + T(ATA))

    # eigenvalue decomposition, w.shape == (..., M), V.shape == (..., M, M)
    # v is sorted in descending order
    w, V = np.linalg.eigh(ATA)

    # project the right member in the eigenspace of ATA
    b_tilde = matvec(T(V), ATb)  # shape (n_dim,)

    # find the largest zero of the secular equation
    lambda_, ellapsed_epochs = root_secular(w, b_tilde, n_iter=max_iter, tol=tol)

    if ellapsed_epochs == max_iter:
        import warnings

        warnings.warn(
            "unit_ls: maximum number of equations reached without satisfying tolerance"
        )

    # compute the solution
    x = np.linalg.solve(ATA + lambda_[:, None, None] * np.eye(M)[None, :, :], ATb)

    # remove an extra dimension if it was added
    if dim_added_flag:
        x = x[0]

    return x.reshape(shape + (M,))
