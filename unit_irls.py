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
        x[I] = 0.5 * (x[I] + lower_bound)
        x[nI] = new_x[nI]

        if np.max(np.abs(f)) < tol:
            break

    return (
        w_max * x,  # rescale before returning
        epoch,  # number of iterations for info
    )


def unit_irls(
    x,
    A,
    b,
    func_aux_var=None,
    n_iter=None,
    tol=None,
    secular_n_iter=10,
    secular_tol=1e-10,
):
    """
    Iterative Reweighted Least-Squares algorithm with a unit norm constraint

    Parameters
    ----------
    A: array_like, shape (..., D)
        The data matrix
    b: array_like, shape (...,)
        The shape of b should the same as A.shape[:-1]
    func_aux_var: func
        The function computing the auxiliary variables (right-hand side, weights)
    n_iter: int
        The maximum number of iterations
    tol: float
        The algorithm stops when the surrogate cost function is smaller than this value
    """

    shape, n_dim = A.shape[:-1], A.shape[-1]
    assert b.shape == shape

    A_ = A.reshape((-1, n_dim))
    b_ = b.flatten()

    for epoch in range(n_iter):

        # shape == b.shape
        y, w = func_aux_var(x, A, b)
        y_ = y.flatten()
        w_ = w.flatten()

        # apply the weighting
        Aw = A_ * w_[:, None]

        ATA = Aw.T @ A_  # shape (n_dim, n_dim)
        ATb = Aw.T @ y_  # shape (n_dim,)

        # make sure ATA is PSD
        ATA = 0.5 * (ATA + ATA.T)

        # eigenvalue decomposition, v.shape == (n_dim,), W.shape == (n_dim, n_dim)
        # v is sorted in descending order
        w, V = np.linalg.eigh(ATA)

        # project the right member in the eigenspace of ATA
        b_tilde = V.T @ ATb  # shape (n_dim,)

        # find the largest zero of the secular equation
        lambda_, secular_epoch = root_secular(
            w[None, :], b_tilde[None, :], n_iter=secular_n_iter, tol=secular_tol
        )
        lambda_ = lambda_[0]

        # compute the solution
        new_x = np.linalg.solve(ATA + lambda_ * np.eye(n_dim), ATb)

        progress = np.abs(new_x - x)
        x[:] = new_x

        if tol is not None and progress < tol:
            break

    return x, epoch
