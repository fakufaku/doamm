import numpy as np


def extract_off_diagonal(X):
    """
    Parameters
    ----------
    X: array_like, shape (..., M, M)
        A multi dimensional array

    Returns
    -------
    Y: array_like, shape (..., M * (M - 1) / 2)
        The linearized entries under the main diagonal
    """
    # we need to format the sensors
    M = X.shape[-1]
    assert X.shape[-2] == M
    indices = np.arange(M)
    mask = np.ravel_multi_index(np.where(indices[:, None] > indices[None, :]), (M, M))
    print(indices[:, None] > indices[None, :])
    print(mask)

    return X.reshape(X.shape[:-2] + (X.shape[-2] * X.shape[-1],))[..., mask]
    return X[..., mask]


if __name__ == "__main__":

    n_dim = 3
    n_mics = 4
    n_subsp = 2
    w = 2 * np.pi * 20.0 / 343.0  # 440 Hz @ 343 m/s

    # direction vector
    q = np.random.randn(n_dim)
    q /= np.linalg.norm(q)

    # microphones
    I = np.arange(n_dim * n_mics)
    np.random.shuffle(I)
    L = I.reshape((n_dim, n_mics))

    # noise subspace matrix
    E = np.random.randn(n_mics, n_subsp) + 1j * np.random.randn(n_mics, n_subsp)
    E /= np.linalg.norm(E, axis=0, keepdims=True)

    steervec = np.exp(1j * w * L.T @ q)

    # computation 1

    ell1 = np.linalg.norm(np.conj(E).T @ steervec) ** 2

    # computation 2
    V = E @ np.conj(E.T)
    V = 0.5 * (V + np.conj(V.T))
    diag_fact = np.trace(np.abs(V))
    diag_fact2 = np.sum(np.linalg.norm(E, axis=1) ** 2)
    Ld = extract_off_diagonal(L[:, :, None] - L[:, None, :])
    E_vec_cpx = extract_off_diagonal(V)
    E_vec = np.r_[np.real(E_vec_cpx), np.imag(E_vec_cpx)]

    e = w * Ld.T @ q
    steervec_real = np.r_[np.cos(e), np.sin(e)]

    ell2 = diag_fact + 2 * np.inner(steervec_real, E_vec)

    # computation 3
    steervec_3 = np.exp(-1j * w * Ld.T @ q)
    ell3 = diag_fact + 2 * np.sum(np.real(steervec_3 * E_vec_cpx))

    # computation 4
    ell4 = np.real(np.conj(steervec) @ V @ steervec)

    # computation 5
    ell5 = np.real(np.trace((steervec[:, None] @ np.conj(steervec[None, :])) @ V))

    print(f"diag_fact 1: {diag_fact} 2: {diag_fact2}")
    print(f"ell 1: {ell1} 2: {ell2} 3: {ell3} 4: {ell4} 5: {ell5}")
