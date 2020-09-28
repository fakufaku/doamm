import numpy as np


def cartesian_to_spherical(p):
    """
    Parameters
    ----------
    p: array_like, shape (3, n_points)
        A collection of vectors

    Returns
    -------
    doa: ndarray, shape (2, n_points)
        The (colatitude, azimuth) pairs giving the direction of the vectors
    r: ndarray, shape (n_points,)
        The norms of the vectors
    """

    r = np.linalg.norm(p, axis=0)
    u = p / r[None, :]

    doa = np.zeros((2, p.shape[1]), dtype=p.dtype)

    # colatitude
    doa[0, :] = np.arctan2(np.sqrt(p[0, :] ** 2 + p[1, :] ** 2), p[2, :])

    # azimuths
    doa[1, :] = np.arctan2(p[1, :], p[0, :])
    I = doa[1, :] < 0
    doa[1, I] = 2 * np.pi + doa[1, I]

    return doa, r


def spherical_to_cartesian(doa, distance, ref=None):
    """
    Transform spherical coordinates to cartesian

    Parameters
    ----------
    doa: (n_points, 2)
        The doa of the sources as (colatitude, azimuth) pairs
    distance: float or array (n_points)
        The distance of the source from the reference point
    ref: ndarray, shape (3,)
        The reference point, defaults to zero if not specified

    Returns
    -------
    R: array (3, n_points)
        An array that contains the cartesian coordinates of the points
        in its columns
    """

    doa = np.array(doa)
    distance = np.array(distance)

    if distance.ndim == 0:
        distance = distance[None]

    assert doa.ndim == 2
    assert doa.shape[1] == 2
    assert distance.ndim < 3

    R = np.zeros((3, doa.shape[0]), dtype=doa.dtype)

    R[0, :] = np.cos(doa[:, 1]) * np.sin(doa[:, 0])
    R[1, :] = np.sin(doa[:, 1]) * np.sin(doa[:, 0])
    R[2, :] = np.cos(doa[:, 0])
    R *= distance[None, :]

    if ref is not None:
        assert ref.ndim == 1
        assert ref.shape[0] == 3
        R += ref[:, None]

    return R
