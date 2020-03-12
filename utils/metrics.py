import numpy as np
from scipy.optimize import linear_sum_assignment


def doa_eval(doa_ref, doa_est):
    """
    Evaluate the error between a set of reference and estimated DOA.
    The optimal permutation between the two sets is also computed.

    Parameters
    ----------
    doa_ref: shape (n_doas,) or (n_doas, 2)
        The first array of doas
    doa_est: shape (n_doas,) or (n_doas, 2)

    Returns
    -------
    rmse: float
        The root mean squared error
    permuation: array
        The permutation of doa_est that gives smallest rmse
    """

    assert (
        doa_ref.ndim == doa_est.ndim,
        "The two DOA arrays should have the same number of dimensions",
    )

    if doa_ref.ndim == 1 or (doa_ref.ndim == 2 and doa_ref.shape[1] == 1):

        if doa_ref.ndim == 1:
            doa_ref = doa_ref[:, None]
            doa_est = doa_est[:, None]

        D = circ_dist(doa_ref, doa_est.T)

    elif doa_ref.ndim == 2 and doa_ref.shape[1] == 2:

        D = great_circ_dist(
            doa_ref[:, :1], doa_ref[:, 1:], doa_est[:, :1].T, doa_est[:, 1:].T
        )

    row_ind, col_ind = linear_sum_assignment(D)

    return D[row_ind, col_ind], col_ind


def circ_dist(azimuth1, azimuth2, radius=1.0):
    """
    Returns the shortest distance between two points on a circle

    Parameters
    ----------
    azimuth1:
        azimuth of point 1
    azimuth2:
        azimuth of point 2
    r: optional
        radius of the circle (Default 1)
    """
    return np.arccos(np.cos(azimuth1 - azimuth2))


def great_circ_dist(colatitude1, azimuth1, colatitude2, azimuth2, radius=1.0):
    """
    calculate great circle distance for points located on a sphere

    Parameters
    ----------

    colatitude1: colatitude of point 1
    azimuth1: azimuth of point 1
    colatitude2: colatitude of point 2
    azimuth2: azimuth of point 2
    radius: radius of the sphere

    Returns
    -------
    float or ndarray
        great-circle distance
    """

    d_azimuth = np.abs(azimuth1 - azimuth2)
    dist = radius * np.arctan2(
        np.sqrt(
            (np.sin(colatitude2) * np.sin(d_azimuth)) ** 2
            + (
                np.sin(colatitude1) * np.cos(colatitude2)
                - np.cos(colatitude1) * np.sin(colatitude2) * np.cos(d_azimuth)
            )
            ** 2
        ),
        np.cos(colatitude1) * np.cos(colatitude2)
        + np.sin(colatitude1) * np.sin(colatitude2) * np.cos(d_azimuth),
    )
    return dist
