from enum import Enum

import numpy as np

import pyroomacoustics as pra
from external_mdsbf import MDSBF
from unit_ls import unit_ls
from utils import geom


class MMUSICType(Enum):
    Linear = 0
    Quadratic = 1


def power_mean(X, s=1.0, *args, **kwargs):

    if s != 1.0:
        return (np.mean(X ** s, *args, **kwargs)) ** (1.0 / s)
    else:
        return np.mean(X, *args, **kwargs)


def mmusic_cost(q, mics, wavenumbers, data, s=1.0):
    """
    Parameters
    ----------
    q: array_like, shape (n_dim)
        the current propagation vector estimate
    mics: ndarray, shape (n_mics, n_dim)
        the location of the microphones
    wavenumbers: ndarray, shape (n_freq)
        the wavenumbers corresponding to each data bin
    data: ndarray, shape (n_freq, n_mics)
        the data that defines the noise subspace
    s: float
        the power mean parameter

    Returns
    -------
    cost: float
        the value of the MMUSIC cost function
    """
    n_mics, n_dim = mics.shape
    n_freq, _ = data.shape

    weights = np.zeros_like(data)

    delta_t = mics @ q  # shape (n_mics)
    e = wavenumbers[:, None] @ delta_t[None, :]  # shape (n_freq, n_mics)

    E = np.concatenate((np.cos(e), np.sin(e)), axis=-1)  # shape (n_freq, 2 * n_mics)
    ell = n_mics + 2 * np.sum(data * E, axis=-1)

    cost = power_mean(ell, s=s)

    return cost


def mmusic_cosine_majorization(q, mics, wavenumbers, data, s=1.0):
    """
    Parameters
    ----------
    q: array_like, shape (n_dim)
        the current propagation vector estimate
    mics: ndarray, shape (n_mics, n_dim)
        the location of the microphones
    wavenumbers: ndarray, shape (n_freq)
        the wavenumbers corresponding to each data bin
    data: ndarray, shape (n_freq, n_mics)
        the data that defines the noise subspace
    s: float
        the power mean parameter

    Returns
    -------
    new_data: ndarray, shape (n_points, n_mics)
        the auxiliary right hand side
    weights: ndarray, shape (n_points, n_mics)
        the new weights
    """
    n_mics, n_dim = mics.shape
    n_freq, _ = data.shape

    weights = np.zeros_like(data)
    new_data = np.zeros_like(data)
    e = np.zeros_like(data)

    I = data > 0.0

    delta_t = mics @ q  # shape (n_mics)
    e_base = wavenumbers[:, None] @ delta_t[None, :]
    e[:, :n_mics] = e_base  # shape (n_freq, n_mics)
    e[:, n_mics:] = 0.5 * np.pi - e[:, :n_mics]

    e[I] = np.pi - e[I]

    # compute the offset to pi
    z = np.round(e / (2.0 * np.pi))
    zpi = 2 * z * np.pi
    phi = e - zpi

    # adjust right-hand side
    new_data[:, :n_mics] = zpi[:, :n_mics]
    I_up = I[:, :n_mics]
    new_data[:, :n_mics][I_up] = np.pi - zpi[:, :n_mics][I_up]

    new_data[:, n_mics:] = 0.5 * np.pi - zpi[:, n_mics:]
    I_down = I[:, n_mics:]
    new_data[:, n_mics:][I_down] = zpi[:, n_mics:][I_down] - 0.5 * np.pi

    # compute the weights
    weights[:] = np.abs(data) * np.sinc(phi / np.pi)

    # this the time-frequency bin weight corresponding to the robustifying function
    # shape (n_points)
    if s < 1.0:
        E = np.concatenate((np.cos(e_base), np.sin(e_base)), axis=-1)
        ell = n_mics + 2 * np.sum(data * E, axis=-1)
        r = (1.0 / n_freq) * ell ** (s - 1.0) / np.mean(ell ** s) ** (1.0 - 1.0 / s)
        weights *= r[:, None]

    # We can reduce the number of terms by completing the squares
    w_red = np.sum(weights * wavenumbers[:, None] ** 2, axis=0)
    weights_red = w_red[:n_mics] + w_red[n_mics:]

    r_red = np.sum(new_data * weights * wavenumbers[:, None], axis=0)
    data_red = (r_red[:n_mics] + r_red[n_mics:]) / weights_red

    return data_red, weights_red


class MMUSIC(pra.doa.MUSIC):
    """
    Implements the MUSCI DOA algorithm with optimization directly on the array
    manifold using an MM algorithm

    .. note:: Run locate_sources() to apply MMUSIC

    Parameters
    ----------
    L: array_like, shape (n_dim, n_mics)
        Contains the locations of the microphones in the columns
    fs: int or float
        Sampling frequency
    nfft: int
        FFT size
    c: float, optional
        Speed of sound
    num_src: int, optional
        The number of sources to recover (default 1)
    s: float
        The exponent for the robustifying function, we expect that making beta larger
        should make the method less sensitive to outliers/noise
    n_grid: int
        The size of the grid search for initialization
    track_cost: bool
        If True, the cost function will be recorded
    init_grid: int
        Size of the grid for the rough initialization method
    verbose: bool, optional
        Whether to output intermediate result for debugging purposes
    """

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        s=-1.0,
        mm_type=MMUSICType.Quadratic,
        mode="far",
        dim=None,
        n_iter=30,
        n_grid=100,
        track_cost=False,
        verbose=False,
        *args,
        **kwargs,
    ):
        """
        The init method
        """
        self.s = s
        self.mm_type = mm_type
        self.n_iter = n_iter
        self._track_cost = track_cost
        self.verbose = verbose

        L = np.array(L)

        assert (
            self.s <= 1.0
        ), "The parameter s of MMUSIC should be smaller or equal to 1."

        if dim is None:
            dim = L.shape[0]

        super().__init__(
            L, fs, nfft, c=c, num_src=num_src, dim=dim, n_grid=n_grid, *args, **kwargs,
        )

        # differential microphone locations (for x-corr measurements)
        # shape (n_dim, n_mics * (n_mics - 1) / 2)
        self._L_diff = self._extract_off_diagonal(
            self.L[:, :, None] - self.L[:, None, :]
        )

        # for the linear type algorithm, we need
        self._L_diff2 = self._L_diff @ self._L_diff.T
        self.ev_max = np.max(np.linalg.eigvals(self._L_diff2))

    def _extract_off_diagonal(self, X):
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

        mask = np.ravel_multi_index(
            np.where(indices[:, None] > indices[None, :]), (M, M)
        )

        return X.reshape(X.shape[:-2] + (X.shape[-2] * X.shape[-1],))[..., mask]

    def _cost(self, q, mics, wavenumbers, data):
        """ Compute the cost of the function """
        return mmusic_cost(q, mics, wavenumbers, data, s=self.s)

    def _optimize_direction(
        self, q, mics, wavenumbers, data, n_iter=1,
    ):
        """
        Parameters
        ----------
        q: array_like, shape (n_dim,)
            the initial direction vector
        mics: ndarray, shape (n_mics, n_dim)
            the location of the microphones
        wavenumbers: ndarray, shape (n_freq)
            the wavenumbers corresponding to each data bin
        data: ndarray, shape (n_freq, n_mics)
            the phase of the measurements
        """

        n_mics, n_dim = mics.shape
        n_freq, _ = data.shape
        assert wavenumbers.shape[0] == n_freq

        # buffers
        qs = np.zeros((1, n_dim), dtype=q.dtype)
        qs[0, :] = q
        weights = np.zeros((1, n_mics), dtype=data.dtype)
        rhs = np.zeros((1, n_mics), dtype=data.dtype)

        # broadcast the microphone matrix to the same size as weights and RHS
        mics_bc = np.broadcast_to(mics, (1, n_mics, n_dim))

        for epoch in range(n_iter):

            # new_data.shape == ()
            # new_weights.shape == ()
            # the applies the cosine majorization
            new_data, new_weights = mmusic_cosine_majorization(
                qs[0], mics, wavenumbers, data, s=self.s
            )

            if self.mm_type == MMUSICType.Quadratic:
                qs[:] = unit_ls(
                    mics_bc,
                    new_data[None, :],
                    weights=new_weights[None, :],
                    tol=1e-8,
                    max_iter=1000,
                )
            elif self.mm_type == MMUSICType.Linear:
                C = np.max(new_weights) * self.ev_max
                y = self._L_diff @ (new_data * new_weights).T
                Lq = self._L_diff @ (new_weights[:, None] * (self._L_diff.T @ qs.T))

                # compute new direction
                qs[...] = y.T - Lq.T + C * qs

                # apply norm constraint
                qs /= np.linalg.norm(qs, axis=1, keepdims=True)

        return qs[0], epoch

    def _process(self, X):
        """
        Process the input data and computes the DOAs

        Parameters
        ----------
        X: array_like, shape (n_channels, n_frequency, n_frames)
            The multichannel STFT of the microphone signals
            Set of signals in the frequency (RFFT) domain for current 
            frame. Size should be M x F x S, where M should correspond to the 
            number of microphones, F to nfft/2+1, and S to the number of snapshots 
            (user-defined). It is recommended to have S >> M.
        """
        n_dim = self.dim
        n_freq = len(self.freq_bins)
        n_frames = X.shape[2]

        assert self.L.shape[1] == X.shape[0]

        # STEP 1: Classic MUSIC

        # compute the covariance matrices
        self.Pssl = np.zeros((self.num_freq, self.grid.n_points))
        C_hat = self._compute_correlation_matricesvec(X)

        # subspace decomposition (we need these for STEP 2)
        Es, En, ws, wn = self._subspace_decomposition(C_hat)

        # the mode vectors, shape (n_grid, n_freq, n_mics)
        mod_vec = np.transpose(
            np.array(self.mode_vec[self.freq_bins, :, :]), axes=[2, 0, 1]
        )
        self.Pssl = np.linalg.norm(
            np.conj(mod_vec[:, :, None, :]) @ En[None, :, :], axis=(-1, -2)
        )
        self.grid.set_values(1.0 / power_mean(self.Pssl, s=self.s, axis=1))

        # find the peaks for the initial estimate
        self.src_idx = self.grid.find_peaks(k=self.num_src)
        qs = self.grid.cartesian[:, self.src_idx].T

        # STEP 2: refinement via the MM algorithm

        # the wavenumbers (n_freq * n_frames)
        wavenumbers = 2 * np.pi * self.freq_hz / self.c

        # For x-corr measurements, we consider differences of microphones as sensors
        # n_mics = n_channels * (n_channels - 1) / 2
        n_mics = self._L_diff.shape[1]

        # shape (n_mics, n_dim)
        mics = self._L_diff.T

        # shape (n_freq, n_mics)
        v_c = self._extract_off_diagonal(En @ np.conj(En).swapaxes(-2, -1))
        data = np.concatenate((np.real(v_c), np.imag(v_c)), axis=-1)

        # Run the DOA algoritm
        self.cost = [[] for k in range(self.num_src)]

        for epoch in range(self.n_iter):

            for k, q in enumerate(qs):

                qs[k, :], epochs = self._optimize_direction(
                    q, mics, wavenumbers, data, n_iter=1,
                )

                if self.verbose:
                    doa, r = geom.cartesian_to_spherical(qs[k, None, :].T)
                    print(f"Epoch {epoch} Source {k}")
                    print(
                        f"  colatitude={np.degrees(doa[0, :])}\n"
                        f"  azimuth=   {np.degrees(doa[1, :])}\n"
                    )

                if self._track_cost:
                    c = self._cost(qs[k], mics, wavenumbers, data)
                    self.cost[k].append(c)
                    if self.verbose:
                        print(f"  cost: {c}")

        # Now we need to convert to azimuth/doa
        # self._doa_recon, _ = geom.cartesian_to_spherical(qs.T)
        self._doa_recon, _ = geom.cartesian_to_spherical(qs.T)

        # self.plot(mics, wavenumbers, data, En)

    def locate_sources(self, *args, **kwargs):

        super().locate_sources(*args, **kwargs)
        self.colatitude_recon = self._doa_recon[0, :]
        self.azimuth_recon = self._doa_recon[1, :]

        # make azimuth always positive
        I = self.azimuth_recon < 0.0
        self.azimuth_recon[I] = 2.0 * np.pi + self.azimuth_recon[I]

    def plot(self, mics, wavenumbers, data, En):
        """
        Plot the cost function for each cluster
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        grid = pra.doa.GridSphere(n_points=1000)

        def func_cost(x, y, z):
            qs = np.c_[x, y, z]
            cost = []
            for q in qs:
                c = mmusic_cost(q, mics, wavenumbers, data, self.s)
                cost.append(c)

            return np.array(cost)

        grid.apply(func_cost)

        grid.plot(plotly=False)
