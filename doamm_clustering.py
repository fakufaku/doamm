from enum import Enum

import numpy as np

import pyroomacoustics as pra
from external_mdsbf import MDSBF
from unit_ls import unit_ls
from utils import geom


class Measurement(Enum):
    DIRECT = "direct"
    XCORR = "x-corr"


def doa_mm_cost_per_bin(q, mics, wavenumbers, data, beta=1.0):
    """
    Parameters
    ----------
    q: array_like, shape (n_dim)
        the current propagation vector estimate
    mics: array_like, shape (n_mics, n_dim)
        the regression vectors corresponding to the microphone locations
        weighted by the wavenumbers
    wavenumbers: ndarray, shape (n_points)
        the wavenumbers corresponding to each data bin
    data: array_like, shape (n_points, n_mics)
        the phase of the measurements
    beta: float
        exponent of the robustifying function

    Returns
    -------
    c: ndarray, shape (n_points,)
        the cost associated with each data point
    """

    delta_t = mics @ q
    e = data - wavenumbers[:, None] @ delta_t[None, :]
    c = (0.5 * (1 + np.mean(np.cos(e), axis=-1))) ** beta
    return c


def doa_mm_cost(q, clusters, mics, wavenumbers, data, beta=1.0):
    """
    Parameters
    ----------
    q: array_like, shape (len(clusters), n_dim)
        the current propagation vector estimate
    clusters: list of array_like
        each element of the list is a list of the bins belonging to a cluster
    mics: array_like, shape (n_mics, n_dim)
        the regression vectors corresponding to the microphone locations
        weighted by the wavenumbers
    wavenumbers: ndarray, shape (n_points)
        the wavenumbers corresponding to each data bin
    data: array_like, shape (n_points, n_mics)
        the phase of the measurements
    beta: float
        exponent of the robustifying function

    Returns
    -------
    c: float
        The cost of the current solution
    """

    c = 0.0
    for q, S in zip(q, clusters):
        c += np.sum(doa_mm_cost_per_bin(q, mics, wavenumbers[S], data[S, :], beta=beta))

    return c


def doa_mm_auxiliary_variables(q, mics, wavenumbers, data, beta=1.0):
    """
    Parameters
    ----------
    q: array_like, shape (n_dim)
        the current propagation vector estimate
    mics: ndarray, shape (n_mics, n_dim)
        the location of the microphones
    wavenumbers: ndarray, shape (n_points)
        the wavenumbers corresponding to each data bin
    data: ndarray, shape (n_points, n_mics)
        the phase of the measurements
    beta: float
        exponent of the robustifying function

    Returns
    -------
    new_data: ndarray, shape (n_points, n_mics)
        the auxiliary right hand side
    weights: ndarray, shape (n_points, n_mics)
        the new weights
    """
    n_mics, n_dim = mics.shape
    n_points, _ = data.shape

    weights = np.zeros_like(data)

    delta_t = mics @ q  # shape (n_mics)
    e = data - wavenumbers[:, None] @ delta_t[None, :]  # shape (n_points, n_mics)

    # compute the offset to pi
    z = np.round(e / (2 * np.pi))
    phi = e - 2 * np.pi * z

    # adjust right-hand side
    new_data = data - 2 * np.pi * z

    # compute the weights
    weights[:] = 0.25 / n_dim * np.sinc(phi / np.pi)

    # this the time-frequency bin weight corresponding to the robustifying function
    # shape (n_points)
    if beta > 1.0:
        r = 0.5 * (1.0 + np.mean(np.cos(e), axis=-1))
        weights *= beta * r[:, None] ** (beta - 1)

    return new_data, weights


class DOAMM(pra.doa.DOA):
    """
    Implements the DOA-MM k-means like algorithm

    .. note:: Run locate_sources() to apply DOAMM

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
    measurement_type: str
        Which type of measurements to use:
        - 'Measurement.XCORR': Cross correlation measurements
        - 'Measurement.DIRECT': Microphone signals
    beta: float
        The exponent for the robustifying function, we expect that making beta larger
        should make the method less sensitive to outliers/noise
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
        mode="far",
        dim=None,
        measurements=Measurement.DIRECT,
        beta=1.0,
        n_iter=30,
        track_cost=False,
        init_grid=100,
        verbose=False,
        *args,
        **kwargs,
    ):
        """
        The init method
        """
        self._measurements = measurements
        self.beta = beta
        self.n_iter = n_iter
        self._track_cost = track_cost
        self._init_doa = pra.doa.MUSIC(
            L, fs, nfft, c=c, num_src=num_src, dim=dim, n_grid=init_grid
        )
        """
        self._init_doa = MDSBF(
            L, fs, nfft, c=c, num_src=num_src, dim=dim, n_grid=init_grid
        )
        """
        self.verbose = verbose

        L = np.array(L)

        if dim is None:
            dim = L.shape[0]

        super().__init__(
            L, fs, nfft, c=c, num_src=num_src, dim=dim, *args, **kwargs,
        )

        # differential microphone locations (for x-corr measurements)
        # shape (n_dim, n_mics * (n_mics - 1) / 2)
        self._L_diff = self._extract_off_diagonal(
            self.L[:, :, None] - self.L[:, None, :]
        )

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
        indices = np.arange(M)

        mask = np.ravel_multi_index(
            np.where(indices[:, None] > indices[None, :]), (M, M)
        )

        return X.reshape(X.shape[:-2] + (X.shape[-2] * X.shape[-1],))[..., mask]

    def _cost(self, qs, clusters, mics, wavenumbers, data):
        """ Compute the cost of the function """
        return doa_mm_cost(qs, clusters, mics, wavenumbers, data, beta=self.beta)

    def _recompute_centers(
        self, qs, clusters, mics, wavenumbers, data, n_iter=1,
    ):
        """
        Parameters
        ----------
        qs: array_like, shape (n_clusters, n_dim)
            the cluster direction vectors
        clusters: list of ndarray
            each entry in the list is the list of indices belonging to a cluster
        mics: ndarray, shape (n_mics, n_dim)
            the location of the microphones
        wavenumbers: ndarray, shape (n_points)
            the wavenumbers corresponding to each data bin
        data: ndarray, shape (n_points, n_mics)
            the phase of the measurements
        """

        n_clusters = len(clusters)
        n_points, n_mics = data.shape
        n_mics, n_dim = mics.shape
        assert wavenumbers.shape[0] == n_points

        # buffers
        weights = np.zeros((n_clusters, n_mics), dtype=data.dtype)
        rhs = np.zeros((n_clusters, n_mics), dtype=data.dtype)

        # broadcast the microphone matrix to the same size as weights and RHS
        mics_bc = np.broadcast_to(mics, (n_clusters, n_mics, n_dim))

        for epoch in range(n_iter):

            for i, (q, S) in enumerate(zip(qs, clusters)):

                new_data, new_weights = doa_mm_auxiliary_variables(
                    q, mics, wavenumbers[S], data[S, :], beta=self.beta
                )

                weights[i, :] = np.sum(new_weights * wavenumbers[S, None] ** 2, axis=0)
                rhs[i, :] = (
                    np.sum(new_data * new_weights * wavenumbers[S, None], axis=0)
                    / weights[i, :]
                )

            qs[:] = unit_ls(mics_bc, rhs, weights=weights, tol=1e-8, max_iter=1000)

        return qs, epoch

    def _recompute_clusters(self, qs, mics, wavenumbers, data):
        """
        Parameters
        ----------
        qs: array_like, shape (n_clusters, n_dim)
            the current propagation vector estimates
        mics: array_like, shape (n_points, n_mics, n_dim)
            the regression vectors corresponding to the microphone locations
            weighted by the wavenumbers
        wavenumbers: array_like, shape (n_points)
            the wavenumber for the data points
        data: array_like, shape (n_points, n_mics)
            the phase of the measurements
        """
        n_mics, n_dim = mics.shape
        n_points, _ = data.shape
        n_clusters = qs.shape[0]

        cost_per_bin = np.zeros((n_clusters, n_points))
        for i, q in enumerate(qs):
            cost_per_bin[i, :] = doa_mm_cost_per_bin(
                q, mics, wavenumbers, data, beta=self.beta
            )

        best_q = np.argmax(cost_per_bin, axis=0)

        clusters = []
        for i, q in enumerate(qs):
            S = np.where(best_q == i)[0]
            clusters.append(S)
            ## here is how to only include 10% best fit
            # n_best = int(0.1 * len(S))  # select best 10%
            # bestest = np.argsort(cost_per_bin[i, S])
            # clusters.append(S[bestest[-n_best:]])
            # clusters.append(np.random.choice(S, size=n_best, replace=False))

        return clusters

    def _cluster_center_init(self, X, mics, data):
        """
        Find a good initialization of the cluster centers

        For now, we do random
        """

        # We make the initialization random on the sphere
        q_init = np.random.randn(self.num_src, self.dim)
        q_init /= np.linalg.norm(q_init, axis=1, keepdims=True)

        return q_init

    def _doa_mm_run(self, qs0, mics, wavenumbers, data):
        """ Run the DOA/clustering """

        # initialize the algorithm
        qs = qs0
        clusters = self._recompute_clusters(qs, mics, wavenumbers, data)

        if self.verbose:
            doa, r = geom.cartesian_to_spherical(qs.T)
            print("Initial:")
            print(
                f"  colatitude={np.degrees(doa[0, :])}\n"
                f"  azimuth=   {np.degrees(doa[1, :])}\n"
                f"  clust_size={[len(S) for S in clusters]}\n"
            )

        if self._track_cost:
            c = self._cost(qs, clusters, mics, wavenumbers, data)
            self.cost = [c]
            if self.verbose:
                print(f"  cost {c}")

        # run the iterations
        for epoch in range(self.n_iter):

            # Re-estimate cluster centers
            qs[:], _ = self._recompute_centers(
                qs, clusters, mics, wavenumbers, data, n_iter=1
            )

            clusters = self._recompute_clusters(qs, mics, wavenumbers, data)

            if self.verbose:
                doa, r = geom.cartesian_to_spherical(qs.T)
                print(f"Epoch {epoch}")
                print(
                    f"  colatitude={np.degrees(doa[0, :])}\n"
                    f"  azimuth=   {np.degrees(doa[1, :])}\n"
                    f"  clust_size={[len(S) for S in clusters]}\n"
                )

            if self._track_cost:
                c = self._cost(qs, clusters, mics, wavenumbers, data)
                self.cost.append(c)
                if self.verbose:
                    print(f"  cost: {c}")

        return qs, clusters

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

        # remove unnecessary frequencies
        # shape (n_freq, n_frames, n_channels)
        X_ = X[:, self.freq_bins, :].transpose([1, 2, 0])

        # the wavenumbers (n_freq * n_frames)
        wavenumbers = np.broadcast_to(
            2 * np.pi * self.freq_hz[:, None] / self.c, (n_freq, n_frames)
        ).flatten()

        # First, we need to compute the measurements
        if self._measurements == Measurement.XCORR:

            # For x-corr measurements, we consider differences of microphones as sensors
            # n_mics = n_channels * (n_channels - 1) / 2
            n_mics = self._L_diff.shape[1]

            # shape (n_mics, n_dim)
            mics = self._L_diff.T

            # shape (n_freq * n_frames, n_mics)
            data = np.angle(
                self._extract_off_diagonal(
                    X_[..., :, None] @ np.conjugate(X_[..., None, :])
                )
            ).reshape((-1, n_mics))

        elif self._measurements == Measurement.DIRECT:

            n_mics = self.L.shape[1]
            mics = self.L.T

            # shape (n_freq * n_frames * n_mics)
            data = np.angle(X_).reshape((-1, n_mics))

        else:
            raise ValueError("Invalid measurement type.")

        # init with grid based doa method
        num_src_init = self.num_src
        self._init_doa.locate_sources(X, num_src=num_src_init, freq_bins=self.freq_bins)
        qs = geom.spherical_to_cartesian(
            doa=np.c_[self._init_doa.colatitude_recon, self._init_doa.azimuth_recon],
            distance=np.ones(num_src_init),
        ).T
        """

        # init with a simple grid
        qs = pra.doa.GridSphere(n_points=self._init_doa.grid.n_points).cartesian.T
        """

        # qs = self._cluster_center_init(mics, data)

        # Run the DOA algorithm
        qs, self.clusters = self._doa_mm_run(qs, mics, wavenumbers, data)

        # sort the clusters by size
        the_clusters = [
            {"center": q, "bins": bins} for q, bins in zip(qs, self.clusters)
        ]
        sorted(the_clusters, key=lambda S: len(S["bins"]), reverse=True)

        qs_sel = np.vstack([S["center"] for S in the_clusters[: self.num_src]])

        # Now we need to convert to azimuth/doa
        # self._doa_recon, _ = geom.cartesian_to_spherical(qs.T)
        self._doa_recon, _ = geom.cartesian_to_spherical(qs_sel.T)

        # self.plot(self.clusters, mics, data)

    def locate_sources(self, *args, **kwargs):

        super().locate_sources(*args, **kwargs)
        self.colatitude_recon = self._doa_recon[0, :]
        self.azimuth_recon = self._doa_recon[1, :]

        # make azimuth always positive
        I = self.azimuth_recon < 0.0
        self.azimuth_recon[I] = 2.0 * np.pi + self.azimuth_recon[I]

    def plot(self, clusters, mics, data):
        """
        Plot the cost function for each cluster
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            import warnings

            warnings.warn("Matplotlib is required for plotting")
            return

        n_clusters = len(clusters)

        grids = [pra.doa.GridSphere(n_points=500) for S in clusters]

        for i, (S, G) in enumerate(zip(clusters, grids)):

            def func_cost(x, y, z):
                qs = np.c_[x, y, z]
                cost = []
                for q in qs:
                    c = np.mean(
                        doa_mm_cost_per_bin(
                            q, mics, wavenumbers[S, :], data[S, :], self.beta
                        )
                    )
                    cost.append(c)
                # import pdb

                # pdb.set_trace()
                return np.array(cost)

            G.apply(func_cost)

        for G in grids:
            G.plot(plotly=False)
