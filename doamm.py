from enum import Enum

import pyroomacoustics as pra
from unit_irls import unit_irls


class Measurement(Enum):
    DIRECT = "direct"
    XCORR = "x-corr"


def doa_mm_weight(q, mics, data, beta=1.0):
    """
    Parameters
    ----------
    x: array_like, shape (n_dim)
        the current propagation vector estimate
    mics: array_like, shape (n_points, n_dim)
        the regression vectors corresponding to the microphone locations
        weighted by the wavenumbers
    data: array_like, shape (n_points)
        the phase of the measurements
    beta: float
        exponent of the robustifying function
    """
    n_points, n_dim = mics.shape

    weights = np.zeros_like(mics)

    e = data - mics @ q

    z = np.round(e / (2 * np.pi))
    phi = e - 2 * np.pi * z

    I = np.abs(phi) > 1e-15
    weights[:] = 0.25 / n_dim
    weights[I] *= np.sin(phi) / phi

    # this the time-frequency bin weight corresponding to the robustifying function
    # shape (n_points)
    if beta > 1.0:
        r = 0.5 * (1.0 + np.mean(np.cos(data - mics @ q), axis=-1))
        weights *= beta * r[:, None] ** (beta - 1)

    return weights


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
        *args,
        **kwargs
    ):
        """
        The init method
        """
        self._measurements = measurements

        L = np.array(L)

        if dim is None:
            dim = L.shape[0]

        super().__init__(
            L, fs, nff, c=c, num_src=num_src, dim=dim, *args, **kwargs,
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
        mask = np.where(indices[:, None] > indices[None, :])

        return X.reshape(X.shape[:-2] + (X.shape[-2] * X.shape[-1],))[..., mask]

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

        # the wavenumbers (n_freq)
        wavenum = 2 * np.pi * self.freq_hz / self.c

        # First, we need to compute the measurements
        if self._measurements == Measurement.XCORR:

            # For x-corr measurements, we consider differences of microphones as sensors
            # n_mics = n_channels * (n_channels - 1) / 2
            n_mics = self._L_diff.shape[1]

            # shape (n_freq, n_frames, n_mics, n_dim)
            mics = np.broadcast_to(
                wavenum[:, None, None, None] * self._L_diff.T[None, None, :, :],
                (n_freq, n_frames, n_mics, n_dim),
            ).reshape((-1, n_dim))

            # shape (n_freq * n_frames * n_mics)
            data = np.angle(
                self._extract_off_diagonal(
                    X_[..., :, None] @ np.conjugate(X_[..., None, :])
                )
            ).flatten()

        elif self._measurements == Measurement.DIRECT:

            n_mics = self.L.shape[1]
            mics = np.broadcast_to(
                wavenum[:, None, None, None] * self.L.T[None, None, :, :],
                (n_freq, n_frames, n_mics, n_dim),
            ).reshape((-1, n_dim))

            # shape (n_freq * n_frames * n_mics)
            data = np.angle(X_).flatten()

        else:
            raise ValueError("Invalid measurement type.")
