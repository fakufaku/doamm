# Implementation of MDSBF algorithm
#
# Copyright 2020 Masahito Togami and Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Masahito Togami
# Date: Mar 13, 2020
# Modified by Robin Scheibler
from __future__ import division, print_function

from pyroomacoustics.doa import *


class MDSBF(DOA):
    """
    Class to apply Steered Response Power (SRP) direction-of-arrival (DoA) for
    a particular microphone array.
    .. note:: Run locate_source() to apply the SRP-PHAT algorithm.
    Parameters
    ----------
    L: numpy array
        Microphone array positions. Each column should correspond to the
        cartesian coordinates of a single microphone.
    fs: float
        Sampling frequency.
    nfft: int
        FFT length.
    c: float
        Speed of sound. Default: 343 m/s
    num_src: int
        Number of sources to detect. Default: 1
    mode: str
        'far' or 'near' for far-field or near-field detection
        respectively. Default: 'far'
    r: numpy array
        Candidate distances from the origin. Default: np.ones(1)
    azimuth: numpy array
        Candidate azimuth angles (in radians) with respect to x-axis.
        Default: np.linspace(-180.,180.,30)*np.pi/180
    colatitude: numpy array
        Candidate elevation angles (in radians) with respect to z-axis.
        Default is x-y plane search: np.pi/2*np.ones(1)
    """

    def __init__(
        self,
        L,
        fs,
        nfft,
        c=343.0,
        num_src=1,
        mode="far",
        r=None,
        azimuth=None,
        colatitude=None,
        **kwargs,
    ):

        DOA.__init__(
            self,
            L=L,
            fs=fs,
            nfft=nfft,
            c=c,
            num_src=num_src,
            mode=mode,
            r=r,
            azimuth=azimuth,
            colatitude=colatitude,
            **kwargs,
        )

        self.num_pairs = self.M * (self.M - 1) / 2

        # self.mode_vec = np.conjugate(self.mode_vec)

    def _process_org(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response
        spectrum.
        """

        ones = np.ones(self.L.shape[1])

        srp_cost = np.zeros(self.grid.n_points)

        # apply PHAT weighting
        absX = np.abs(X)
        absX[absX < tol] = tol
        pX = X / absX

        CC = []
        for k in self.freq_bins:
            CC.append(np.dot(pX[:, k, :], np.conj(pX[:, k, :]).T))
        CC = np.array(CC)

        for n in range(self.grid.n_points):

            # get the mode vector axis: (frequency, microphones)
            mode_vec = self.mode_vec[self.freq_bins, :, n]

            # compute the outer product along the microphone axis
            mode_mat = np.conj(mode_vec[:, :, None]) * mode_vec[:, None, :]

            # multiply covariance by mode vectors and sum over the frequencies
            R = np.sum(CC * mode_mat, axis=0)

            # Now sum over all distince microphone pairs
            sum_val = np.inner(ones, np.dot(np.triu(R, 1), ones))

            # Finally normalize
            srp_cost[n] = (
                np.abs(sum_val) / self.num_snap / self.num_freq / self.num_pairs
            )

        self.grid.set_values(srp_cost)

    def _process_kdtree(self, X):
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
        from scipy.spatial import KDTree, cKDTree

        n_dim = self.dim
        n_freq = len(self.freq_bins)
        n_frames = X.shape[2]
        n_mics = self.L.shape[1]

        assert self.L.shape[1] == X.shape[0]

        # remove unnecessary frequencies
        # shape (n_freq, n_frames, n_channels)
        X_ = X[:, self.freq_bins, :].transpose([1, 2, 0])
        X_ /= np.maximum(np.abs(X_), 1e-18)

        # the delta time with the grid (n_grid, n_mics)
        DT = self.grid.cartesian.T @ self.L

        cost = np.zeros(self.grid.n_points, dtype=np.int)

        for k, f_hz in enumerate(self.freq_hz):
            # steering vectors
            A = np.exp(2.0j * np.pi * f_hz / self.c * DT)  # shape (n_grid, n_mics)
            Ari = np.hstack((np.real(A), np.imag(A)))

            tree = cKDTree(Ari)

            Xk_ri = np.hstack((np.real(X_[k]), np.imag(X_[k])))
            distances, nearest_neighbors = tree.query(Xk_ri)
            bin_indices, counts = np.unique(nearest_neighbors, return_counts=True)

            cost[bin_indices] += counts

        self.grid.set_values(cost)

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response
        spectrum.
        """

        self._process_kdtree(X)
        return

        ones = np.ones(self.L.shape[1])

        mdsbf_cost = np.zeros(self.grid.n_points)

        # apply PHAT weighting
        # absX = np.abs(X)
        # absX[absX < tol] = tol
        # pX = X / absX

        # CC = []
        # for k in self.freq_bins:
        #    CC.append( np.dot(pX[:,k,:], np.conj(pX[:,k,:]).T) )
        # CC = np.array(CC)
        # channels, freq,time
        # frequency, mic, grid
        # conjugateが不安
        # print(self.grid.n_points)
        # print(self.freq_bins)
        sub_freq_bins = np.array_split(self.freq_bins, 5)

        # for k in sub_freq_bins:
        for k in self.freq_bins:
            # k=np.array(k,dtype=np.int)

            mode_vec = self.mode_vec[k, :, :]
            mode_vec = np.conjugate(mode_vec)
            # print(mode_vec)
            prod = np.einsum("mi,mt->ti", mode_vec, X[:, k, :])
            # prod=np.einsum("mi,mt->ti",mode_vec,X[:,k,:])
            amp = np.abs(prod)
            # print(k,np.max(amp))
            index = np.argmax(amp, axis=-1)
            # ft
            # mode_vec=self.mode_vec
            for n in range(self.grid.n_points):
                mdsbf_cost[n] = mdsbf_cost[n] + np.count_nonzero(index == n)

        self.grid.set_values(mdsbf_cost)
