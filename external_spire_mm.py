# Author: Masahito Togami
# Date: Mar 13, 2020
from __future__ import division, print_function

import scipy.spatial as spatial

import localization.generate_steering_vector as gsv
import pyroomacoustics.doa as doa
from pyroomacoustics.doa import *
from utils import arrays, geom, metrics


class ModeVector2(object):
    """
    This is a class for look-up tables of mode vectors. This look-up table
    is an outer product of three vectors running along candidate locations, time,
    and frequency. When the grid becomes large, the look-up table might be
    too large to store in memory. In that case, this class allows to only compute
    the outer product elements when needed, only keeping the three vectors in memory.
    When the table is small, a `precompute` option can be set to True to compute
    the whole table in advance.
    """

    def __init__(self, L, fs, nfft, c, grid, mode="far", precompute=False):
        """
        The constructor

        Parameters
        ----------
        L: ndarray
            contains the locations of the sensors in the columns of the array
        fs: int
            the sampling frequency of the input signal
        nfft: int
            the FFT length
        c: float
            the speed of sound
        grid: pyroomacoustcs.doa.Grid object
            the underlying grid on which to evaluate the mode vectors
        mode: string, optional
            specify if the mode vectors are far- or near-field
        precompute: bool
            if True, the whole look-up table is computed in advance
            (default False)
        """

        if nfft % 2 == 1:
            raise ValueError("Signal length must be even.")

        # this flag controls if the look-up table should be stored
        # or computed on the fly
        self.precompute = precompute

        # short hands for propagation vectors, upped to 3D array
        p_x = grid.x[None, None, :]
        p_y = grid.y[None, None, :]
        p_z = grid.z[None, None, :]

        # short hands for microphone locations, upped to 3D array
        r_x = L[0, None, :, None]
        r_y = L[1, None, :, None]

        if L.shape[0] == 3:
            r_z = L[2, None, :, None]
        else:
            r_z = np.zeros((1, L.shape[1], 1))

        # Here we compute the time of flights from source candidate locations
        # to microphones
        if mode == "near":
            # distance
            dist = np.sqrt((p_x - r_x) ** 2 + (p_y - r_y) ** 2 + (p_z - r_z) ** 2)

        elif mode == "far":
            # projection
            dist = (p_x * r_x) + (p_y * r_y) + (p_z * r_z)

        # shape (nfft // 2 + 1)
        self.tau = dist / c
        print(np.shape(self.tau))
        print(p_x[:, :, 16041], p_y[:, :, 16041], p_z[:, :, 16041])
        print(p_x[:, :, 1225], p_y[:, :, 1225], p_z[:, :, 1225])
        print(np.shape(r_x))
        print(r_x[:, 40, :], r_y[:, 40, :], r_z[:, 40, :])
        print(r_x[:, 41, :], r_y[:, 41, :], r_z[:, 41, :])

        print(dist[0, 40:42, 13889])
        print(dist[0, 40:42, 10475])

        # shape (1, num_mics, grid_size)
        self.omega = 2 * np.pi * fs * np.arange(nfft // 2 + 1) / nfft

        if precompute:
            self.mode_vec = np.exp(1j * self.omega[:, None, None] * self.tau)
        else:
            self.mode_vec = None

    def __getitem__(self, ref):

        # If the look up table was precomputed
        if self.precompute:
            return self.mode_vec[ref]

        # we use this to test if an integer is passed
        integer = (
            int,
            np.int,
            np.int16,
            np.int32,
            np.int64,
            np.uint,
            np.uint16,
            np.uint32,
            np.uint64,
        )

        # Otherwise compute values on the fly
        if isinstance(ref[1], integer) and isinstance(ref[2], integer):
            w = self.omega[ref[0]]
        elif isinstance(ref[1], integer) or isinstance(ref[2], integer):
            w = self.omega[ref[0], None]
        else:
            w = self.omega[ref[0], None, None]

        if isinstance(ref[0], integer):
            tref0 = 0
        else:
            tref0 = slice(None, None, None)

        if len(ref) == 1:
            return np.exp(1j * w * self.tau[tref0, :, :])
        elif len(ref) == 2:
            return np.exp(1j * w * self.tau[tref0, ref[1], :])
        elif len(ref) == 3:
            return np.exp(1j * w * self.tau[tref0, ref[1], ref[2]])
        else:
            raise ValueError("Too many axis")


class SPIRE_MM(DOA):
    """
    Class to apply SPIRE_MM (SPIRE_MM) direction-of-arrival (DoA) for 
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
        **kwargs
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
            **kwargs
        )

        self.num_pairs = self.M * (self.M - 1) / 2

        n_rough_grid = kwargs["n_rough_grid"]

        # mic_pair
        self.mic_pairs = kwargs["mic_pairs"]

        self._make_rough_mode_vectors(n_rough_grid=n_rough_grid)

        # n_channels, n_dim
        self.mic_positions = kwargs["mic_positions"]

        # MM法のIteration数
        self.n_mm_itertaions = kwargs["n_mm_iterations"]

        # 二分法のIteration数
        self.n_bisec_search = kwargs["n_bisec_search"]

    def _make_rough_mode_vectors(self, n_rough_grid=None):
        # Use a default grid size
        if self.dim == 2:
            if n_rough_grid is None:
                n_rough_grid = 36

            self.rough_grid = GridCircle(n_points=n_rough_grid)

        elif self.dim == 3:
            if n_rough_grid is None:
                n_rough_grid = 18 * 9

            self.rough_grid = GridSphere(n_points=n_rough_grid)

        # build lookup table to candidate locations from r, azimuth, colatitude
        from pyroomacoustics.doa.frida import FRIDA

        if not isinstance(self, FRIDA):
            self.rough_mode_vec = ModeVector(
                self.L, self.fs, self.nfft, self.c, self.rough_grid
            )

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response 
        spectrum.
        """
        # 周波数毎に実施する
        ones = np.ones(self.L.shape[1])

        spire_cost = np.zeros(self.grid.n_points)

        # 初期のポジションベクトル
        n_channels = np.shape(X)[0]
        n_freq_bins = np.shape(X)[1]
        n_frames = np.shape(X)[2]

        d = None
        n_mic_pair = 0
        # for m1 in range(1):

        step = 2

        mic_pairs = self.mic_pairs
        # mic_pairs=[[m1,m2] for m1 in range(n_channels-1) for m2 in range(m1+1,np.minimum(m1+step+1,n_channels)) ]
        mic_pairs = np.array(mic_pairs)

        n_mic_pair = np.shape(mic_pairs)[0]
        d = np.array(self.mic_positions[mic_pairs[:, 1]]) - np.array(
            self.mic_positions[mic_pairs[:, 0]]
        )
        # d: n_mic_pair,dim

        # print("hogehoge")

        # 時間周波数毎の初期のポジションベクトル
        position_vector = np.zeros(shape=(n_freq_bins, n_frames, self.dim))

        X_temp = X[:, self.freq_bins, :]

        sigma = np.angle(X_temp[mic_pairs[:, 1], ...] / X_temp[mic_pairs[:, 0], ...])
        sigma = np.transpose(sigma, (1, 2, 0))

        sigma = np.where(np.abs(sigma) < 1.0e-18, np.zeros_like(sigma) + 1.0e-18, sigma)
        z = np.zeros(shape=(n_freq_bins, n_frames, n_mic_pair), dtype=np.int)
        x = np.random.normal(size=n_freq_bins * n_frames * n_mic_pair)
        x = np.reshape(x, newshape=(n_freq_bins, n_frames, n_mic_pair))
        # 初期化
        mode_vec = self.rough_mode_vec[self.freq_bins, :, :]
        mode_vec = np.conjugate(mode_vec)
        # print(mode_vec)
        prod = np.einsum("fmi,mft->fti", mode_vec, X[:, self.freq_bins, :])
        # prod=np.einsum("mi,mt->ti",mode_vec,X[:,k,:])
        amp = np.abs(prod)
        # print(k,np.max(amp))
        # ft
        index = np.argmax(amp, axis=-1)
        org_shape = np.shape(index)
        index = np.reshape(index, [-1])

        # indexに相当する方向を取る
        if self.dim == 2:
            rough_azimuth_recon = self.rough_grid.azimuth[index]
            # ダミー
            rough_colatitude_recon = np.zeros_like(rough_azimuth_recon) + np.pi
        elif self.dim == 3:
            rough_azimuth_recon = self.rough_grid.azimuth[index]
            rough_colatitude_recon = self.rough_grid.colatitude[index]

        doas = np.concatenate(
            (
                rough_colatitude_recon[:, None],  # colatitude [0, pi]
                rough_azimuth_recon[:, None],  # azimuth [0, 2 pi]
            ),
            axis=-1,
        )
        distance = 3.0

        # source_locations: 3, n_frames
        source_locations = geom.spherical_to_cartesian(doa=doas, distance=distance)
        source_locations = np.reshape(source_locations, (3, org_shape[0], org_shape[1]))

        position_vector[self.freq_bins, :, :] = np.transpose(
            source_locations[: self.dim, :, :], (1, 2, 0)
        )

        size = np.einsum("fti,fti->ft", np.conjugate(position_vector), position_vector)
        size = np.sqrt(size)[..., np.newaxis]
        position_vector = position_vector / np.maximum(size, 1.0e-18)

        use_clustering = False
        cluster_index = np.random.randint(0, self.num_src, size=n_freq_bins * n_frames)
        cluster_index = np.reshape(cluster_index, (n_freq_bins, n_frames))
        cluster_center = np.random.normal(size=self.num_src * self.dim)
        cluster_center = np.reshape(cluster_center, newshape=(self.num_src, self.dim))
        size = np.einsum("ci,ci->c", np.conjugate(cluster_center), cluster_center)
        size = np.sqrt(size)[..., np.newaxis]
        cluster_center = cluster_center / np.maximum(size, 1.0e-18)
        if use_clustering == True:
            # pを作る
            for k in self.freq_bins:
                for l in range(n_frames):
                    position_vector[k, l, :] = cluster_center[cluster_index[k, l], :]

        # print("start")
        est_p = position_vector[self.freq_bins, ...]
        z = z[self.freq_bins, ...]
        x = x[self.freq_bins, ...]
        freqs = self.freq_hz
        cluster_index = cluster_index[self.freq_bins, ...]

        silent_mode = True
        freqs_d = np.einsum("f,pi->fpi", freqs, d)
        for i in range(self.n_mm_itertaions):
            #
            (
                org_cost_0,
                org_cost_1,
                org_cost_2,
                org_cost_3,
                cost_0,
                cost_1,
                cost_2,
                cost_3,
                est_p,
                z,
                x,
            ) = doa_estimation_one_iteration(
                freqs_d,
                est_p,
                sigma,
                z,
                x,
                cluster_index=cluster_index,
                cluster_center=cluster_center,
                iter_num2=self.n_bisec_search,
                silent_mode=silent_mode,
            )
            # print(cost_1-cost_0,cost_2-cost_1,cost_3-cost_2)
            # print(org_cost_0,org_cost_3,org_cost_3-org_cost_0)
            if silent_mode == False:
                print(cost_0, cost_1, cost_2, cost_3)
        # est_pから
        # fti
        position_vector[self.freq_bins, ...] = est_p

        size = np.einsum("fti,fti->ft", np.conjugate(position_vector), position_vector)
        size = np.sqrt(size)[..., np.newaxis]
        position_vector = position_vector / np.maximum(size, 1.0e-18)

        # gridを探す

        # position_vectorに相当する方向を取る
        if self.dim == 2:
            azimuth_recon = self.grid.azimuth
            # ダミー
            colatitude_recon = np.zeros_like(azimuth_recon) + np.pi
        elif self.dim == 3:
            azimuth_recon = self.grid.azimuth
            colatitude_recon = self.grid.colatitude

        doas = np.concatenate(
            (
                colatitude_recon[:, None],  # colatitude [0, pi]
                azimuth_recon[:, None],  # azimuth [0, 2 pi]
            ),
            axis=-1,
        )
        distance = 3.0
        # source_locations: 3, n_grid_num
        grid_locations = geom.spherical_to_cartesian(doa=doas, distance=distance)
        size = np.einsum("in,in->n", np.conjugate(grid_locations), grid_locations)
        size = np.sqrt(size)[np.newaxis, ...]
        grid_locations = grid_locations / np.maximum(size, 1.0e-18)

        # kd treeを使って探索
        # print("start KD tree")
        # tree=spatial.KDTree(grid_locations.T)
        # print("end KD tree")

        # position_vector=np.reshape(position_vector,[-1,np.shape(position_vector)[2]])
        # _,grid_index=tree.query(position_vector)
        # print("end query")
        # for n in range(self.grid.n_points):
        #    spire_cost[n]=spire_cost[n]+np.count_nonzero(grid_index==n)

        grid_index_buf = []
        for k in self.freq_bins:
            # print(k)
            prod = np.einsum("in,ti->tn", grid_locations, position_vector[k, ...])
            grid_index = np.argmax(prod, axis=-1)
            grid_index_buf.append(grid_index)
        grid_index_buf = np.array(grid_index_buf)

        for n in range(self.grid.n_points):
            spire_cost[n] = spire_cost[n] + np.count_nonzero(grid_index_buf == n)

        """
        # Same code, but with a kd-tree (Robin version)
        tree = spatial.cKDTree(self.grid.cartesian.T)
        _, nn = tree.query(position_vector.reshape((-1, position_vector.shape[-1])))
        bin_indices, bin_count = np.unique(nn, return_counts=True)
        spire_cost = np.zeros(self.grid.n_points, dtype=np.int)
        spire_cost[bin_indices] = bin_count
        """

        self.grid.set_values(spire_cost)


class SPIRE_MM_CIRCULAR(DOA):
    """
    Class to apply SPIRE_MM (SPIRE_MM) direction-of-arrival (DoA) for 
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
        **kwargs
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
            **kwargs
        )

        self.num_pairs = self.M * (self.M - 1) / 2

        n_rough_grid = kwargs["n_rough_grid"]

        # mic_pair
        self.mic_pairs = kwargs["mic_pairs"]

        # reject閾値
        self.reject_th = kwargs["reject_th"]

        self._make_rough_mode_vectors(n_rough_grid=n_rough_grid)

        # n_channels, n_dim
        self.mic_positions = kwargs["mic_positions"]

        # MM法のIteration数
        self.n_mm_itertaions = kwargs["n_mm_iterations"]

        # 二分法のIteration数
        self.n_bisec_search = kwargs["n_bisec_search"]

    def _make_rough_mode_vectors(self, n_rough_grid=None):
        # Use a default grid size
        if self.dim == 2:
            if n_rough_grid is None:
                n_rough_grid = 36

            self.rough_grid = GridCircle(n_points=n_rough_grid)

        elif self.dim == 3:
            if n_rough_grid is None:
                n_rough_grid = 18 * 9

            self.rough_grid = GridSphere(n_points=n_rough_grid)

        # build lookup table to candidate locations from r, azimuth, colatitude
        from pyroomacoustics.doa.frida import FRIDA

        if not isinstance(self, FRIDA):
            self.rough_mode_vec = ModeVector(
                self.L, self.fs, self.nfft, self.c, self.rough_grid
            )

    def _process(self, X):
        """
        Perform SRP-PHAT for given frame in order to estimate steered response 
        spectrum.
        """
        # 周波数毎に実施する
        ones = np.ones(self.L.shape[1])

        spire_cost = np.zeros(self.grid.n_points)

        # 初期のポジションベクトル
        n_channels = np.shape(X)[0]
        n_freq_bins = np.shape(X)[1]
        n_frames = np.shape(X)[2]

        d = None
        n_mic_pair = 0
        # for m1 in range(1):

        step = 2

        mic_pairs = self.mic_pairs
        # mic_pairs=[[m1,m2] for m1 in range(n_channels-1) for m2 in range(m1+1,np.minimum(m1+step+1,n_channels)) ]
        mic_pairs = np.array(mic_pairs)

        n_mic_pair = np.shape(mic_pairs)[0]
        d = np.array(self.mic_positions[mic_pairs[:, 1]]) - np.array(
            self.mic_positions[mic_pairs[:, 0]]
        )
        # d: n_mic_pair,dim

        # print("hogehoge")

        # 時間周波数毎の初期のポジションベクトル
        position_vector = np.zeros(shape=(n_freq_bins, n_frames, self.dim))

        X_temp = X[:, self.freq_bins, :]

        sigma = np.angle(X_temp[mic_pairs[:, 1], ...] / X_temp[mic_pairs[:, 0], ...])
        sigma = np.transpose(sigma, (1, 2, 0))

        sigma = np.where(np.abs(sigma) < 1.0e-18, np.zeros_like(sigma) + 1.0e-18, sigma)
        z = np.zeros(shape=(n_freq_bins, n_frames, n_mic_pair), dtype=np.int)
        x = np.random.normal(size=n_freq_bins * n_frames * n_mic_pair)
        x = np.reshape(x, newshape=(n_freq_bins, n_frames, n_mic_pair))
        # 初期化
        mode_vec = self.rough_mode_vec[self.freq_bins, :, :]
        mode_vec = np.conjugate(mode_vec)
        # print(mode_vec)
        prod = np.einsum("fmi,mft->fti", mode_vec, X[:, self.freq_bins, :])
        # prod=np.einsum("mi,mt->ti",mode_vec,X[:,k,:])
        amp = np.abs(prod)
        # print(k,np.max(amp))
        # ft
        index = np.argmax(amp, axis=-1)
        org_shape = np.shape(index)
        index = np.reshape(index, [-1])

        # indexに相当する方向を取る
        if self.dim == 2:
            rough_azimuth_recon = self.rough_grid.azimuth[index]
            # ダミー
            rough_colatitude_recon = np.zeros_like(rough_azimuth_recon) + np.pi
        elif self.dim == 3:
            rough_azimuth_recon = self.rough_grid.azimuth[index]
            rough_colatitude_recon = self.rough_grid.colatitude[index]

        doas = np.concatenate(
            (
                rough_colatitude_recon[:, None],  # colatitude [0, pi]
                rough_azimuth_recon[:, None],  # azimuth [0, 2 pi]
            ),
            axis=-1,
        )
        distance = 3.0

        # source_locations: 3, n_frames
        source_locations = geom.spherical_to_cartesian(doa=doas, distance=distance)
        source_locations = np.reshape(source_locations, (3, org_shape[0], org_shape[1]))

        position_vector[self.freq_bins, :, :] = np.transpose(
            source_locations[: self.dim, :, :], (1, 2, 0)
        )

        size = np.einsum("fti,fti->ft", np.conjugate(position_vector), position_vector)
        size = np.sqrt(size)[..., np.newaxis]
        position_vector = position_vector / np.maximum(size, 1.0e-18)

        use_clustering = False
        cluster_index = np.random.randint(0, self.num_src, size=n_freq_bins * n_frames)
        cluster_index = np.reshape(cluster_index, (n_freq_bins, n_frames))
        cluster_center = np.random.normal(size=self.num_src * self.dim)
        cluster_center = np.reshape(cluster_center, newshape=(self.num_src, self.dim))
        size = np.einsum("ci,ci->c", np.conjugate(cluster_center), cluster_center)
        size = np.sqrt(size)[..., np.newaxis]
        cluster_center = cluster_center / np.maximum(size, 1.0e-18)
        if use_clustering == True:
            # pを作る
            for k in self.freq_bins:
                for l in range(n_frames):
                    position_vector[k, l, :] = cluster_center[cluster_index[k, l], :]

        # print("start")
        est_p = position_vector[self.freq_bins, ...]
        z = z[self.freq_bins, ...]
        x = x[self.freq_bins, ...]
        freqs = self.freq_hz
        cluster_index = cluster_index[self.freq_bins, ...]

        silent_mode = True
        freqs_d = np.einsum("f,pi->fpi", freqs, d)
        x_non_const_power_vector = np.zeros(shape=(n_freq_bins, n_frames))

        for i in range(self.n_mm_itertaions):
            #
            # org_cost_0,org_cost_1,org_cost_2,org_cost_3,cost_0,cost_1,cost_2,cost_3,est_p,z,x= doa_estimation_one_iteration(freqs_d,est_p,sigma,z,x,use_clustering=use_clustering,cluster_index=cluster_index,cluster_center=cluster_center,iter_num2=self.n_bisec_search,silent_mode=silent_mode)
            (
                org_cost_0,
                org_cost_1,
                org_cost_2,
                org_cost_3,
                cost_0,
                cost_1,
                cost_2,
                cost_3,
                est_p,
                z,
                x,
                x_non_const_power,
            ) = coplaner_doa_estimation_one_iteration(
                freqs_d,
                est_p,
                sigma,
                z,
                x,
                use_clustering=use_clustering,
                cluster_index=cluster_index,
                cluster_center=cluster_center,
                iter_num2=self.n_bisec_search,
                silent_mode=silent_mode,
                zero_feature_index=2,
            )

            # print(cost_1-cost_0,cost_2-cost_1,cost_3-cost_2)
            # print(org_cost_0,org_cost_3,org_cost_3-org_cost_0)
            if silent_mode == False:
                print(cost_0, cost_1, cost_2, cost_3)

        # est_pから
        # fti
        position_vector[self.freq_bins, ...] = est_p

        x_non_const_power_vector[self.freq_bins, :] = x_non_const_power[:, :, 0]

        size = np.einsum("fti,fti->ft", np.conjugate(position_vector), position_vector)
        size = np.sqrt(size)[..., np.newaxis]
        position_vector = position_vector / np.maximum(size, 1.0e-18)

        # gridを探す

        # position_vectorに相当する方向を取る
        if self.dim == 2:
            azimuth_recon = self.grid.azimuth
            # ダミー
            colatitude_recon = np.zeros_like(azimuth_recon) + np.pi
        elif self.dim == 3:
            azimuth_recon = self.grid.azimuth
            colatitude_recon = self.grid.colatitude

        doas = np.concatenate(
            (
                colatitude_recon[:, None],  # colatitude [0, pi]
                azimuth_recon[:, None],  # azimuth [0, 2 pi]
            ),
            axis=-1,
        )
        distance = 3.0
        # source_locations: 3, n_grid_num
        grid_locations = geom.spherical_to_cartesian(doa=doas, distance=distance)
        size = np.einsum("in,in->n", np.conjugate(grid_locations), grid_locations)
        size = np.sqrt(size)[np.newaxis, ...]
        grid_locations = grid_locations / np.maximum(size, 1.0e-18)

        # kd treeを使って探索
        # tree=spatial.KDTree(grid_locations.T)
        # _,grid_index=tree.query(position_vector)
        # for n in range(self.grid.n_points):
        #    spire_cost[n]=spire_cost[n]+np.count_nonzero(grid_index==n)

        grid_index_buf = []

        # 制約なし解のパワーが1を大幅に超えて居たらReject
        print(np.average(x_non_const_power_vector))
        valid_index = x_non_const_power_vector < self.reject_th
        for k in self.freq_bins:
            # print(k)
            # frame
            # valid_index[k,:]

            prod = np.einsum("in,ti->tn", grid_locations, position_vector[k, ...])
            grid_index = np.argmax(prod, axis=-1)

            # print(np.shape(grid_index))
            # print(np.shape(valid_index))

            grid_index = grid_index[valid_index[k, :]]

            grid_index_buf.append(grid_index)
        grid_index_buf = np.array(grid_index_buf)

        for n in range(self.grid.n_points):
            spire_cost[n] = spire_cost[n] + np.count_nonzero(grid_index_buf == n)

        self.grid.set_values(spire_cost)


# a: freq,sample,feature
# r: freq,time,sample
# alpha: freq,time, sample
# 最小二乗解。3次元だが、最後の1次元が落ちている。Planer Array
def coplanar_least_squares_st_norm_one(
    a, r, alpha, zero_feature_index=0, iter_num=10, eps=1.0e-18
):
    feature_num = np.shape(a)[-1]
    coef = np.sqrt(feature_num)

    a_temp = np.delete(a, zero_feature_index, axis=2)

    cov = np.einsum("fti,fid,fik->ftdk", alpha, a_temp, np.conjugate(a_temp))
    cor = np.einsum("fti,fid,fti->ftd", alpha, a_temp, np.conjugate(r))
    cov_H = np.conjugate(np.swapaxes(cov.copy(), -2, -1))
    cov_res = (cov_H + cov) / 2.0
    W, V = np.linalg.eigh(cov_res)
    # VWV^H
    # inv: V^-H W^-1 V^-1  = V W^-1 V^H
    p = np.einsum("ftdk,ftd->ftk", np.conjugate(V), cor)

    def make_x(V, inv_lamb_W, p):
        x = np.einsum("ftij,ftj,ftj->fti", V, inv_lamb_W, p)
        return x

    p_2 = np.einsum("ftk,ftk->ftk", p, np.conjugate(p))
    coef_p_abs = coef * np.abs(p)

    # 左端を求める
    left_side = -1.0 * np.min(np.real(W), axis=-1)
    # 右端を求める
    right_side = np.max(coef_p_abs - np.real(W), axis=-1)

    inv_W = 1.0 / np.maximum(np.real(W), eps)

    # ft dim
    x_non_const = make_x(V, inv_W, p)

    # x_constを求める。

    def cost(lamb):
        value = (
            np.sum(
                p_2 / np.maximum(np.square(lamb[..., np.newaxis] + np.real(W)), eps),
                axis=-1,
            )
            - 1.0
        )
        return value

    # print(cost(left_side))
    # print("hogehoge")
    # print(cost(right_side))
    for i in range(iter_num):
        # print("size")
        # size=np.einsum("bi,bi->b",np.conjugate(temp_x),temp_x)
        # print(size-1.0)
        # print("confirm")
        # confirm_solutions(a,r,alpha,left_side)
        # print(cost(left_side))
        # print("hogehgoe")

        # confirm_solutions(a,r,right_side)
        # print(cost(right_side))
        # print("hogehoge2")
        # print("--i {}".format(i))
        mid = (left_side + right_side) / 2.0
        val = cost(mid)
        new_left_side = np.where(val > 0, mid, left_side)
        new_right_side = np.where(val > 0, right_side, mid)
        left_side = new_left_side
        right_side = new_right_side

        # print(cost(left_side)[10])
        # print(cost(right_side)[10])
        # print(np.shape(left_side))
        # print("hogehoge")
        # print(cost(right_side))
    lamb = left_side
    # print(lamb)
    # print(np.shape(lamb))
    inv_lamb_W = 1.0 / np.maximum(lamb[..., np.newaxis] + np.real(W), eps)

    x_const = make_x(V, inv_lamb_W, p)
    x_non_const_power = np.einsum("fti,fti->ft", x_non_const, np.conjugate(x_non_const))
    x_non_const_power = np.real(np.zeros_like(x_const)) + x_non_const_power[..., None]
    # print(np.shape(x_non_const_power))
    # print(np.shape(lamb))
    # for k in range(np.shape(x_non_const_power)[0]):
    #    print(np.concatenate((x_non_const_power[...,0],lamb),axis=-1)[k,...])
    # exit(0)
    x_result = np.where(x_non_const_power < 1, x_non_const, x_const)
    x_power = np.einsum("fti,fti->ft", x_result, np.conjugate(x_result))
    # print(x_power[x_power>1])

    y_result = np.sqrt(np.maximum(1.0 - x_power, eps))

    pre = zero_feature_index - 0
    post = feature_num - zero_feature_index
    if pre == 0:
        x_result = np.concatenate((y_result[..., None], x_result), axis=-1)
    elif post == 0:
        x_result = np.concatenate((x_result, y_result[..., None]), axis=-1)
    else:
        x_result = np.concatenate(
            (
                x_result[..., :zero_feature_index],
                y_result[..., None],
                x_result[..., zero_feature_index:],
            ),
            axis=-1,
        )

    return (lamb, x_result, x_non_const_power)


# 最小二乗解。
# a: freq,sample,feature
# r: freq,time,sample
# alpha: freq,time, sample
# feaure=2, feature_index=0の要素がゼロ以外
def linear_least_squares_st_norm_one(a, r, alpha, feature_index=0, eps=1.0e-18):
    # cov: batch,feature,feature
    feature_num = np.shape(a)[-1]
    if feature_num != 2:
        print("dimension error")

    coef = np.sqrt(feature_num)

    cov = np.einsum(
        "fti,fi,fi->ft",
        alpha,
        a[..., feature_index],
        np.conjugate(a[..., feature_index]),
    )
    cor = np.einsum("fti,fi,fti->ft", alpha, a[..., feature_index], np.conjugate(r))

    x_non_const = cor / np.maximum(cov, eps)
    x_const = cor / np.maximum(np.abs(cor), eps)
    x_result = np.where(np.abs(x_non_const) < 1, x_non_const, x_const)
    y_result = np.sqrt(np.maximum(1.0 - x_result * x_result, eps))
    if feature_index == 0:
        temp_x = np.concatenate(
            (x_result[..., np.newaxis], y_result[..., np.newaxis]), axis=-1
        )
    else:
        temp_x = np.concatenate(
            (y_result[..., np.newaxis], x_result[..., np.newaxis]), axis=-1
        )

    return temp_x
    # cov2=np.einsum("...ki,...i,...hi->...kh",V,W,np.conjugate(V))


# 最小二乗解。
# a: freq,sample,feature
# r: freq,time,sample
# alpha: freq,time, sample
def least_squares_st_norm_one(a, r, alpha, iter_num=10, eps=1.0e-18):

    # cov: batch,feature,feature
    feature_num = np.shape(a)[-1]
    coef = np.sqrt(feature_num)

    cov = np.einsum("fti,fid,fik->ftdk", alpha, a, np.conjugate(a))
    cor = np.einsum("fti,fid,fti->ftd", alpha, a, np.conjugate(r))
    cov_H = np.conjugate(np.swapaxes(cov.copy(), -2, -1))
    cov_res = (cov_H + cov) / 2.0
    W, V = np.linalg.eigh(cov_res)
    # VWV^H
    # inv: V^-H W^-1 V^-1  = V W^-1 V^H
    p = np.einsum("ftdk,ftd->ftk", np.conjugate(V), cor)

    def make_x(V, inv_lamb_W, p):
        x = np.einsum("ftij,ftj,ftj->fti", V, inv_lamb_W, p)
        return x

    p_2 = np.einsum("ftk,ftk->ftk", p, np.conjugate(p))
    coef_p_abs = coef * np.abs(p)
    # 左端を求める
    left_side = -1.0 * np.min(np.real(W), axis=-1)
    # 右端を求める
    right_side = np.max(coef_p_abs - np.real(W), axis=-1)

    def cost(lamb):
        value = (
            np.sum(
                p_2 / np.maximum(np.square(lamb[..., np.newaxis] + np.real(W)), eps),
                axis=-1,
            )
            - 1.0
        )
        return value

    # print(cost(left_side))
    # print("hogehoge")
    # print(cost(right_side))
    for i in range(iter_num):
        # print("size")
        # size=np.einsum("bi,bi->b",np.conjugate(temp_x),temp_x)
        # print(size-1.0)
        # print("confirm")
        # confirm_solutions(a,r,alpha,left_side)
        # print(cost(left_side))
        # print("hogehgoe")

        # confirm_solutions(a,r,right_side)
        # print(cost(right_side))
        # print("hogehoge2")
        # print("--i {}".format(i))
        mid = (left_side + right_side) / 2.0
        val = cost(mid)
        new_left_side = np.where(val > 0, mid, left_side)
        new_right_side = np.where(val > 0, right_side, mid)
        left_side = new_left_side
        right_side = new_right_side

        # print(cost(left_side)[10])
        # print(cost(right_side)[10])
        # print(np.shape(left_side))
        # print("hogehoge")
        # print(cost(right_side))
    lamb = left_side
    inv_lamb_W = 1.0 / np.maximum(lamb[..., np.newaxis] + np.real(W), eps)

    temp_x = make_x(V, inv_lamb_W, p)
    return (lamb, temp_x)
    # cov2=np.einsum("...ki,...i,...hi->...kh",V,W,np.conjugate(V))


# 補助関数の値
def calc_auxiliary_function_cost(
    freqs_d, p, sigma, z, x, SOUND_SPEED=343.0, eps=1.0e-8
):
    # 補助変数を更新する
    # tau=-np.einsum("pd,ftd->ftp",d,p)/SOUND_SPEED
    two_pi_f_tau = -np.einsum("fpd,ftd->ftp", freqs_d, p) * (2.0 * np.pi / SOUND_SPEED)

    # temp=2.*np.pi*np.einsum("f,ftp->ftp",freqs,tau)+sigma+2.*np.pi*z
    temp = two_pi_f_tau + sigma + 2.0 * np.pi * z
    org_cost = np.cos(temp)
    sign_x = np.sign(x)
    sign_x = np.where(np.abs(sign_x) < 0.3, np.ones_like(sign_x), sign_x)
    x_eps = np.maximum(np.abs(x), eps) * sign_x

    alpha = np.where(
        np.abs(x) < eps, -0.5 * np.ones_like(x), -0.5 * np.sin(x_eps) / x_eps
    )
    # alpha_x=-0.5*np.sin(x_eps)/x_eps

    cost = alpha * temp * temp + np.cos(x_eps) + 0.5 * x_eps * np.sin(x_eps)
    cost = np.sum(cost, axis=-1)
    cost = np.average(cost)

    org_cost = np.sum(org_cost, axis=-1)
    org_cost = np.average(org_cost)

    return (org_cost, cost)


# 一次元の音源方向推定
# 特別扱い。featureは2次元だが、ランク落ちしているので注意。(xもしくはyだけが値を持っている。)
# freqs_d: freq,pair,feature
# non_zero_feature_index: 値を持っている軸を指定するインデックス
# d: pair,feature
# p: freq,frame,feature
# sigma: freq,frame,pair,
# z: freq,frame, pair
# x: freq,frame,pair
# freqs: freq
# use_clustering: K-meansクラスタリングにより音源方向を求めるか
# cluster_index: freq,frame
# cluster_center: cluster_num,feature
def linear_doa_estimation_one_iteration(
    freqs_d,
    non_zero_feature_index,
    p,
    sigma,
    z,
    x,
    use_clustering=False,
    cluster_index=None,
    cluster_center=None,
    SOUND_SPEED=343.0,
    silent_mode=False,
    eps=1.0e-18,
):
    freq_num = np.shape(p)[0]
    frame_num = np.shape(p)[1]
    pair_num = np.shape(freqs_d)[1]
    feature_dim = np.shape(freqs_d)[2]
    if feature_dim > 2:
        print("feature dimension error\n")

    # 補助関数とかは一切変化しない

    # print(x[400,0,:])
    if silent_mode == False:
        org_cost_0, cost_0 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_0 = 0
        cost_0 = 0

    # 補助変数を更新する
    # tau=-np.einsum("pd,ftd->ftp",d,p)/SOUND_SPEED
    two_pi_tau = -np.einsum("fpd,ftd->ftp", freqs_d, p) * (2.0 * np.pi / SOUND_SPEED)

    coef = two_pi_tau + sigma

    right_side = 0.5 - coef / (2.0 * np.pi)
    # zを変更
    z = np.floor(right_side)

    if silent_mode == False:
        org_cost_1, cost_1 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_1 = 0
        cost_1 = 0

    x = coef + 2.0 * np.pi * z

    # xを変更

    if silent_mode == False:
        org_cost_2, cost_2 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_2 = 0
        cost_2 = 0

    sign_x = np.sign(x)
    sign_x = np.where(np.abs(sign_x) < 0.3, np.ones_like(sign_x), sign_x)
    x_eps = np.maximum(np.abs(x), eps) * sign_x

    alpha = np.where(np.abs(x) < 1.0e-8, np.ones_like(x), np.sin(x_eps) / x_eps)

    # ここから最小解を求める
    # r: ftmn
    # a: ftmnd
    # alpha: ftmn
    r = sigma + 2.0 * np.pi * z
    a = freqs_d * (2.0 * np.pi / SOUND_SPEED)

    # a=2.*np.pi*np.einsum("f,pd->fpd",freqs,d)/SOUND_SPEED

    r = np.reshape(r, (freq_num, frame_num, pair_num))
    a = np.reshape(a, (freq_num, pair_num, feature_dim))
    alpha = np.reshape(alpha, (freq_num, frame_num, pair_num))

    if use_clustering == True:
        # k-meansクラスタリングにより音源方向推定実施する
        cluster_num = np.shape(cluster_center)[0]

        # cluster_indexを求める
        alpha2 = np.reshape(alpha, (freq_num, frame_num, pair_num))
        two_pi_f_cluster_tau = -np.einsum("fpd,cd->fcp", freqs_d, cluster_center) * (
            2.0 * np.pi / SOUND_SPEED
        )

        # cluster_tau=-np.einsum("pd,cd->cp",d,cluster_center)/SOUND_SPEED
        cluster_x = (
            two_pi_f_cluster_tau[:, np.newaxis, :, :]
            + sigma[..., np.newaxis, :]
            + 2.0 * np.pi * z[..., np.newaxis, :]
        )

        # cluster_x=2.*np.pi*np.einsum("f,cp->fcp",freqs,cluster_tau)[:,np.newaxis,:,:]+sigma[...,np.newaxis,:]+2.*np.pi*z[...,np.newaxis,:]
        cluster_cost = np.einsum("ftp,ftcp->ftc", alpha2, cluster_x * cluster_x)

        # cluster_index: freq,time
        cluster_index = np.argmin(cluster_cost, axis=2)

        # pを作る
        for k in range(freq_num):
            for l in range(frame_num):
                p[k, l, :] = cluster_center[cluster_index[k, l], :]

        r2 = np.reshape(r, (freq_num, frame_num, pair_num))
        a2 = np.reshape(a, (freq_num, pair_num, feature_dim))

        for c in range(cluster_num):
            # cluster_indexがcとなる時間周波数成分を纏めて方向推定する
            a3 = None  # a3: comp, feature_dim
            r3 = None  # r3:comp
            alpha3 = None  # alpha3: comp
            for k in range(freq_num):
                for l in range(frame_num):
                    if cluster_index[k, l] == c:
                        if a3 is None:
                            a3 = a2[k, :, :]
                            r3 = r2[k, l, :]
                            alpha3 = alpha2[k, l, :]
                        else:
                            a3 = np.concatenate((a3, a2[k, :, :]), axis=0)
                            r3 = np.concatenate((r3, r2[k, l, :]), axis=0)
                            alpha3 = np.concatenate((alpha3, alpha2[k, l, :]), axis=0)
            #
            p_temp = linear_least_squares_st_norm_one(
                a3[np.newaxis, ...],
                r3[np.newaxis, np.newaxis, ...],
                alpha3[np.newaxis, np.newaxis, ...],
                feature_index=non_zero_feature_index,
                eps=eps,
            )

            # lamb_temp,p_temp=least_squares_st_norm_one(a3[np.newaxis,...],r3[np.newaxis,np.newaxis,...],alpha3[np.newaxis,np.newaxis,...],iter_num=iter_num2,eps=eps)
            # p_temp: 1, feature_dim
            cluster_center[c, :] = p_temp
        # pを作る
        for k in range(freq_num):
            for l in range(frame_num):
                p[k, l, :] = cluster_center[cluster_index[k, l], :]

        # 音源方向を求める
        # cluster_center: freq,time,cluster, pair

    else:
        p = linear_least_squares_st_norm_one(
            a, r, alpha, feature_index=non_zero_feature_index, eps=eps
        )

    # lamb: batch
    # lamb=np.reshape(lamb,(freq_num,frame_num))
    p = np.reshape(p, (freq_num, frame_num, feature_dim))
    # print(p)
    # pを変更

    if silent_mode == False:
        org_cost_3, cost_3 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_3 = 0
        cost_3 = 0
    return (
        org_cost_0,
        org_cost_1,
        org_cost_2,
        org_cost_3,
        cost_0,
        cost_1,
        cost_2,
        cost_3,
        p,
        z,
        x,
    )

    # lamb,p=least_squares_st_norm_one(a,r,alpha,iter_num=iter_num2,eps=eps)


# freqs_d: freq,pair,feature
# d: pair,feature
# p: freq,frame,feature
# sigma: freq,frame,pair,
# z: freq,frame, pair
# x: freq,frame,pair
# freqs: freq
# use_clustering: K-meansクラスタリングにより音源方向を求めるか
# cluster_index: freq,frame
# cluster_center: cluster_num,feature

# SOUND_SPEED: 343
def doa_estimation_one_iteration(
    freqs_d,
    p,
    sigma,
    z,
    x,
    cluster_index=None,
    cluster_center=None,
    iter_num2=100,
    SOUND_SPEED=343.0,
    silent_mode=False,
    eps=1.0e-18,
):
    freq_num = np.shape(p)[0]
    frame_num = np.shape(p)[1]
    pair_num = np.shape(freqs_d)[1]
    feature_dim = np.shape(freqs_d)[2]

    # print(x[400,0,:])
    if silent_mode == False:
        org_cost_0, cost_0 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_0 = 0
        cost_0 = 0

    # 補助変数を更新する
    two_pi_tau = -np.einsum("fpd,ftd->ftp", freqs_d, p) * (2.0 * np.pi / SOUND_SPEED)

    # zをどう決めるか
    coef = two_pi_tau + sigma

    # zを変更
    right_side = 0.5 - coef / (2.0 * np.pi)
    z = np.floor(right_side)

    if silent_mode == False:
        org_cost_1, cost_1 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_1 = 0
        cost_1 = 0

    # xを変更
    x = coef + 2.0 * np.pi * z

    if silent_mode == False:
        org_cost_2, cost_2 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_2 = 0
        cost_2 = 0

    sign_x = np.sign(x)
    sign_x = np.where(np.abs(sign_x) < 0.3, np.ones_like(sign_x), sign_x)
    x_eps = np.maximum(np.abs(x), eps) * sign_x

    alpha = np.where(np.abs(x) < 1.0e-8, np.ones_like(x), np.sin(x_eps) / x_eps)

    # ここから最小解を求める
    # r: ftmn
    # a: ftmnd
    # alpha: ftmn
    r = sigma + 2.0 * np.pi * z
    a = freqs_d * (2.0 * np.pi / SOUND_SPEED)

    r = np.reshape(r, (freq_num, frame_num, pair_num))
    a = np.reshape(a, (freq_num, pair_num, feature_dim))
    alpha = np.reshape(alpha, (freq_num, frame_num, pair_num))

    lamb, p = least_squares_st_norm_one(a, r, alpha, iter_num=iter_num2, eps=eps)

    # lamb: batch
    # lamb=np.reshape(lamb,(freq_num,frame_num))
    p = np.reshape(p, (freq_num, frame_num, feature_dim))
    # pを変更

    if silent_mode == False:
        org_cost_3, cost_3 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_3 = 0
        cost_3 = 0
    return (
        org_cost_0,
        org_cost_1,
        org_cost_2,
        org_cost_3,
        cost_0,
        cost_1,
        cost_2,
        cost_3,
        p,
        z,
        x,
    )


# freqs_d: freq,pair,feature
# d: pair,feature
# p: freq,frame,feature
# sigma: freq,frame,pair,
# z: freq,frame, pair
# x: freq,frame,pair
# freqs: freq

# SOUND_SPEED: 343
def coplaner_doa_estimation_one_iteration(
    freqs_d,
    p,
    sigma,
    z,
    x,
    iter_num2=100,
    SOUND_SPEED=343.0,
    silent_mode=False,
    zero_feature_index=2,
    eps=1.0e-18,
):
    freq_num = np.shape(p)[0]
    frame_num = np.shape(p)[1]
    pair_num = np.shape(freqs_d)[1]
    feature_dim = np.shape(freqs_d)[2]

    # print(x[400,0,:])
    if silent_mode == False:
        org_cost_0, cost_0 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_0 = 0
        cost_0 = 0

    # 補助変数を更新する
    two_pi_tau = -np.einsum("fpd,ftd->ftp", freqs_d, p) * (2.0 * np.pi / SOUND_SPEED)

    # zをどう決めるか
    # coef=2.*np.pi*np.einsum("f,ftp->ftp",freqs,tau)+sigma
    coef = two_pi_tau + sigma

    right_side = 0.5 - coef / (2.0 * np.pi)
    # zを変更
    z = np.floor(right_side)

    if silent_mode == False:
        org_cost_1, cost_1 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_1 = 0
        cost_1 = 0
    # print(cost_0,cost_1)

    x = coef + 2.0 * np.pi * z

    # xを変更

    if silent_mode == False:
        org_cost_2, cost_2 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_2 = 0
        cost_2 = 0

    sign_x = np.sign(x)
    sign_x = np.where(np.abs(sign_x) < 0.3, np.ones_like(sign_x), sign_x)
    x_eps = np.maximum(np.abs(x), eps) * sign_x

    alpha = np.where(np.abs(x) < 1.0e-8, np.ones_like(x), np.sin(x_eps) / x_eps)

    # ここから最小解を求める
    # r: ftmn
    # a: ftmnd
    # alpha: ftmn
    r = sigma + 2.0 * np.pi * z
    a = freqs_d * (2.0 * np.pi / SOUND_SPEED)

    r = np.reshape(r, (freq_num, frame_num, pair_num))
    a = np.reshape(a, (freq_num, pair_num, feature_dim))
    alpha = np.reshape(alpha, (freq_num, frame_num, pair_num))

    lamb, p, x_non_const_power = coplanar_least_squares_st_norm_one(
        a, r, alpha, zero_feature_index=zero_feature_index, iter_num=iter_num2, eps=eps,
    )

    # lamb: batch
    # lamb=np.reshape(lamb,(freq_num,frame_num))
    p = np.reshape(p, (freq_num, frame_num, feature_dim))
    # pを変更

    if silent_mode == False:
        org_cost_3, cost_3 = calc_auxiliary_function_cost(
            freqs_d, p, sigma, z, x, SOUND_SPEED, eps
        )
    else:
        org_cost_3 = 0
        cost_3 = 0
    return (
        org_cost_0,
        org_cost_1,
        org_cost_2,
        org_cost_3,
        cost_0,
        cost_1,
        cost_2,
        cost_3,
        p,
        z,
        x,
        x_non_const_power,
    )
