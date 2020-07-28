"""
DOA Algorithms Experiment
=========================

This script will generate simulate some random rooms and place a number of sources
as well as a microphone array inside.

It will then play some sound samples from the sources and compute the DOAs of
the sources from the microphone array input.

This example demonstrates how to use the DOA object to perform direction of arrival
finding in 2D using one of several algorithms
- MUSIC
- SRP-PHAT
- CSSM
- WAVES
- TOPS
- FRIDA
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

import pyroomacoustics as pra
from doamm import DOAMM, Measurement
from external_mdsbf import MDSBF
from external_spire_mm import SPIRE_MM
from get_data import samples_dir
from mmusic import MMUSIC
from pyroomacoustics.doa import circ_dist
from samples.generate_samples import sampling, wav_read_center
from utils import arrays, geom, metrics

#######################
# add external modules
pra.doa.algorithms["MDSBF"] = MDSBF
pra.doa.algorithms["SPIRE_MM"] = SPIRE_MM
pra.doa.algorithms["DOAMM"] = DOAMM
pra.doa.algorithms["MMUSIC"] = MMUSIC

#######################
# algorithms parameters
stft_nfft = 256  # FFT size
stft_hop = 128  # stft shift
freq_bins = np.arange(6, 60)  # FFT bins to use for estimation
algo_names = ["SRP", "MUSIC", "SPIRE_MM", "DOAMM", "MMUSIC"]
# algo_names = ["DOAMM"]
# algo_names = ["MDSBF"]
# algo_names = ["MUSIC", "MMUSIC"]
#######################

#########################
# Simulation parameters #
SNR = 10.0  # signal-to-noise ratio
c = 343.0  # speed of sound
fs = 16000  # sampling frequency
room_dim = np.r_[10.0, 10.0, 10.0]  # room dimensions

# available: "pyramic", "amazon_echo"
mic_array_name = "pyramic"

# number of sources to simulate
n_sources = 3
# Number of loops in the simulation
n_repeat = 1

# random number seed
seed = 1691531260
seed = None
#########################

# Fix randomness
if seed is not None:
    np.random.seed(seed)
else:
    print("New random seed")
    seed = np.random.randint(2 ** 31)
    np.random.seed(seed)
print(f"The seed is {seed}")

# get the file names
files = sampling(n_repeat, n_sources, os.path.join(samples_dir, "metadata.json"))

# pick the source locations at random upfront
# doa: shape (n_repeat, n_sources, 2)
doas = np.concatenate(
    (
        np.pi * np.random.rand(n_repeat, n_sources, 1),  # colatitude [0, pi]
        2.0 * np.pi * np.random.rand(n_repeat, n_sources, 1),  # azimuth [0, 2 pi]
    ),
    axis=-1,
)

# we let the sources all be at the same distance
distance = 3.0  # meters

# get the locations of the microphones
# we place the microphone a little bit off center to avoid artefacts in the simulation
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1  # a little off center

# get the microphone array
R = arrays.get_by_name(name=mic_array_name, center=mic_array_loc)
R = R[:, ::2]

for rep in range(n_repeat):

    # Create an anechoic room
    e_abs, max_order = pra.inverse_sabine(0.6, room_dim)
    # max_order = 0
    room = pra.ShoeBox(
        room_dim, fs=fs, max_order=max_order, materials=pra.Material(e_abs)
    )

    # We use a circular array with radius 15 cm # and 12 microphones
    room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))

    # read source signals
    signals = wav_read_center(files[rep], center=True, seed=0)

    # source locations
    source_locations = geom.spherical_to_cartesian(
        doa=doas[rep],
        distance=distance * np.ones(doas[rep].shape[0]),
        ref=mic_array_loc,
    )

    # add the source
    for k in range(n_sources):
        signals[k] /= np.std(signals[k])
        room.add_source(source_locations[:, k], signal=signals[k])

    # run the simulation
    room.simulate(snr=SNR)
    mean_rt60 = np.mean(room.measure_rt60())
    print(f"Room RT60 {mean_rt60}")

    ################################
    # Compute the STFT frames needed
    # shape (n_frames, n_freq, n_channels)
    X = pra.transform.analysis(room.mic_array.signals.T, stft_nfft, stft_hop)

    # the DOA localizer takes a different ordering of dimensions
    X = X.transpose([2, 1, 0])

    ##############################################
    # Now we can test all the algorithms available

    n_grid_init = 250
    n_mm_iter = 10

    for algo_name in algo_names:
        # Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[algo_name](
            R,
            fs,
            stft_nfft,
            dim=3,
            c=c,
            # MUSIC/SRP parameters
            n_grid=1000,
            # DOA-MM parameters
            s=1.0,
            beta=1.0,
            n_iter=n_mm_iter,
            init_grid=n_grid_init,
            track_cost=False,
            verbose=False,
            # SPIRE parameters
            n_bisec_search=8,
            n_rough_grid=n_grid_init,
            n_mm_iterations=n_mm_iter,
            mic_positions=R.T,
            mic_pairs=[
                [m1, m2]
                for m1 in range(R.shape[1] - 1)
                for m2 in range(m1 + 1, np.minimum(m1 + 1 + 1, R.shape[1]))
            ],
        )

        # this call here perform localization on the frames in X
        t1 = time.perf_counter()
        doa.locate_sources(X, num_src=n_sources, freq_bins=freq_bins)
        t2 = time.perf_counter()

        # wrap azimuth in positive orthant
        I = doa.azimuth_recon < 0.0
        doa.azimuth_recon[I] = 2.0 * np.pi + doa.azimuth_recon[I]

        # pack
        estimate = np.c_[doa.colatitude_recon, doa.azimuth_recon]

        rmse, perm = metrics.doa_eval(doas[rep], estimate)

        print(f"Algorithm: {algo_name}")
        print(f"Computation time {t2 - t1:.6f}")
        print("Co est:", np.degrees(doa.colatitude_recon[perm]))
        print("Co  gt:", np.degrees(doas[rep][:, 0]))
        print("Az est:", np.degrees(doa.azimuth_recon[perm]))
        print("Az  gt:", np.degrees(doas[rep][:, 1]))

        print(f"{algo_name} Error:", np.degrees(rmse))
        print(f"{algo_name} Total:", np.degrees(np.mean(rmse)))
        print()

        try:
            if len(doa.cost) > 0:
                plt.figure()
                plt.plot(np.array(doa.cost).T)
                plt.show()
        except:
            plt.close()
