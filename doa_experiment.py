# This runs a single instance of an experiment mainly for test and debug.
#
# Copyright 2020 Robin Scheibler
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
import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist
from scipy.signal import fftconvolve

from doamm import MMMUSIC, MMSRP, DOAMMBase, SurrogateType
from external_mdsbf import MDSBF
from external_spire_mm import SPIRE_MM
from get_data import samples_dir
from samples.generate_samples import sampling, wav_read_center
from tools import arrays, geom, metrics

#######################
# add external modules
pra.doa.algorithms["SPIRE_MM"] = SPIRE_MM
pra.doa.algorithms["MMMUSIC"] = MMMUSIC
pra.doa.algorithms["MMSRP"] = MMSRP

#######################
# algorithms parameters
stft_nfft = 256  # FFT size
stft_hop = 128  # stft shift
freq_bins = np.arange(4, 100)  # FFT bins to use for estimation

# DOA-MM parameters

# algo_names = ["SRP", "MUSIC", "MMMUSIC"]
# algo_names = ["DOAMM"]
# algo_names = ["MDSBF"]
# algo_names = ["MUSIC", "MMMUSIC"]
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
n_sources = 2
# Number of loops in the simulation
n_repeat = 1

# random number seed
seed = 1784850191
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
R = R[:, ::8]

# Grid size for all DOA algorithms
n_grid = 500

# DOA-MM options
s = 1.0
mm_iter = 0
track_cost = False

# SPIRE-MM options
use_kd_tree = True
spire_mm_iter = 10

for rep in range(n_repeat):

    # Create an anechoic room
    e_abs, max_order = pra.inverse_sabine(0.8, room_dim)
    # max_order = 0
    room = pra.ShoeBox(
        room_dim, fs=fs, max_order=max_order, materials=pra.Material(e_abs)
    )

    # We use a circular array with radius 15 cm # and 12 microphones
    room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))

    # read source signals
    signals = wav_read_center(files[rep], center=True, seed=0)
    # signals = signals[:, signals.shape[1] // 4 : signals.shape[1] // 4 + stft_nfft * 10]

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

    algorithms = {
        "SRP": {"name": "SRP", "kwargs": {"n_grid": n_grid}},
        "MUSIC": {"name": "MUSIC", "kwargs": {"n_grid": n_grid}},
        "MMMUSIC-Lin": {
            "name": "MMMUSIC",
            "kwargs": {
                "n_grid": n_grid,
                "s": s,
                "n_iter": mm_iter,
                "track_cost": track_cost,
                "verbose": False,
                "mm_type": SurrogateType.Linear,
            },
        },
        "MMMUSIC-Quad": {
            "name": "MMMUSIC",
            "kwargs": {
                "n_grid": n_grid,
                "s": s,
                "n_iter": mm_iter,
                "track_cost": track_cost,
                "verbose": False,
                "mm_type": SurrogateType.Quadratic,
            },
        },
        "MMSRP-Lin": {
            "name": "MMSRP",
            "kwargs": {
                "n_grid": n_grid,
                "s": s,
                "n_iter": mm_iter,
                "track_cost": track_cost,
                "verbose": False,
                "mm_type": SurrogateType.Linear,
            },
        },
        "MMSRP-Quad": {
            "name": "MMSRP",
            "kwargs": {
                "n_grid": n_grid,
                "s": s,
                "n_iter": mm_iter,
                "track_cost": track_cost,
                "verbose": False,
                "mm_type": SurrogateType.Quadratic,
            },
        },
    }

    algorithms["SPIRE_MM-Quad"] = {
        "name": "SPIRE_MM",
        "kwargs": {
            "n_grid": n_grid,
            "rooting_n_iter": 5,
            "n_rough_grid": 250,
            "n_mm_iterations": spire_mm_iter,
            "mic_positions": R.T,
            "mic_pairs": [
                [m1, m2]
                for m1 in range(R.shape[1] - 1)
                for m2 in range(m1 + 1, np.minimum(m1 + 1 + 1, R.shape[1]))
            ],
            "mm_type": SurrogateType.Quadratic,
            "use_kd_tree": use_kd_tree,
        },
    }

    algorithms["SPIRE_MM-Lin"] = {
        "name": "SPIRE_MM",
        "kwargs": {
            "n_grid": n_grid,
            "rooting_n_iter": 5,
            "n_rough_grid": 250,
            "n_mm_iterations": spire_mm_iter,
            "mic_positions": R.T,
            "mic_pairs": [
                [m1, m2]
                for m1 in range(R.shape[1] - 1)
                for m2 in range(m1 + 1, np.minimum(m1 + 1 + 1, R.shape[1]))
            ],
            "mm_type": SurrogateType.Linear,
            "use_kd_tree": use_kd_tree,
        },
    }

    for variant_name, p in algorithms.items():
        # Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[p["name"]](R, fs, stft_nfft, dim=3, c=c, **p["kwargs"])
        print(X.shape)

        # this call here perform localization on the frames in X
        t1 = time.perf_counter()
        doa.locate_sources(X, num_src=n_sources, freq_bins=freq_bins)
        t2 = time.perf_counter()

        if isinstance(doa, DOAMMBase):
            print(
                "before:",
                np.degrees(doa.colatitude_recon),
                np.degrees(doa.azimuth_recon),
            )
            doa.refine(n_iter=20)
            print(
                "after:",
                np.degrees(doa.colatitude_recon),
                np.degrees(doa.azimuth_recon),
            )

        # wrap azimuth in positive orthant
        I = doa.azimuth_recon < 0.0
        doa.azimuth_recon[I] = 2.0 * np.pi + doa.azimuth_recon[I]

        # pack
        estimate = np.c_[doa.colatitude_recon, doa.azimuth_recon]

        rmse, perm = metrics.doa_eval(doas[rep], estimate)

        print(f"Algorithm: {variant_name}")
        print(f"Computation time {t2 - t1:.6f}")
        print("Co est:", np.degrees(doa.colatitude_recon[perm]))
        print("Co  gt:", np.degrees(doas[rep][:, 0]))
        print("Az est:", np.degrees(doa.azimuth_recon[perm]))
        print("Az  gt:", np.degrees(doas[rep][:, 1]))

        print(f"{variant_name} Error:", np.degrees(rmse))
        print(f"{variant_name} Total:", np.degrees(np.mean(rmse)))
        print()

        try:
            if doa._track_cost:
                plt.figure()
                plt.title(variant_name)
                plt.plot(np.array(doa.cost).T)
                plt.xlabel("MM Iterations")
                plt.show()
        except:
            plt.close()
