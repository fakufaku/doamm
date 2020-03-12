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
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import pyroomacoustics as pra
from pyroomacoustics.doa import circ_dist

from get_data import samples_dir
from utils import geom, arrays, metrics

from samples.generate_samples import wav_read_center, sampling

#######################
# algorithms parameters
stft_nfft = 256  # FFT size
stft_hop = 128  # stft shift
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation
algo_names = ["SRP", "MUSIC"]
#######################

#########################
# Simulation parameters #
SNR = 0.0  # signal-to-noise ratio
c = 343.0  # speed of sound
fs = 16000  # sampling frequency
room_dim = np.r_[10.0, 10.0, 10.0]  # room dimensions
SNR = 10

# available: "pyramic", "amazon_echo"
mic_array_name = "pyramic"

# number of sources to simulate
n_sources = 2
# Number of loops in the simulation
n_repeat = 1

# random number seed
seed = 0
#########################

# Fix randomness
np.random.seed(seed)

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

for rep in range(n_repeat):

    # Create an anechoic room
    room = pra.ShoeBox(room_dim, fs=fs, max_order=0)

    # We use a circular array with radius 15 cm # and 12 microphones
    room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))

    # read source signals
    signals = wav_read_center(files[rep], center=True, seed=0)

    # source locations
    source_locations = geom.spherical_to_cartesian(
        doa=doas[rep], distance=distance, ref=mic_array_loc
    )

    # add the source
    for k in range(n_sources):
        signals[k] /= np.std(signals[k])
        room.add_source(source_locations[:, k], signal=signals[k])

    # run the simulation
    room.simulate(snr=SNR)

    ################################
    # Compute the STFT frames needed
    # shape (n_frames, n_freq, n_channels)
    X = pra.transform.analysis(room.mic_array.signals.T, stft_nfft, stft_hop)

    # the DOA localizer takes a different ordering of dimensions
    X = X.transpose([2, 1, 0])

    ##############################################
    # Now we can test all the algorithms available

    for algo_name in algo_names:
        # Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only
        doa = pra.doa.algorithms[algo_name](R, fs, stft_nfft, dim=3, c=c, n_points=250)

        # this call here perform localization on the frames in X
        doa.locate_sources(X, num_src=n_sources, freq_bins=freq_bins)

        estimate = np.c_[doa.colatitude_recon, doa.azimuth_recon]

        rmse, perm = metrics.doa_eval(doas[rep], estimate)

        print(f"Algorithm: {algo_name}")
        print("Co est:", np.degrees(doa.colatitude_recon[perm]))
        print("Co  gt:", np.degrees(doas[rep][:, 0]))
        print("Az est:", np.degrees(doa.azimuth_recon[perm]))
        print("Az  gt:", np.degrees(doas[rep][:, 1]))

        print(f"{algo_name} Error:", np.degrees(rmse))
        print(f"{algo_name} Total:", np.degrees(np.mean(rmse)))
        print()
