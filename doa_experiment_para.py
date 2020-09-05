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

import multiprocessing as multi
import os
import pprint
import random as random
import threading
import time as time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

import pyroomacoustics as pra
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
pra.doa.algorithms["MMUSIC"] = SPIRE_MM


#######################
# algorithms parameters
stft_nfft = 256  # FFT size
stft_hop = 128  # stft shift
freq_bins = np.arange(5, 60)  # FFT bins to use for estimation

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
n_sources_list = [1, 2]

# RT60 [(max_order,absorption)]
# reverb_list=[{"name":"0_35","max_order":17,"absorption":0.35},{"name":"0_70","max_order":34,"absorption":0.2}]
reverb_list = [{"name": "0_35", "max_order": 17, "absorption": 0.35}]

eval_list = [
    (
        n_sources,
        reverb,
        "./doa_result_{}_{}.txt".format(n_sources, reverb["name"]),
        "./doa_raw_{}_{}.txt".format(n_sources, reverb["name"]),
    )
    for n_sources in n_sources_list
    for reverb in reverb_list
]
# 評価する手法群


# Number of loops in the simulation
n_repeat = 100
n_repeat_actual = 10

# random number seed
seed = 342

# Fix randomness
np.random.seed(seed)
random.seed(seed)


# we let the sources all be at the same distance
distance = 3.0  # meters

# get the locations of the microphones
# we place the microphone a little bit off center to avoid artefacts in the simulation
mic_array_loc = room_dim / 2 + np.random.randn(3) * 0.1  # a little off center

# get the microphone array
R = arrays.get_by_name(name=mic_array_name, center=mic_array_loc)

# mic pairs
n_channels = np.shape(R)[1]
mic_pairs_dict = {}
mic_pairs_dict["pairs_1"] = [
    [m1, m2]
    for m1 in range(n_channels - 1)
    for m2 in range(m1 + 1, np.minimum(m1 + 1 + 1, n_channels))
]
mic_pairs_dict["pairs_2"] = [
    [m1, m2]
    for m1 in range(n_channels - 1)
    for m2 in range(m1 + 1, np.minimum(m1 + 2 + 1, n_channels))
]
mic_pairs_dict["pairs_4"] = [
    [m1, m2]
    for m1 in range(n_channels - 1)
    for m2 in range(m1 + 1, np.minimum(m1 + 4 + 1, n_channels))
]

# 手法リスト
# grid_list=[180*90,90*90,90*45,60*30,30*15,18*9,12*6]
grid_list = [100, 500, 1000]
algo_meta_info = {}

for grid in grid_list:
    algo_meta_info["MDSBF_{}".format(grid)] = {
        "algo_name": "MDSBF",
        "mic_pairs": mic_pairs_dict["pairs_1"],
        "n_mm_iterations": 5,
        "n_bisec_search": 8,
        "n_rough_grid": 12 * 6,
        "n_precise_grid": grid,
        "s": -1,
    }
    algo_meta_info["SRP_{}".format(grid)] = {
        "algo_name": "SRP",
        "mic_pairs": mic_pairs_dict["pairs_1"],
        "n_mm_iterations": 5,
        "n_bisec_search": 8,
        "n_rough_grid": 12 * 6,
        "n_precise_grid": grid,
        "s": -1,
    }
    algo_meta_info["MUSIC_{}".format(grid)] = {
        "algo_name": "MUSIC",
        "mic_pairs": mic_pairs_dict["pairs_1"],
        "n_mm_iterations": 5,
        "n_bisec_search": 8,
        "n_rough_grid": 12 * 6,
        "n_precise_grid": grid,
        "s": -1,
    }
    algo_meta_info["MMUSIC_{}".format(grid)] = {
        "algo_name": "MMUSIC",
        "mic_pairs": mic_pairs_dict["pairs_1"],
        "n_mm_iterations": 10,
        "n_bisec_search": 8,
        "n_rough_grid": 12 * 6,
        "n_precise_grid": grid,
        "s": -1,
    }
# 提案法のリスト


# rough_grid_list=[18*9,12*6]
rough_grid_list = [12 * 6]

# rmse: n_sample, n_source
def eval_func(
    rmse,
    use_sample_mean=False,
    percentiles=[5, 10, 25, 50, 75, 90, 95],
    admit_errors=[0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0],
):
    n_sample = np.shape(rmse)[0]
    n_source = np.shape(rmse)[1]
    print(n_sample, n_source)

    if use_sample_mean == True:
        # 音源数の方向に平均を取る
        rmse = np.mean(rmse, axis=-1)

    print(np.degrees(rmse))
    # 平均を計算
    eval_mean = np.degrees(np.mean(rmse))
    # メディアンを計算する
    eval_median = np.degrees(np.median(rmse))
    # 標準偏差
    eval_std = np.degrees(np.std(rmse))
    # percentileを計算する
    eval_percentile = np.percentile(np.degrees(rmse), percentiles)
    # 許容誤差以内のデータ数を計算する
    whole_num = np.size(rmse)
    eval_accuracy = []
    for admit_error in admit_errors:
        num = np.sum(np.degrees(rmse) < admit_error)
        eval_accuracy.append(np.float(num) / np.float(whole_num))
    eval_accuracy = np.array(eval_accuracy)

    out_dict = {
        "mean": eval_mean,
        "median": eval_median,
        "std": eval_std,
        "eval_percentile": eval_percentile,
        "eval_accuracy": eval_accuracy,
    }

    return out_dict

    # (n_sources,reverb,"./doa_result_{}_{}.txt".format(n_sources,reverb["name"]),"./doa_raw_{}_{}.txt".format(n_sources,reverb["name"]))


def doa_experiment(algo_meta_info, eval_cond, R, seed=0):
    n_sources = eval_cond[0]
    max_order = eval_cond[1]["max_order"]
    absorption = eval_cond[1]["absorption"]
    path_w = eval_cond[2]
    path_raw = eval_cond[3]

    # Fix randomness
    np.random.seed(seed)
    import random as random

    random.seed(seed)

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

    error_dict = {}
    elapsed_time_dict = {}
    wave_time_list = []

    for meta_info_name in algo_meta_info:
        error_dict[meta_info_name] = []
        elapsed_time_dict[meta_info_name] = []

    for rep in range(n_repeat_actual):

        # Create an anechoic room
        room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, absorption=absorption)

        # We use a circular array with radius 15 cm # and 12 microphones
        room.add_microphone_array(pra.MicrophoneArray(R, fs=room.fs))

        # read source signals
        signals = wav_read_center(files[rep], center=True, seed=0)

        # source locations
        source_locations = geom.spherical_to_cartesian(
            doa=doas[rep], distance=distance, ref=mic_array_loc
        )

        # print(source_locations[:,0]-mic_array_loc)
        # print(source_locations[:,1]-mic_array_loc)
        # print(R[:,0])
        # for s in range(2):
        #    for m in range(40,42):
        #        dist=np.sqrt(np.sum(np.square(R[:,m]-source_locations[:,s])))
        #        print(s,m,dist)

        # add the source
        for k in range(n_sources):
            signals[k] /= np.std(signals[k])
            room.add_source(source_locations[:, k], signal=signals[k])

        # run the simulation
        room.simulate(snr=SNR)

        rt60 = pra.experimental.measure_rt60(room.rir[0][0], fs=fs)
        print("rt60:{} [sec]".format(rt60))

        wave_length = np.float(np.shape(room.mic_array.signals)[-1]) / np.float(fs)
        print("wave: {} [sec]".format(wave_length))
        wave_time_list.append(wave_length)

        ################################
        # Compute the STFT frames needed
        # shape (n_frames, n_freq, n_channels)
        X = pra.transform.analysis(room.mic_array.signals.T, stft_nfft, stft_hop)
        # print(np.shape(room.mic_array.signals))
        # print(np.shape(X))
        # the DOA localizer takes a different ordering of dimensions
        X = X.transpose([2, 1, 0])

        ##############################################
        # Now we can test all the algorithms available

        for meta_info_name in algo_meta_info:
            meta_info = algo_meta_info[meta_info_name]
            algo_name = meta_info["algo_name"]
            print(meta_info)

            start = time.time()

            # Construct the new DOA object
            # the max_four parameter is necessary for FRIDA only
            doa = pra.doa.algorithms[algo_name](
                R,
                fs,
                stft_nfft,
                dim=3,
                c=c,
                s=meta_info["s"],
                n_grid=meta_info["n_precise_grid"],
                mic_positions=R.T,
                n_mm_iterations=meta_info["n_mm_iterations"],
                n_bisec_search=meta_info["n_bisec_search"],
                n_rough_grid=meta_info["n_rough_grid"],
                mic_pairs=meta_info["mic_pairs"],
            )

            # this call here perform localization on the frames in X
            doa.locate_sources(X, num_src=n_sources, freq_bins=freq_bins)

            estimate = np.c_[doa.colatitude_recon, doa.azimuth_recon]

            rmse, perm = metrics.doa_eval(doas[rep], estimate)

            elapsed_time = time.time() - start
            print(f"Algorithm: {meta_info_name}")
            print("Co est:", np.degrees(doa.colatitude_recon[perm]))
            print("Co  gt:", np.degrees(doas[rep][:, 0]))
            print("Az est:", np.degrees(doa.azimuth_recon[perm]))
            print("Az  gt:", np.degrees(doas[rep][:, 1]))

            rmse_num = np.shape(rmse)[-1]
            if rmse_num != n_sources:

                # 平均値を足しとく
                mean = np.mean(rmse)
                for s in range(n_sources - rmse_num):
                    rmse = np.concatenate((rmse, mean[None]), axis=-1)

            print(f"{meta_info_name} Error:", np.degrees(rmse))
            print(f"{meta_info_name} Total:", np.degrees(np.mean(rmse)))
            print(f"{meta_info_name} elapsed_time [sec]:", elapsed_time)

            error_dict[meta_info_name].append(rmse)
            elapsed_time_dict[meta_info_name].append(elapsed_time)
            print()

        for meta_info_name in algo_meta_info:
            error_rmse = error_dict[meta_info_name]
            elapsed_time = elapsed_time_dict[meta_info_name]
            wave_time = np.average(np.array(wave_time_list))
            elapsed_time = np.average(elapsed_time)

            rmse = np.array(error_rmse)

            print(meta_info_name)
            print(error_rmse)

            print(np.shape(rmse))
            out_dict = eval_func(rmse, False)
            # print("{},{},{},{}".format(meta_info_name,rep,out_dict["mean"],out_dict["median"]))
            print(f"{meta_info_name} {rep} {wave_time} {elapsed_time} [sec]")
            pprint.pprint(out_dict)
            # out_dict={"mean":eval_mean,"median":eval_median,"eval_percentile":eval_percentile,"eval_accuracy":eval_accuracy}

            # print(f"{meta_info_name} {rep} Partial Result:", np.degrees(np.mean(rmse)), np.degrees(np.median(rmse)), np.degrees(np.std(rmse)))

        f = open(path_raw + "{}.txt".format(rep), mode="w")
        for meta_info_name in algo_meta_info:
            error_rmse = error_dict[meta_info_name]
            elapsed_time = elapsed_time_dict[meta_info_name]
            wave_time = np.average(np.array(wave_time_list))
            elapsed_time = np.average(elapsed_time)
            rmse = np.array(error_rmse)
            f.write(
                "{} wave {} [sec] elapsed {} [sec]\n".format(
                    meta_info_name, wave_time, elapsed_time
                )
            )
            # f.write(algo_meta_info[meta_info_name])
            f.write("\n")
            f.write("{}\n".format(np.degrees(rmse)))

        f.close()

        f = open(path_w + "{}.txt".format(rep), mode="w")
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        admit_errors = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]

        f.write("algo_name,mean,median,std,")
        for percent in percentiles:
            f.write("percentile_{} [degree],".format(percent))
        for admit_error in admit_errors:
            f.write("accuracy_{} [0-1],".format(admit_error))
        f.write("\n")

        for meta_info_name in algo_meta_info:
            error_rmse = error_dict[meta_info_name]
            rmse = np.array(error_rmse)
            print(f"{meta_info_name} Whole Result:", np.degrees(np.median(rmse)))
            # out_dict={"mean":eval_mean,"median":eval_median,"eval_percentile":eval_percentile,"eval_accuracy":eval_accuracy}

            out_dict = eval_func(
                rmse, False, percentiles=percentiles, admit_errors=admit_errors
            )
            f.write(
                "{},{},{},{},".format(
                    meta_info_name,
                    out_dict["mean"],
                    out_dict["median"],
                    out_dict["std"],
                )
            )
            for i in range(np.size(percentiles)):
                f.write("{},".format(out_dict["eval_percentile"][i]))
            for i in range(np.size(admit_errors)):
                f.write("{},".format(out_dict["eval_accuracy"][i]))
            f.write("\n")
            # f.write("{},Whole Result,{},{},{}:\n".format(meta_info_name,np.degrees(np.mean(rmse)), np.degrees(np.median(rmse)),np.degrees(np.std(rmse))))

        f.close()

    f = open(path_raw, mode="w")
    for meta_info_name in algo_meta_info:
        error_rmse = error_dict[meta_info_name]
        elapsed_time = elapsed_time_dict[meta_info_name]
        wave_time = np.average(np.array(wave_time_list))
        elapsed_time = np.average(elapsed_time)
        rmse = np.array(error_rmse)
        f.write(
            "{} wave {} [sec] elapsed {} [sec]\n".format(
                meta_info_name, wave_time, elapsed_time
            )
        )
        # f.write(algo_meta_info[meta_info_name])
        f.write("\n")
        f.write("{}\n".format(np.degrees(rmse)))

    f.close()

    f = open(path_w, mode="w")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    admit_errors = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]

    f.write("algo_name,mean,median,std,")
    for percent in percentiles:
        f.write("percentile_{} [degree],".format(percent))
    for admit_error in admit_errors:
        f.write("accuracy_{} [0-1],".format(admit_error))
    f.write("\n")

    for meta_info_name in algo_meta_info:
        error_rmse = error_dict[meta_info_name]
        rmse = np.array(error_rmse)
        print(f"{meta_info_name} Whole Result:", np.degrees(np.median(rmse)))
        # out_dict={"mean":eval_mean,"median":eval_median,"eval_percentile":eval_percentile,"eval_accuracy":eval_accuracy}

        out_dict = eval_func(
            rmse, False, percentiles=percentiles, admit_errors=admit_errors
        )
        f.write(
            "{},{},{},{},".format(
                meta_info_name, out_dict["mean"], out_dict["median"], out_dict["std"]
            )
        )
        for i in range(np.size(percentiles)):
            f.write("{},".format(out_dict["eval_percentile"][i]))
        for i in range(np.size(admit_errors)):
            f.write("{},".format(out_dict["eval_accuracy"][i]))
        f.write("\n")
        # f.write("{},Whole Result,{},{},{}:\n".format(meta_info_name,np.degrees(np.mean(rmse)), np.degrees(np.median(rmse)),np.degrees(np.std(rmse))))

    f.close()


# 実行シングル実行

for eval_cond in eval_list:
    doa_experiment(algo_meta_info, eval_cond, R, seed=342)


def process3(arg):
    temp_eval_list = arg[0]
    for eval_cond in temp_eval_list:
        doa_experiment(algo_meta_info, eval_cond, R, seed=342)


print(multi.cpu_count())

all_len = len(eval_list)
Nprocess = 1
Nlen = all_len // Nprocess
args = []
jobs = []
for index in range(0, all_len, Nlen):
    nstart = index
    nend = index + Nlen
    temp_eval_list = eval_list[nstart:nend]
    p = multi.Process(target=process3, args=([temp_eval_list],))
    jobs.append(p)
    p.start()
    print("index{}:".format(index))


for p in jobs:
    p.join()
