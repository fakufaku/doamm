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

import argparse
import copy
import datetime
import functools
import itertools
import json
import multiprocessing as multi
import os
import pprint
import random as random
import threading
import time as time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra
import yaml
from joblib import Parallel, delayed
from pyroomacoustics.doa import circ_dist
from scipy.signal import fftconvolve

from doamm import MMMUSIC, MMSRP
from external_mdsbf import MDSBF
from external_spire_mm import SPIRE_MM
from get_data import samples_dir
from samples.generate_samples import sampling, wav_read_center
from tools import arrays, geom, metrics


def generate_args(config):

    params = config["params"]
    sweep = config["conditions_sweep"]

    np.random.seed(params["seed"])

    # get the microphone array
    mic_array_loc = params["mic_array_location"]
    R = arrays.get_by_name(name=params["mic_array_name"], center=mic_array_loc)
    R = R[:, :: params["mic_array_downsampling"]]

    # compute the parameters for the room simulation
    e_abs, max_order = pra.inverse_sabine(params["rt60"], params["room_dim"])
    p_reverb = {"e_abs": e_abs, "max_order": max_order}

    # pick the source locations at random upfront
    # doa: shape (n_repeat, n_sources, 2)
    n_sources_max = max(sweep["n_sources"])

    # colatitude [0, pi]
    # azimuth [0, 2 pi]
    doas = np.concatenate(
        (
            np.pi * np.random.rand(params["repeat"], n_sources_max, 1),
            2.0 * np.pi * np.random.rand(params["repeat"], n_sources_max, 1),
        ),
        axis=-1,
    )

    # we also choose all the source samples upfront
    files = sampling(
        params["repeat"], n_sources_max, os.path.join(samples_dir, "metadata.json")
    )

    # generate the variable arguments
    all_args = []
    for n_sources in sweep["n_sources"]:
        for snr in sweep["snr"]:
            for n_grid in sweep["n_grid"]:
                for rep in range(params["repeat"]):
                    seed = np.random.randint(2 ** 32 - 1)
                    all_args.append(
                        {
                            "n_sources": n_sources,
                            "snr": snr,
                            "n_grid": n_grid,
                            "doas": doas[rep],
                            "files": files[rep],
                            "rep": rep,
                            "seed": seed,
                        }
                    )

    return R, p_reverb, all_args


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


def doa_experiment(config, R, p_reverb, n_sources, snr, n_grid, doas, files, rep, seed):

    # reduce number of threads to 1
    import mkl

    mkl.set_num_threads(1)

    # Fix randomness
    np.random.seed(seed)

    fs = config["params"]["fs"]

    # Create a Room
    e_abs, max_order = p_reverb["e_abs"], p_reverb["max_order"]
    room = pra.ShoeBox(
        config["params"]["room_dim"],
        fs=fs,
        max_order=max_order,
        materials=pra.Material(e_abs),
    )

    # We use a circular array with radius 15 cm # and 12 microphones
    room.add_microphone_array(R)

    # read source signals
    signals = wav_read_center(files[:n_sources], center=True, seed=0)

    # _params source locations
    source_locations = geom.spherical_to_cartesian(
        doa=doas,
        distance=config["params"]["source_distance"],
        ref=np.array(config["params"]["mic_array_location"]),
    )

    # add the source
    for k in range(n_sources):
        signals[k] /= np.std(signals[k])
        room.add_source(source_locations[:, k], signal=signals[k])

    # run the simulation
    room.simulate(snr=snr)

    rt60 = pra.experimental.measure_rt60(room.rir[0][0], fs=fs)
    wave_length = np.float(np.shape(room.mic_array.signals)[-1]) / np.float(fs)

    ################################
    # Compute the STFT frames needed
    # shape (n_frames, n_freq, n_channels)
    p_stft = config["params"]["stft"]
    X = pra.transform.analysis(room.mic_array.signals.T, p_stft["nfft"], p_stft["hop"])

    # the DOA localizer takes a different ordering of dimensions
    X = X.transpose([2, 1, 0])  # (n_channels, n_freq, n_frames)

    # The frequency range to use
    freq_bins_bnd = [int(f / fs * p_stft["nfft"]) for f in config["params"]["freq_hz"]]
    freq_bins = np.arange(*freq_bins_bnd)

    pairs = [[m1, m2] for m1 in range(R.shape[1]) for m2 in range(m1 + 1, R.shape[1])]

    doa_algorithms = {
        "SRP": pra.doa.SRP,
        "MUSIC": pra.doa.MUSIC,
        "SPIRE_MM": SPIRE_MM,
        "MMMUSIC": MMMUSIC,
        "MMSRP": MMSRP,
    }

    # Now generate all the algorithms to evaluate
    algorithms = {}
    for name, p in config["algorithms"].items():

        if p["name"] == "SPIRE_MM":
            new_p = copy.deepcopy(p)
            new_p["kwargs"]["mic_positions"] = R.T
            new_p["kwargs"]["mic_pairs"] = pairs
            algorithms[name] = new_p

        elif p["name"] in config["mm_algos"]:

            sweep = config["algo_sweep"]

            # we always start by zero iterations
            p["kwargs"]["n_iter"] = 0

            prod = itertools.product(
                *[sweep[val] for val in ["mm_types", "s"] if val in sweep]
            )
            for t in prod:
                new_p = copy.deepcopy(p)

                new_p["kwargs"]["mm_type"] = t[0]
                new_name = f"{name}_{t[0]}"

                if "s" in sweep:
                    new_p["kwargs"]["s"] = t[1]
                    new_name += f"_s{t[1]:.1f}"

                algorithms[new_name] = new_p

    ##############################################
    # Now we can test all the algorithms available

    results = []
    result_tmp = {
        "name": "",
        "rmse": [],
        "runtime": 0.0,
        "rt60": float(rt60),
        "sample_length": float(wave_length),
        "n_sources": n_sources,
        "snr": snr,
        "n_grid": n_grid,
        "rep": rep,
        "seed": seed,
    }

    def make_new_result(name, rmse, runtime):
        new_res = result_tmp.copy()
        new_res["name"] = name
        new_res["rmse"] = rmse.tolist()
        new_res["runtime"] = runtime
        return new_res

    for name, p in algorithms.items():

        # Construct the new DOA object
        # the max_four parameter is necessary for FRIDA only
        c = pra.constants.get("c")
        doa = doa_algorithms[p["name"]](
            R, fs, p_stft["nfft"], dim=3, c=c, n_grid=n_grid, **p["kwargs"]
        )

        # this call here perform localization on the frames in X
        start = time.perf_counter()
        doa.locate_sources(X, num_src=n_sources, freq_bins=freq_bins)
        elapsed_time = time.perf_counter() - start

        estimate = np.c_[doa.colatitude_recon, doa.azimuth_recon]
        rmse, perm = metrics.doa_eval(doas, estimate)

        if p["name"] not in config["mm_algos"]:
            results.append(make_new_result(name, rmse, elapsed_time))
        else:
            new_name = name + "_it0"
            results.append(make_new_result(new_name, rmse, elapsed_time))

            spent_iter = 0
            for mm_iter in config["algo_sweep"]["mm_iter"]:
                # run MM for a few more iterations
                start = time.perf_counter()
                doa.refine(n_iter=mm_iter - spent_iter)
                elapsed_time = elapsed_time + time.perf_counter() - start

                spent_iter = mm_iter

                estimate = np.c_[doa.colatitude_recon, doa.azimuth_recon]
                rmse, perm = metrics.doa_eval(doas, estimate)

                new_name = name + f"_it{mm_iter}"
                results.append(make_new_result(new_name, rmse, elapsed_time))

    return results


def main_run(args):

    # open the configuration
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    R, p_reverb, all_args = generate_args(config)

    run_doa = functools.partial(doa_experiment, config=config, R=R, p_reverb=p_reverb)

    if args.test:
        results = run_doa(**all_args[0])
    else:
        # Now run this in parallel with joblib
        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(run_doa)(**kwargs) for kwargs in all_args
        )

    date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = args.output / f"{date}_{config['name']}"
    os.makedirs(output_dir, exist_ok=True)

    # save the results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f)

    # save together with the config file used
    with open(output_dir / "config.yml", "w") as f:
        # json.dump(results, f)
        yaml.dump(config, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DOA experiments")
    parser.add_argument("config", type=Path, help="Path to configuration file")
    parser.add_argument(
        "--output", type=Path, default="sim_results", help="Path to configuration file"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode (single loop)"
    )
    args = parser.parse_args()

    main_run(args)
