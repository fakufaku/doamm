# This file runs the experiment on the pyramic data and also plots the results
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
Evaluate Direction of Arrival Algorithms
========================================

This example evaluate the performance of three direction of arrival (DOA)
algorithms on the recorded samples. It compares the discrepancy between the
output of the DOA algorithm and the calibrated locations (manual and
optimized).

The script requires `numpy`, `scipy`, `pyroomacoustics`, and `joblib` to run.

The three DOA algorithms are `MUSIC` [1], `SRP-PHAT` [2], and `WAVES` [3].

References
----------

.. [1] R. Schmidt, *Multiple emitter location and signal parameter estimation*, 
    IEEE Trans. Antennas Propag., Vol. 34, Num. 3, pp 276--280, 1986

.. [2] J. H. DiBiase, *A high-accuracy, low-latency technique for talker localization 
    in reverberant environments using microphone arrays*, PHD Thesis, Brown University, 2000

.. [3] E. D. di Claudio, R. Parisi, *WAVES: Weighted average of signal subspaces for 
    robust wideband direction finding*, IEEE Trans. Signal Process., Vol. 49, Num. 10, 
    2179--2191, 2001
"""
import json

import numpy as np
import pyroomacoustics as pra
from joblib import Parallel, delayed
from scipy.io import wavfile

from doamm import MMMUSIC, MMSRP
from tools import metrics

random_seed = 1622441898

# We choose the sample to use
sample_names = [f"fq_sample{s}" for s in range(5)]
fn = "pyramic_dataset/segmented/{name}/{name}_spkr{spkr}_angle{angle}.wav"

fs = 16000  # This needs to be changed to 48000 for 'noise', 'sweep_lin', 'sweep_exp'
nfft = 256
stft_hop = 256

# We use a spherical grid with 30000 points
algorithms = {
    "SRP-PHAT": {"algo_obj": "SRP", "n_grid": 10000},
    "MUSIC": {"algo_obj": "MUSIC", "n_grid": 10000},
    "MMMUSIC": {"algo_obj": "MMMUSIC", "n_grid": 100, "n_iter": 30, "s": -1},
    "MMSRP": {"algo_obj": "MMSRP", "n_grid": 100, "n_iter": 30, "s": 1},
}

locate_kwargs = {
    "MUSIC": {"freq_range": [300.0, 6500.0]},
    "SRP-PHAT": {"freq_range": [300.0, 6500.0]},
    "MMMUSIC": {"freq_range": [300.0, 6500.0]},
    "MMSRP": {"freq_range": [300.0, 6500.0]},
}


def read_mix_samples(names, spkrs, angles):

    fs = None
    n_chan = None
    data_lst = []

    for name, spkr, angle in zip(names, spkrs, angles):

        filename = fn.format(name=name, spkr=spkr, angle=angle)
        fs_data, data = wavfile.read(filename)

        if fs is None:
            fs = fs_data
        else:
            assert fs_data == fs

        if n_chan is None:
            n_chan = data.shape[1]
        else:
            assert n_chan == data.shape[1]

        if data.dtype == np.int16:
            data = data / 2 ** 15  # make float and within [-1, 1]
        elif data.dtype not in [np.float32, np.float64]:
            raise ValueError(f"Unsupported data format {data.dtype}")

        data_lst.append(data)

    # find the longest signal
    mlen = max([d.shape[0] for d in data_lst])

    # sum up the centered signals
    data_out = np.zeros((mlen, n_chan), dtype=data.dtype)
    for d in data_lst:
        s = (mlen - d.shape[0]) // 2
        e = s + d.shape[0]
        data_out[s:e, :] += d
    data_out /= len(data_lst)

    return fs, data_out


def run_doa(
    calibration_file, angles, heights, algo, doa_kwargs, freq_range, speakers_numbering
):
    """ Run the doa localization for one source location and one algorithm """

    pra.doa.algorithms["MMSRP"] = MMSRP
    pra.doa.algorithms["MMMUSIC"] = MMMUSIC

    with open(calibration_file, "r") as f:
        locations = json.load(f)

    c = locations["sound_speed_mps"]
    n_src = len(angles)
    assert n_src == len(heights)

    # microphone locations
    mic_array = np.array(locations["microphones"]).T

    # Prepare the DOA localizer object
    algo_key = doa_kwargs["algo_obj"]
    doa = pra.doa.algorithms[algo_key](
        mic_array, fs, nfft, c=c, num_src=n_src, dim=3, **doa_kwargs
    )

    # get the loudspeaker index from its name
    spkrs = [speakers_numbering[h] for h in heights]

    # open the recording file
    fs_data, data = read_mix_samples(sample_names[: len(angles)], spkrs, angles)

    if fs_data != fs:
        raise ValueError("Sampling frequency mismatch")

    # do time-freq decomposition
    X = pra.transform.stft.analysis(data, nfft, stft_hop)
    X = X.transpose([2, 1, 0])

    # run doa
    doa.locate_sources(X, freq_range=freq_range)
    col, az = doa.colatitude_recon, doa.azimuth_recon
    estimate = np.c_[col, az]

    # manual calibration groundtruth
    col_gt_man = np.array([locations["speakers_manual_colatitude"][h] for h in heights])
    az_gt_man = np.radians([int(a) for a in angles])
    doas_gt_man = np.c_[col_gt_man, az_gt_man]
    errors_man, perm = metrics.doa_eval(doas_gt_man, estimate)

    # optimized calibration groundtruth
    col_gt_opt = np.array(
        [locations["sources"][h]["colatitude"][a] for h, a in zip(heights, angles)]
    )
    az_gt_opt = np.array(
        [locations["sources"][h]["azimuth"][a] for h, a in zip(heights, angles)]
    )
    doas_gt_opt = np.c_[col_gt_opt, az_gt_opt]
    errors_opt, perm = metrics.doa_eval(doas_gt_opt, estimate)
    print(f"{algo}:")
    for h, a, e_man, e_opt in zip(heights, angles, errors_man, errors_opt):
        print(f"{h, a}: Err Man={e_man} Opt={e_opt}")

    return {
        "algo": algo,
        "angles": angles,
        "spkr_height": heights,
        "loc_man": (col_gt_man.tolist(), az_gt_man.tolist()),
        "loc_opt": (col_gt_opt.tolist(), az_gt_opt.tolist()),
        "loc_doa": (col.tolist(), az.tolist()),
        "error_man": errors_man.tolist(),
        "error_opt": errors_opt.tolist(),
    }


def main_run(args):

    np.random.seed(random_seed)

    with open(args.calibration_file, "r") as f:
        locations = json.load(f)

    # Recover the list of all sources locations
    spkr_azimuths = list(locations["sources"]["low"]["azimuth"].keys())
    spkr_height = list(locations["sources"].keys())

    all_args = []
    for algo, doa_kwargs in algorithms.items():

        # This will loop over all sources locations
        for h in spkr_height:
            for angle in spkr_azimuths:

                angles = [angle]
                heights = [h]

                if args.sources > 1:
                    angles += np.random.choice(spkr_azimuths, args.sources - 1).tolist()
                    heights += np.random.choice(spkr_height, args.sources - 1).tolist()

                all_args.append(
                    (
                        args.calibration_file,
                        angles,
                        heights,
                        algo,
                        doa_kwargs,
                        locate_kwargs[algo]["freq_range"],
                        locations["speakers_numbering"],
                    )
                )

    # Now run this in parallel with joblib
    results = Parallel(n_jobs=args.workers)(
        delayed(run_doa)(*args) for args in all_args
    )

    with open(args.output, "w") as f:
        json.dump(results, f)


def main_plot(args):
    """ Plot the result of the Evaluation """

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    table = []

    for result_file in args.result_files:

        with open(result_file, "r") as f:
            results = json.load(f)

        for res in results:
            if "angle" in res:
                # old simulation file format
                res["Sources"] = 1
                table.append(
                    {
                        "algo": res["algo"],
                        "Sources": 1,
                        "angle": res["angle"],
                        "spkr_height": res["spkr_height"],
                        "error_man": res["error_man"],
                        "error_opt": res["error_opt"],
                    }
                )

            else:
                for a, h, e_man, e_opt in zip(
                    res["angles"],
                    res["spkr_height"],
                    res["error_man"],
                    res["error_opt"],
                ):
                    table.append(
                        {
                            "algo": res["algo"],
                            "Sources": len(res["angles"]),
                            "angle": a,
                            "spkr_height": h,
                            "error_man": e_man,
                            "error_opt": e_opt,
                        }
                    )

    df = pd.DataFrame(table)

    # Compute the average error with the grid used
    grid = pra.doa.GridSphere(n_points=30000)
    avg_error = np.degrees(grid.min_max_distance()[2])

    # remove the location columns, only keep error
    # df = df[["algo", "angle", "spkr_height", "error_man", "error_opt"]]
    df2 = pd.melt(
        df,
        value_vars=["error_man", "error_opt"],
        value_name="Error [deg.]",
        var_name="Calibration",
        id_vars=["algo", "angle", "spkr_height", "Sources"],
    )

    df2["Error [deg.]"] = df2["Error [deg.]"].apply(np.degrees)
    df2["Calibration"] = df2["Calibration"].replace(
        {"error_man": "Manual", "error_opt": "Optimized"}
    )

    df2["algo_class"] = df2["algo"]
    df2["algo_class"].replace(to_replace={"MMSRP": "SRP-PHAT"}, inplace=True)
    df2["algo_class"].replace(to_replace={"MMMUSIC": "MUSIC"}, inplace=True)
    df2["algo"].replace(
        to_replace={"SRP-PHAT": "Gr. 10000", "MUSIC": "Gr. 10000"}, inplace=True
    )
    df2["algo"].replace(
        to_replace={"MMSRP": "Gr. 100/30 it.", "MMMUSIC": "Gr. 100/30 it."},
        inplace=True,
    )

    df2 = df2.rename(index=str, columns={"algo": "Algorithms"})

    palette = sns.color_palette("viridis", n_colors=4)
    sns.set_theme(context="paper", style="ticks", font_scale=0.5, palette=palette)
    sns.set_context("paper")

    g = sns.catplot(
        kind="box",
        row="algo_class",
        col="Sources",
        y="Algorithms",
        x="Error [deg.]",
        data=df2[df2["Calibration"] == "Optimized"],
        palette="viridis",
        row_order=["SRP-PHAT", "MUSIC"],
        col_order=[1, 2],
        order=["Gr. 10000", "Gr. 100/30 it."],
        margin_titles=True,
        fliersize=0.5,
    )

    g.set_titles(col_template="", row_template="{row_name}")
    g.axes[0, 0].set_title("1 Source")
    g.axes[0, 1].set_title("2 Sources")

    g.fig.set_size_inches(3.38846, 1.75)
    for r in range(2):
        g.axes[r, 0].set_ylabel("")
        g.axes[r, 0].set_xlim([0, 4])
        g.axes[r, 1].set_xlim([0, 4])

    # plt.ylabel("")
    # plt.xlim([0, 6])
    # sns.despine(offset=5, left=True, bottom=True)
    g.fig.tight_layout(pad=0.1, h_pad=0.5)

    if args.save is not None:
        plt.savefig(args.save, dpi=300)

    plt.show()

    return df2


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a few DOA algorithms")
    subparsers = parser.add_subparsers()

    parser_run = subparsers.add_parser(
        "run", description="Run the DOA algorithms on the measurements"
    )
    parser_run.set_defaults(func=main_run)
    parser_run.add_argument(
        "calibration_file",
        type=str,
        help="The JSON file containing the calibrated locations",
    )
    parser_run.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of cores to use for the processing",
    )
    parser_run.add_argument(
        "--output",
        "-o",
        type=str,
        default="pyramic_doa_results.json",
        help="The JSON file where to save the results",
    )
    parser_run.add_argument(
        "--sources", "-s", type=int, default=1, help="Number of sources"
    )

    parser_plot = subparsers.add_parser(
        "plot", description="Plot the results of the evaluation"
    )
    parser_plot.set_defaults(func=main_plot)
    parser_plot.add_argument(
        "result_files",
        type=str,
        nargs="+",
        metavar="RESULTS",
        help="The JSON file containing the results",
    )
    parser_plot.add_argument("-s", "--save", metavar="FILE", type=str, help="Save plot")

    args = parser.parse_args()
    ret = args.func(args)
