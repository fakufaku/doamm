# Generates an overview figure of the simulation results
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
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from results_loader import load_results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Load simulation data into a pandas frame"
    )
    parser.add_argument(
        "-p",
        "--pickle",
        action="store_true",
        help="Read the aggregated data table from a pickle cache",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default="figures",
        help="Output directory for the figures",
    )
    parser.add_argument(
        "dirs",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="The files containing the simulation output results.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, config = load_results(args.dirs, pickle=args.pickle)

    mm_iter = max(config["algo_sweep"]["mm_iter"])

    for n_grid in config["conditions_sweep"]["n_grid"]:
        select = (df["Grid Size"] == n_grid) & (
            (df["Iterations"] == mm_iter) | (df["Iterations"] == "NA")
        )
        df_loc = df[select]

        g = sns.catplot(
            data=df_loc,
            col="Sources",
            row="Grid Size",
            hue="Algorithm",
            x="s",
            y="RMSE [deg]",
            kind="box",
        )

        plt.savefig(args.out / f"figure_1_s_grid{n_grid}.pdf")

    plt.show()
