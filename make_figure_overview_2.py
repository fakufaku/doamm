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
        "files",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="The files containing the simulation output results.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df, config = load_results(args.files, pickle=args.pickle)

    for metric in ["RMSE [deg]", "Runtime [s]"]:
        g = sns.catplot(
            data=df,
            col="Grid Size",
            row="Sources",
            hue="Algorithm",
            x="SNR",
            y=metric,
            kind="box",
            fliersize=1,
        )

        met = metric.split()[0]

        plt.savefig(args.out / f"figure_2_s_metric{met}.pdf")

    plt.show()
