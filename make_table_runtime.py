# Generates Table 2 containing the runtimes
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
import pandas as pd
import seaborn as sns

from results_loader import load_results


def main(args, df, config):

    # plot parameters
    grid_low = 100
    grid_high = 10000

    for metric in ["Runtime [s]"]:

        for alg_ind, algo in enumerate(["MMSRP", "MMMUSIC"]):

            select_lo = (
                (df["Name"] == algo)
                & (df["Grid Size"] == grid_low)
                & (df["Iterations"] == 30)
            )
            select_hi = (
                (df["Name"] == algo)
                & (df["Grid Size"] == grid_high)
                & (df["Iterations"] == 0)
            )

            df_cat = pd.concat((df[select_lo], df[select_hi]))

            pivot_table = df_cat.pivot_table(
                metric,
                ["Name", "MM type", "Sources"],
                ["Grid Size"],
                aggfunc=np.median,
            )

            print(algo)
            print(pivot_table.to_markdown())
            # this is how we select a row in the pivot table
            # data_row = pivot_table.loc[(algo, mm_type, n_src, 30, 10000)]

    return None


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

    # load the data
    df, config = load_results(args.files, pickle=args.pickle)

    main(args, df, config)
