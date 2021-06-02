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
