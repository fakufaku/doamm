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
