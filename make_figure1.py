import argparse
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
        "files",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="The files containing the simulation output results.",
    )
    args = parser.parse_args()

    df = load_results(args.files, pickle=args.pickle)

    g = sns.catplot(
        data=df,
        col="Sources",
        row="Grid Size",
        hue="Algorithm",
        x="SNR",
        y="RMSE [deg]",
        kind="box",
    )

    plt.show()
