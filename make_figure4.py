import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from results_loader import load_results


def main(args, df, config):

    # general setting of matplotlib/seaborn
    palette = sns.color_palette("viridis", n_colors=8)
    sns.set_theme(context="paper", style="white", font_scale=0.7, palette=palette)

    # plot parameters
    grid = 100
    mm_type = "Quadratic"
    iteration = 30

    # latex single column width
    # 3.3865 in == 8.6 cm
    # latex double column width
    # 7 in == 17.78 cm

    cm2in = 0.39
    fig_width = 17.78  # cm (7 inch)
    fig_height = 4.5  # cm
    leg_space = 1.7  # cm

    figsize = ((fig_width - leg_space) * cm2in, fig_height * cm2in)
    ylim = [-0.1, 10]
    xticks = [-10, -5, 0, 5, 10]
    yticks = [0, 2, 4, 6, 8, 10]

    algo_dict = {"MMSRP": "SRP-PHAT", "MMMUSIC": "MUSIC"}

    pivot_tables = []
    for metric in ["RMSE [deg]"]:
        select = (
            (df["Grid Size"] == grid)
            & (df["MM type"] == mm_type)
            & (df["Iterations"] == iteration)
        )
        df_cat = df[select]

        pivot_table = df_cat.pivot_table(
            metric, ["SNR", "Sources", "Name"], ["s"], aggfunc=np.median,
        )
        pivot_tables.append(pivot_table)

        print(f"Table for {metric}")
        print(pivot_table.to_latex(float_format="%.2f"))

    return pivot_tables


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

    pivot_tables = main(args, df, config)
