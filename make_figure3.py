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
    grid_low = 100
    grid_high = 10000

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

    figs = []
    for metric in ["RMSE [deg]"]:

        fig, axes = plt.subplots(2, 4, figsize=figsize)
        figs.append((fig, axes))

        leg_handles = {}

        for alg_ind, algo in enumerate(["MMSRP", "MMMUSIC"]):

            select_lo = (df["Name"] == algo) & (df["Grid Size"] == grid_low)
            select_hi = (
                (df["Name"] == algo)
                & (df["Grid Size"] == grid_high)
                & ((df["Iterations"] == 0) | (df["Iterations"] == 30))
            )

            df_cat = pd.concat((df[select_lo], df[select_hi]))

            pivot_table = df_cat.pivot_table(
                metric,
                ["Name", "MM type", "Sources", "Iterations", "Grid Size"],
                ["SNR"],
                aggfunc=np.median,
            )

            for c, mm_type in enumerate(["Quadratic", "Linear"]):
                for r, n_src in enumerate([1, 2]):

                    col = 2 * alg_ind + c
                    ax = axes[r, col]

                    # grid == 10000 + 30 it.
                    data_row = pivot_table.loc[(algo, mm_type, n_src, 30, 10000)]
                    ax.plot(
                        data_row.index,
                        data_row,
                        color="k",
                        linestyle="--",
                        linewidth=2,
                    )
                    ax.annotate(
                        "Grid 10000 + 30 MM iter.",
                        (data_row.index[-1], data_row[data_row.index[-1]] - 0.3),
                        fontsize="xx-small",
                        ha="right",
                        va="top",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.5,
                            "boxstyle": "square,pad=0.01",
                            "ec": "none",
                        },
                    )

                    # plot the lines for each of the iterations
                    for it in [1, 2, 3, 4, 5, 10, 20, 30]:
                        data_row = pivot_table.loc[(algo, mm_type, n_src, it, 100)]
                        ax.plot(data_row.index, data_row, label=f"{it:2}")

                    # grid == 100
                    data_row = pivot_table.loc[(algo, mm_type, n_src, 0, 100)]
                    ax.plot(data_row.index, data_row, color="k", linewidth=0.75)
                    ax.annotate(
                        "Grid 100",
                        (data_row.index[-1], data_row[data_row.index[-1]] + 0.3),
                        fontsize="xx-small",
                        ha="right",
                    )

                    # grid == 10000
                    data_row = pivot_table.loc[(algo, mm_type, n_src, 0, 10000)]
                    ax.plot(data_row.index, data_row, color="k", linewidth=0.75)
                    t = ax.annotate(
                        "Grid 10000",
                        (data_row.index[-1], data_row[data_row.index[-1]] + 0.3),
                        fontsize="xx-small",
                        ha="right",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.5,
                            "boxstyle": "square,pad=0.01",
                            "ec": "none",
                        },
                    )
                    # t.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor=None))

                    if r == 0:
                        ax.set_xticks([])
                        ax.set_title(f"{algo_dict[algo]}/{mm_type}")
                    else:
                        ax.set_xticks(xticks)

                    if r == 1:
                        ax.set_xlabel("SNR [dB]")

                    if col == 3:
                        if n_src == 1:
                            ylbl = "1 Source"
                        else:
                            ylbl = f"{n_src} Sources"
                        right_ax = ax.twinx()
                        right_ax.set_yticks([])
                        right_ax.set_ylabel(ylbl, rotation=90)

                    if col == 0:
                        ax.set_ylabel("Error [deg.]")
                        ax.set_yticks(yticks)
                    else:
                        ax.set_yticks(yticks)
                        ax.set_yticklabels([])
                        ax.tick_params(axis="y", length=0, width=0)

                    # collect the legend handles
                    handles, labels = ax.get_legend_handles_labels()
                    for lbl, hand in zip(labels, handles):
                        leg_handles[lbl] = hand

                    ax.set_ylim(ylim)
                    ax.yaxis.grid(True)
                    sns.despine(left=True, bottom=True)

        fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.5)
        figleg = fig.legend(
            leg_handles.values(),
            leg_handles.keys(),
            title="MM Iter.",
            title_fontsize="x-small",
            fontsize="xx-small",
            bbox_to_anchor=[1 - leg_space / fig_width / 2.3, 0.5],
            loc="center",
        )

        fig.subplots_adjust(right=1 - leg_space / (fig_width - leg_space))

        met = metric.split()[0]
        fig.savefig(args.out / f"figure_3_grid_{met}.pdf")

    plt.show()

    return figs


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
