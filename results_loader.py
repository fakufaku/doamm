# This file loads the data from the simulation result file and creates a
# pandas data frame for further processing.
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
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_results(dirs, pickle=False):

    # check if a pickle file exists for these files
    picklefile_name = dirs[0].stem
    pickle_file = f".{picklefile_name}.pickle"

    with open(dirs[0] / "config.yml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if os.path.isfile(pickle_file) and pickle:
        print("Reading existing pickle file...")
        # read the pickle file
        df = pd.read_pickle(pickle_file)

    else:

        # reading all data files in the directory
        records = []
        for i, dir in enumerate(dirs):
            with open(dir / "results.json", "r") as f:
                content = json.load(f)
                for seg in content:
                    records += seg

        # build the data table line by line
        print("Building table")
        columns = [
            "Algorithm",
            "Name",
            "Sources",
            "SNR",
            "Grid Size",
            "seed",
            "rep",
            "rt60",
            "length",
            "Runtime [s]",
            "MM type",
            "s",
            "Iterations",
            "RMSE [deg]",
        ]
        table = []

        copy = [
            "n_sources",
            "snr",
            "n_grid",
            "seed",
            "rep",
            "rt60",
            "sample_length",
            "runtime",
        ]

        for record in records:

            algo_prop = record["name"].split("_")
            if len(algo_prop) < 2:
                raise ValueError("Something seems wrong with the data")

            if algo_prop[0] == "SPIRE":
                mm_type = algo_prop[2]
                short_name = "SPIREMM"
            else:
                mm_type = algo_prop[1]
                short_name = algo_prop[0]

            if algo_prop[0] in config["mm_algos"]:
                if len(algo_prop) == 4 and algo_prop[2].startswith("s"):
                    s = float(algo_prop[2][1:])
                    it = int(algo_prop[3][2:])
                else:
                    s = "NA"
                    it = int(algo_prop[2][2:])
            else:
                s = "NA"
                it = "NA"

            entry = (
                [record["name"], short_name]
                + [record[field] for field in copy]
                + [mm_type, s, it]
            )
            for err in record["rmse"]:
                table.append(entry + [np.degrees(err)])

        # create a pandas frame
        print("Making PANDAS frame...")
        df = pd.DataFrame(table, columns=columns)

        df = df.replace({"Lin": "Linear", "Quad": "Quadratic"})

        df.to_pickle(pickle_file)

    return df, config


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
        "dirs",
        type=Path,
        nargs="+",
        metavar="DIR",
        help="The directory containing the simulation output files.",
    )
    args = parser.parse_args()

    dirs = args.dirs
    pickle = args.pickle

    df, rt60, parameters = load_data(args.dirs, pickle=pickle)
