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
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# This table maps some of the labels we used in the
# simulation to labels we would like to use in the figures
# of the paper
substitutions = {
    "Algorithm": {
        "five_laplace": "FIVE",
        "overiva_ip_laplace": "OverIVA-IP",
        "overiva_ip2_laplace": "OverIVA-IP2",
        "overiva_ip_block_laplace": "OverIVA-IP-NP",
        "overiva_ip2_block_laplace": "OverIVA-IP2-NP",
        "overiva_demix_bg_laplace": "OverIVA-DX/BG",
        "ogive_laplace": "OGIVEs",
        "auxiva_laplace": "AuxIVA-IP (PCA)",
        "auxiva_laplace_nopca": "AuxIVA-IP",
        "auxiva_iss_laplace": "AuxIVA-ISS (PCA)",
        "auxiva_iss_laplace_nopca": "AuxIVA-ISS",
        "auxiva2_laplace": "AuxIVA-IP2 (PCA)",
        "auxiva2_laplace_nopca": "AuxIVA-IP2",
        "auxiva_pca": "PCA+AuxIVA-IP",
        "auxiva_demix_steer_nopca": "AuxIVA-IPA",
        "auxiva_demix_steer_pca": "AuxIVA-IPA (PCA)",
        "overiva_demix_steer": "OverIVA-IPA",
        "pca": "PCA",
    }
}


def load_results(data_files, pickle=False):

    # check if a pickle file exists for these files
    picklefile_name = data_files[0].stem
    pickle_file = f".{picklefile_name}.pickle"

    if os.path.isfile(pickle_file) and pickle:
        print("Reading existing pickle file...")
        # read the pickle file
        df = pd.read_pickle(pickle_file)

    else:

        # reading all data files in the directory
        records = []
        for file in data_files:
            with open(file, "r") as f:
                content = yaml.load(f, Loader=yaml.FullLoader)
                for seg in content:
                    records += seg

        # build the data table line by line
        print("Building table")
        columns = [
            "Sources",
            "SNR",
            "Grid Size",
            "Algorithm",
            "seed",
            "rep",
            "rt60",
            "length",
            "Runtime [s]",
            "RMSE [deg]",
        ]
        table = []

        copy = [
            "n_sources",
            "snr",
            "n_grid",
            "name",
            "seed",
            "rep",
            "rt60",
            "sample_length",
            "runtime",
        ]

        for record in records:
            entry = [record[field] for field in copy]
            for err in record["rmse"]:
                table.append(entry + [np.degrees(err)])

        # create a pandas frame
        print("Making PANDAS frame...")
        df = pd.DataFrame(table, columns=columns)

        df.to_pickle(pickle_file)

    return df


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
