# This file converts a bunch of files from yaml to json
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
import time
import warnings
from pathlib import Path

import yaml

ACCEPTED_SUFFIX = [".yml", ".yaml"]
JSON_SUFFIX = ".json"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Load simulation data into a pandas frame"
    )
    parser.add_argument(
        "files",
        type=Path,
        nargs="+",
        metavar="FILE",
        help="The files containing the simulation output results.",
    )
    args = parser.parse_args()

    for filename in args.files:

        print(f"Processing {filename}")

        if filename.suffix not in ACCEPTED_SUFFIX:
            warnings.warn(f"Wrong suffix, skipping file {filename}")
            continue

        output_filename = filename.with_suffix(JSON_SUFFIX)

        if output_filename.exists():
            warnings.warn(f"Output file {output_filename} already exists. Skip.")
            continue

        # Read the file content
        t = time.perf_counter()
        with open(filename, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        t = time.perf_counter() - t
        print(f"* loaded in {t:.3f} seconds")

        # Save the file content to JSON
        t = time.perf_counter()
        with open(output_filename, "w") as f:
            json.dump(content, f)
        t = time.perf_counter() - t
        print(f"* dumped in {t:.3f} seconds")
