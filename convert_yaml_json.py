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
