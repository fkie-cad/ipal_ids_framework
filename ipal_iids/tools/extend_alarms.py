#!/usr/bin/env python3
import gzip
import json
import sys


def main():
    if len(sys.argv) != 2:
        print("Usage: ./extend-alarms.py [path to file]")
        sys.exit(1)

    def open_file(filename, mode="r"):
        if filename is None:
            return None
        elif filename.endswith(".gz"):
            return gzip.open(filename, mode)
        elif filename == "-":
            return sys.stdin
        else:
            return open(filename, mode)

    ipal = []

    # Load file into memory
    print("Loading data into memory")
    with open_file(sys.argv[1], mode="r") as f:
        for line in f.readlines():
            ipal.append(json.loads(line))

    # Extend alarms
    for i in range(len(ipal)):
        if "adjust" not in ipal[i]:
            continue

        # Adjust alert
        print("Adjusting alert: {}".format(ipal[i]["adjust"]))
        for offset, alert, metric in ipal[i]["adjust"]:
            assert offset <= 0 and i + offset >= 0
            ipal[i + offset]["ids"] = alert
            ipal[i + offset]["metrics"] = {
                k: metric for k in ipal[i + offset]["metrics"]
            }

        del ipal[i]["adjust"]

    # Write file to disc
    print("Writing to disc")
    with open_file(sys.argv[1], "wt") as f:
        for out in ipal:
            f.write(json.dumps(out) + "\n")
