#!/usr/bin/env python3
import argparse
import logging

import orjson

import ipal_iids.iids as iids
import ipal_iids.settings as settings


# Initialize logger
def initialize_logger(args):
    if args.log:
        settings.log = getattr(logging, args.log.upper(), None)

        if not isinstance(settings.log, int):
            logging.getLogger("ipal-extend-alarms").error(
                "Option '--log' parameter not found"
            )
            exit(1)

    if args.logfile:
        settings.logfile = args.logfile
        logging.basicConfig(
            filename=settings.logfile, level=settings.log, format=settings.logformat
        )
    else:
        logging.basicConfig(level=settings.log, format=settings.logformat)

    settings.logger = logging.getLogger("ipal-extend-alarms")


def prepare_arg_parser(parser):
    parser.add_argument(
        "files",
        metavar="FILE",
        help="files to extend alarms ('*.gz' compressed).",
        nargs="+",
    )

    # Logging
    parser.add_argument(
        "--log",
        dest="log",
        metavar="STR",
        help="define logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default is WARNING.",
        required=False,
    )
    parser.add_argument(
        "--logfile",
        dest="logfile",
        metavar="FILE",
        default=False,
        help="File to log to. Default is stderr.",
        required=False,
    )

    # Version number
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {settings.version}"
    )


def extend_alarms(file):
    ipal = []

    # Load file into memory
    with iids.open_file(file, mode="r") as f:
        for line in f:
            ipal.append(orjson.loads(line))

    # Extend alarms
    for i in range(len(ipal)):
        if "adjust" not in ipal[i]:
            continue

        # For each IDS that wants to adjust something
        for ids in ipal[i]["adjust"]:
            # Adjust alert
            for offset, alert, metric in ipal[i]["adjust"][ids]:
                assert offset <= 0

                if i + offset < 0:  # Log warning!
                    settings.logger.error(
                        f"Offset is {offset + i}! Defaulting to dataset start."
                    )
                    offset = -i

                # TODO does not involve the decision of a combiner and uses simply OR!
                ipal[i + offset]["ids"] = ipal[i + offset]["ids"] or alert
                ipal[i + offset]["alerts"][ids] = alert
                ipal[i + offset]["scores"][ids] = metric

        del ipal[i]["adjust"]

    # Write file to disc
    with iids.open_file(file, "wb") as f:
        for out in ipal:
            f.write(orjson.dumps(out, option=orjson.OPT_APPEND_NEWLINE))


def main():
    parser = argparse.ArgumentParser()
    prepare_arg_parser(parser)

    args = parser.parse_args()
    initialize_logger(args)

    N = 0
    for file in args.files:
        N += 1
        settings.logger.info(f"Extending Alarms ({N}/{len(args.files)}) {file}")

        extend_alarms(file)


if __name__ == "__main__":
    main()
