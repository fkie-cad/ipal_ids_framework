#!/usr/bin/env python3
import argparse
import gzip
import json
import logging
import sys

import ipal_iids.settings as settings
from ids.utils import get_all_iidss


# Wrapper for hiding .gz files
def open_file(filename, mode):
    if filename.endswith(".gz"):
        return gzip.open(filename, mode=mode, compresslevel=settings.compresslevel)
    else:
        return open(filename, mode=mode, buffering=1)


# Initialize logger
def initialize_logger(args):

    if args.log:
        settings.log = getattr(logging, args.log.upper(), None)

        if not isinstance(settings.log, int):
            logging.getLogger("ipal-visualize-model").error(
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

    settings.logger = logging.getLogger("ipal-visualize-model")


def prepare_arg_parser(parser):

    parser.add_argument(
        "config",
        metavar="FILE",
        help="load the IDS configuration of the trained model ('*.gz' compressed).",
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


def load_settings(args):

    settings.config = args.config

    with open_file(settings.config, "r") as f:
        try:
            settings.idss = json.load(f)
        except json.decoder.JSONDecodeError as e:
            settings.logger.error("Error parsing config file")
            settings.logger.error(e)
            exit(1)

    # Initialize IDSs
    idss = []
    for name, config in settings.idss.items():
        try:
            idss.append(get_all_iidss()[config["_type"]](name=name))
        except TypeError:
            settings.logger.error(
                "Failed loading model. Make sure you provide a config file, not a model file!"
            )
            sys.exit(1)

    return idss


def plot_models(idss):
    for ids in idss:
        try:  # Try to load the trained models
            if not ids.load_trained_model():
                settings.logger.error(
                    "IDS {} did not load model successfully.".format(ids._name)
                )
                continue
        except NotImplementedError:
            settings.logger.error(
                "Loading model from file not implemented for {}.".format(ids._name)
            )

        try:  # Try to plot the trained models
            plt, fig = ids.visualize_model()
        except NotImplementedError:
            settings.logger.error(
                "Plotting model not implemented for {}.".format(ids._name)
            )
            continue

        if plt is None or fig is None:
            settings.logger.warning("Nothing to render")
        else:
            fig.suptitle(ids._name)
            plt.show()
            plt.close()


def main():
    # Argument parser and settings
    parser = argparse.ArgumentParser()
    prepare_arg_parser(parser)

    args = parser.parse_args()
    initialize_logger(args)

    idss = load_settings(args)
    plot_models(idss)


if __name__ == "__main__":
    main()
