#!/usr/bin/env python3
import argparse
import logging
import sys

import orjson

import ipal_iids.iids as iids
import ipal_iids.settings as settings
from ids.utils import get_all_iidss, set_idss_to_load


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

    parser.add_argument(
        "--output",
        metavar="output",
        help="file to save the plot to (Default: '': show in matplotlib window)",
        required=False,
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

    with iids.open_file(settings.config, "rb") as f:
        try:
            settings.idss = orjson.loads(f.read())
            # IDSs to be dynamically loaded
            set_idss_to_load([settings.idss[k]["_type"] for k in settings.idss.keys()])

        except UnicodeDecodeError as e:
            settings.logger.error("Error parsing config file")
            settings.logger.error(
                "Make sure you provide a config file, not a model file!"
            )
            settings.logger.error(e)
            exit(1)
        except orjson.JSONDecodeError as e:
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


def plot_models(idss, args):
    for ids in idss:
        try:  # Try to load the trained models
            if not ids.load_trained_model():
                settings.logger.error(
                    f"IDS {ids._name} did not load model successfully."
                )
                continue
        except NotImplementedError:
            settings.logger.error(
                f"Loading model from file not implemented for {ids._name}."
            )

        try:  # Try to plot the trained models
            plt, fig = ids.visualize_model()
        except NotImplementedError:
            settings.logger.error(f"Plotting model not implemented for {ids._name}.")
            continue

        if plt is None or fig is None:
            settings.logger.warning("Nothing to render")
        else:
            fig.suptitle(ids._name)

            if args.output is not None:
                plt.savefig(args.output)
            else:
                plt.show()
            plt.close()


def main():
    # Argument parser and settings
    parser = argparse.ArgumentParser()
    prepare_arg_parser(parser)

    args = parser.parse_args()
    initialize_logger(args)

    idss = load_settings(args)
    plot_models(idss, args)


if __name__ == "__main__":
    main()
