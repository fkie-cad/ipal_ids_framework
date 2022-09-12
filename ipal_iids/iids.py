#!/usr/bin/env python3
import argparse
import gzip
import json
import logging
import os
import sys
import time

from pathlib import Path

import ipal_iids.settings as settings

from ids.utils import get_all_iidss

# Prometheus Client
from ipal_iids.tools.prometheus import PrometheusClient
from prometheus_client import Enum, Summary, Counter
IDS_STATE = Enum('ids_state', 'Current IDS state', states=['starting','training','operating'])
PACKET_TIME = Summary('ids_packet_inspection_seconds', 'Time spent inspecting packet', ['method','ids'])
IDS_ALERT = Counter('ids_alert', 'Counts the alert that occur while inspecting', ["ids"])

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
            logging.getLogger("ipal-iids").error("Option '--log' parameter not found")
            exit(1)

    if args.logfile:
        settings.logfile = args.logfile
        logging.basicConfig(
            filename=settings.logfile, level=settings.log, format=settings.logformat
        )
    else:
        logging.basicConfig(level=settings.log, format=settings.logformat)

    settings.logger = logging.getLogger("ipal-iids")


def dump_ids_default_config(name):
    if name not in settings.idss:
        settings.logger.error("IDS {} not found! Use one of:".format(name))
        settings.logger.error(", ".join(settings.idss.keys()))
        exit(1)

    # Create IDSs default config
    config = {
        name: {
            "_type": name,
            **get_all_iidss()[name](name=name)._default_settings,
        }
    }
    config[name]["model-file"] = "./model"

    # Output a pre-filled config file
    print(json.dumps(config, indent=4))
    exit(0)


def prepare_arg_parser(parser):

    # Input and output
    parser.add_argument(
        "--train.ipal",
        dest="train_ipal",
        metavar="FILE",
        help="input file of IPAL messages to train the IDS on ('-' stdin, '*.gz' compressed).",
        required=False,
    )
    parser.add_argument(
        "--train.state",
        dest="train_state",
        metavar="FILE",
        help="input file of IPAL state messages to train the IDS on ('-' stdin, '*.gz' compressed).",
        required=False,
    )
    parser.add_argument(
        "--train.time",
        dest="train_time",
        metavar="INT",
        help="[UNUSED] initial training time for the IDS - only effective when train.ipal==live.ipal or train.state==live.ipal. When both are a simple file, IDS will re-inspect trained data"
    )
    parser.add_argument(
        "--live.ipal",
        dest="live_ipal",
        metavar="FILE",
        help="input file of IPAL messages to perform the live detection on ('-' stdin, '*.gz' compressed).",
        required=False,
    )
    parser.add_argument(
        "--live.state",
        dest="live_state",
        metavar="FILE",
        help="input file of IPAL state messages to perform the live detection on ('-' stdin, '*.gz' compressed).",
        required=False,
    )
    parser.add_argument(
        "--output",
        dest="output",
        metavar="FILE",
        help="output file to write the anotated IDS output to (Default:none, '-' stdout, '*,gz' compress).",
        required=False,
    )
    parser.add_argument(
        "--config",
        dest="config",
        metavar="FILE",
        help="load IDS configuration and parameters from the specified file ('*.gz' compressed).",
        required=False,
    )

    # Further options
    parser.add_argument(
        "--default.config",
        dest="defaultconfig",
        metavar="IDS",
        help="dump the default configuration for the specified IDS to stdout and exit, can be used as a basis for writing IDS config files. Available IIDSs are: {}".format(
            ",".join(settings.idss.keys())
        ),
        required=False,
    )

    parser.add_argument(
        "--retrain",
        dest="retrain",
        help="retrain regardless of a trained model file being present.",
        action="store_true",
        required=False,
    )

    # Logging
    parser.add_argument(
        "--log",
        dest="log",
        metavar="STR",
        help="define logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) (Default: WARNING).",
        required=False,
    )
    parser.add_argument(
        "--logfile",
        dest="logfile",
        metavar="FILE",
        default=False,
        help="file to log to (Default: stderr).",
        required=False,
    )

    # Gzip compress level
    parser.add_argument(
        "--compresslevel",
        dest="compresslevel",
        metavar="INT",
        default=9,
        help="set the gzip compress level. 0 no compress, 1 fast/large, ..., 9 slow/tiny. (Default: 9)",
        required=False,
    )


# Returns IDS according to the arguments
def parse_ids_arguments(args):
    idss = []

    # IDS defined by config file
    for name, config in settings.idss.items():
        idss.append(get_all_iidss()[config["_type"]](name=name))
    return idss


def load_settings(args):  # noqa: C901

    if args.defaultconfig:
        dump_ids_default_config(args.defaultconfig)

    # Gzip compress level
    if args.compresslevel:
        try:
            settings.compresslevel = int(args.compresslevel)
        except ValueError:
            settings.logger.error(
                "Option '--compresslevel' must be an integer from 0-9"
            )
            exit(1)

        if settings.compresslevel < 0 or 9 < settings.compresslevel:
            settings.logger.error(
                "Option '--compresslevel' must be an integer from 0-9"
            )
            exit(1)

    # Catch incompatible combinations
    if not args.config:
        settings.logger.error("no IDS configuration provided, exiting")
        exit(1)

    # Parse training input (TODO: its probably better if these also were file handles like live input for consistency. file handles can also be reset (e.g. fd.seek(0,0)))
    if args.train_ipal:
        settings.train_ipal = args.train_ipal
    if args.train_state:
        settings.train_state = args.train_state

    # Parse live ipal input
    if args.live_ipal:
        settings.live_ipal = args.live_ipal
    if settings.live_ipal:
        if settings.live_ipal != "stdout" and settings.live_ipal != "-":
            settings.live_ipalfd = open_file(settings.live_ipal, "r")
        else:
            settings.live_ipalfd = sys.stdin

    # Parse live state input
    if args.live_state:
        settings.live_state = args.live_state
    if settings.live_state:
        if settings.live_state != "stdin" and settings.live_state != "-":
            settings.live_statefd = open_file(settings.live_state, "r")
        else:
            settings.live_statefd = sys.stdin

    # Parse retrain
    if args.retrain:
        settings.retrain = True
        settings.logger.info("Retraining models")

    # Parse output
    if args.output:
        settings.output = args.output
    if settings.output:
        if settings.output != "stdout" and settings.output != "-":
            # clear the file we are about to write to
            open_file(settings.output, "wt").close()
            settings.outputfd = open_file(settings.output, "wt")
        else:
            settings.outputfd = sys.stdout

    # Parse config
    settings.config = args.config

    config_file = Path(settings.config).resolve()
    if config_file.is_file():
        with open_file(settings.config, "r") as f:
            try:
                settings.idss = json.load(f)
            except json.decoder.JSONDecodeError as e:
                settings.logger.error("Error parsing config file")
                settings.logger.error(e)
                exit(1)
    else:
        settings.logger.error("Could not find config file at {}".format(config_file))
        exit(1)


def train_idss(idss):
    # Try to load an existing model from file
    loaded_from_file = []
    for ids in idss:
        if settings.retrain:
            continue

        try:
            if ids.load_trained_model():
                loaded_from_file.append(ids)
                settings.logger.info(
                    "IDS {} loaded a saved model successfully.".format(ids._name)
                )
        except NotImplementedError:
            settings.logger.info(
                "Loading model from file not implemented for {}.".format(ids._name)
            )

    # Check if all datasets necessary for the selected IDSs are provided
    for ids in idss:
        if ids in loaded_from_file:
            continue
        if ids.requires("train.ipal") and settings.train_ipal:
            continue
        if ids.requires("train.state") and settings.train_state:
            continue

        settings.logger.error(
            "Required arguement: {} for IDS {}".format(ids._requires, ids._name)
        )
        exit(1)

    # Give the various IDSs the dataset they need in their learning phase
    for ids in idss:
        if ids in loaded_from_file:
            continue

        start = time.time()
        settings.logger.info("Training of {} started at {}".format(ids._name, start))

        ids.train(ipal=settings.train_ipal, state=settings.train_state)

        end = time.time()
        settings.logger.info(
            "Training of {} ended at {} ({}s)".format(ids._name, end, end - start)
        )

        # Try to save the trained model
        try:
            if ids.save_trained_model():
                settings.logger.info(
                    "Saved trained model of {} to file.".format(ids._name)
                )
        except NotImplementedError:
            settings.logger.info(
                "Saving model to file not implemented for {}.".format(ids._name)
            )


def live_idss(idss):
    # Keep track of the last state and message information. Then we are capable of delivering them in the right order.
    ipal_msg = None
    state_msg = None
    _first_ipal_msg = True
    _first_state_msg = True

    while True:
        # load a new ipal message
        if ipal_msg is None and settings.live_ipal:
            line = settings.live_ipalfd.readline()
            if line:
                try:
                    ipal_msg = json.loads(line)
                except:
                    continue # dump broken data - e.g. broken pipe recovery 

        # load a new state
        if state_msg is None and settings.live_state:
            line = settings.live_statefd.readline()
            if line:
                try:
                    state_msg = json.loads(line)
                except:
                    continue # dump broken data - e.g. broken pipe recovery 

        # Determine smallest timestamp ipal or state?
        if ipal_msg and state_msg:
            is_ipal_smaller = ipal_msg.timestamp < state_msg.timestamp
        elif ipal_msg:
            is_ipal_smaller = True
        elif state_msg:
            is_ipal_smaller = False
        else:  # handled all messages from files
            break

        # Process next message
        with PACKET_TIME.labels(method="combined", ids="all").time():
            if is_ipal_smaller:
                ipal_msg["metrics"] = {}
                ipal_msg["ids"] = False

                for ids in idss:
                    with PACKET_TIME.labels(method="ipal", ids=ids._name).time():
                        if ids.requires("live.ipal"):
                            alert, metric = ids.new_ipal_msg(ipal_msg)
                            ipal_msg["ids"] = (
                                ipal_msg["ids"] or alert
                            )  # combine alerts with or (TODO config ?)
                            ipal_msg["metrics"][ids._name] = metric
                            if alert:
                                IDS_ALERT.labels(ids=ids._name).inc()

                if settings.output:

                    if _first_ipal_msg:
                        ipal_msg["_iids-config"] = settings.iids_settings_to_dict()
                        _first_ipal_msg = False

                    settings.outputfd.write(json.dumps(ipal_msg) + "\n")
                    settings.outputfd.flush()
                ipal_msg = None
            else:
                state_msg["metrics"] = {}
                state_msg["ids"] = False

                for ids in idss:
                    with PACKET_TIME.labels(method="state", ids=ids._name).time():
                        if ids.requires("live.state"):
                            alert, metric = ids.new_state_msg(state_msg)
                            state_msg["ids"] = (
                                state_msg["ids"] or alert
                            )  # combine alerts with or (TODO config ?)
                            state_msg["metrics"][ids._name] = metric
                            if alert:
                                IDS_ALERT.labels(ids=ids._name).inc()

                if settings.output:

                    if _first_state_msg:
                        state_msg["_iids-config"] = settings.iids_settings_to_dict()
                        _first_state_msg = False

                    settings.outputfd.write(json.dumps(state_msg) + "\n")
                    settings.outputfd.flush()
                state_msg = None


def main():
    _prom = PrometheusClient()
    IDS_STATE.state('starting')
    # Argument parser and settings
    parser = argparse.ArgumentParser()
    prepare_arg_parser(parser)
    args = parser.parse_args()
    initialize_logger(args)
    load_settings(args)
    idss = parse_ids_arguments(args)

    try:
        # Train IDSs
        settings.logger.info("Start IDS training...")
        IDS_STATE.state('training')
        train_idss(idss)

        # Live IDS
        settings.logger.info("Start IDS live...")
        IDS_STATE.state('operating')
        live_idss(idss)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())

    # Finalize and close
    if settings.output and settings.outputfd != sys.stdout:
        settings.outputfd.close()
    if settings.live_ipal:
        settings.live_ipalfd.close()
    if settings.live_state:
        settings.live_statefd.close()


if __name__ == "__main__":
    main()
