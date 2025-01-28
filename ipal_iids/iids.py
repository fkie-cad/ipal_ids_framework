#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import IO

import orjson
from zlib_ng import gzip_ng_threaded as gzip

import combiner.utils
import ipal_iids.settings as settings
import preprocessors.utils
from combiner.utils import get_all_combiner
from ids.ids import BatchIDS
from ids.utils import get_all_iidss, load_ids
from ids.utils import paths as ids_paths
from ids.utils import set_idss_to_load

# global variables used by live_idss_batch to track message state
_FIRST_BATCHED_IPAL_MSG: bool
_FIRST_BATCHED_STATE_MSG: bool


def open_file(
    filename: str | os.PathLike | Path,
    mode: str = "r",
    compresslevel: int | None = None,
    force_gzip: bool = False,
) -> IO | None:
    """
    Wrapper to hide .gz files and stdin/stdout

    :param filename: filename to open
    :param mode: file mode
    :param compresslevel: force compresslevel, if None level is taken from settings
    :param force_gzip: if file should be treated as gzip even without .gz ending
    :return: file-like object or None
    """

    # make sure filename is a string and not path-like object
    filename = str(filename)

    if not compresslevel:
        compresslevel = settings.compresslevel

    if filename == "-" and force_gzip:
        # we can give gzip stdin/stdout to read / write from if explicitly wanted
        if "r" in mode:
            filename = sys.stdin
        elif "w" in mode:
            filename = sys.stdout

    if filename is None:
        return None
    elif filename.endswith(".gz") or force_gzip:
        return gzip.open(filename, mode=mode, compresslevel=compresslevel, threads=-1)
    elif (filename == "-" or filename == "stdin") and "r" in mode:
        return sys.stdin
    elif (filename == "-" or filename == "stdout") and "w" in mode:
        return sys.stdout
    else:
        # do we *need* files to behave line by line?
        if "b" in mode:
            return open(filename, mode=mode)
        else:
            return open(filename, mode=mode, buffering=1)


def copy_file_to_tmp_file(filein):
    # Generate temporary file and read stdin to it
    filename = f"tmp-{random.randint(1000, 9999)}.gz"
    with open_file(filename, "wt") as ftmp:
        try:
            with open_file(filein, "r") as filein:
                ftmp.write(filein.read())
        except:  # noqa: E722
            # Remove tmp file upon failure
            os.remove(filename)
            raise Exception("Failed copying stdin to temporary file")
    return filename


# Initialize logger
def initialize_logger(args):
    # Decide if hostname is added
    if args.hostname:
        settings.hostname = True
        settings.logformat = f"%(asctime)s:{socket.gethostname()}:{settings.logformat}"

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
    if name not in list(ids_paths.keys()):
        settings.logger.error(f"IDS {name} not found! Use one of:")
        settings.logger.error(", ".join(list(ids_paths.keys())))
        exit(1)

    # Create IDSs default config
    config = {
        name: {
            "_type": name,
            **load_ids(name)[name](name=name)._default_settings,
        }
    }

    # Output a pre-filled config file
    print(orjson.dumps(config, option=orjson.OPT_INDENT_2).decode())
    exit(0)


def dump_combiner_default_config(name):
    if name not in list(combiner.utils.paths.keys()):
        settings.logger.error(f"Combiner {name} not found! Use one of:")
        settings.logger.error(", ".join(list(combiner.utils.paths.keys())))
        exit(1)

    # Create Combiners default config
    settings.combiner = {"_type": name}
    config = {
        "_type": name,
        **combiner.utils.load_combiner(name)[name]()._default_settings,
    }

    # Output a pre-filled config file
    print(orjson.dumps(config, option=orjson.OPT_INDENT_2).decode())
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
        "--train.combiner",
        dest="train_combiner",
        metavar="FILE",
        help="input file of IPAL or state messages to train the combiner on ('-' stdin, '*.gz' compressed).",
        required=False,
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
        help="output file to write the annotated IDS output to (Default:none, '-' stdout, '*,gz' compress).",
        required=False,
    )
    parser.add_argument(
        "--config",
        dest="config",
        metavar="FILE",
        help="load IDS configuration and parameters from the specified file ('*.gz' compressed).",
        required=False,
    )
    parser.add_argument(
        "--combiner.config",
        dest="combiner_config",
        metavar="FILE",
        help="load Combiner configuration and parameters from the specified file ('*.gz' compressed).",
        required=False,
    )

    # Further options
    parser.add_argument(
        "--default.config",
        dest="defaultconfig",
        metavar="IDS",
        help=f"dump the default configuration for the specified IDS to stdout and exit, can be used as a basis for "
        f"writing IDS config files. Available IIDSs are: {','.join(list(ids_paths.keys()))}",
        required=False,
    )
    parser.add_argument(
        "--combiner.default.config",
        dest="defaultcombinerconfig",
        metavar="Combiner",
        help=f"dump the default configuration for the specified Combiner to stdout and exit, can be used as a basis "
        f"for writing Combiner config files. "
        f"Available Combiners are: {','.join(list(combiner.utils.paths.keys()))}",
        required=False,
    )

    parser.add_argument(
        "--retrain",
        dest="retrain",
        help="retrain regardless of a trained model file being present.",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--hostname",
        dest="hostname",
        help="Add the hostname to the output.",
        required=False,
        action="store_true",
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
        default=6,
        help="set the gzip compress level. 0 no compress, 1 fast/large, ..., 9 slow/tiny. (Default: 6)",
        required=False,
    )

    # Version number
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {settings.version}"
    )

    parser.add_argument(
        "--extra.config",
        dest="extraconfig",
        help="load IDSs and Combiners residing outside of IPAL",
        metavar="FILE",
        required=False,
    )

    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )

    parser.add_argument(
        "--live.batch",
        dest="live_batch",
        metavar="INT",
        type=int,
        default=0,
        help="commands to use batching and defines the batch size",
        required=False,
    )


# Returns IDS according to the arguments
def parse_ids_arguments():
    idss = []

    # IDS defined by config file
    for name, config in settings.idss.items():
        if name.startswith("_"):
            settings.logger.info(f"Ignored {name} since it starts with '_'")
            continue
        idss.append(get_all_iidss()[config["_type"]](name=name))
    return idss


# Returns IDS according to the arguments
def parse_combiner_arguments():
    return get_all_combiner()[settings.combiner["_type"]]()


def load_settings(args):  # noqa: C901
    if args.defaultconfig:
        dump_ids_default_config(args.defaultconfig)

    if args.defaultcombinerconfig:
        dump_combiner_default_config(args.defaultcombinerconfig)

    # Gzip compress level
    if args.compresslevel:
        try:
            settings.compresslevel = int(args.compresslevel)
        except ValueError:
            settings.logger.error(
                "Option '--compresslevel' must be an integer from 0-9"
            )
            exit(1)

        if 9 < settings.compresslevel < 0:
            settings.logger.error(
                "Option '--compresslevel' must be an integer from 0-9"
            )
            exit(1)

    # Catch incompatible combinations
    if not args.config:
        settings.logger.error("no IDS configuration provided, exiting")
        exit(1)

    # Parse training input
    if args.train_ipal:
        settings.train_ipal = args.train_ipal
    if args.train_state:
        settings.train_state = args.train_state
    if args.train_combiner:
        settings.train_combiner = args.train_combiner

    # Parse live ipal input
    if args.live_ipal:
        settings.live_ipal = args.live_ipal
    if settings.live_ipal:
        settings.live_ipalfd = open_file(settings.live_ipal, "r")

    # Parse live state input
    if args.live_state:
        settings.live_state = args.live_state
    if settings.live_state:
        settings.live_statefd = open_file(settings.live_state, "r")

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
            open_file(settings.output, "wb").close()
            settings.outputfd = open_file(settings.output, "wb")
        else:
            settings.outputfd = sys.stdout

    # Parse IDS config
    settings.config = args.config

    config_file = Path(settings.config).resolve()
    if config_file.is_file():
        with open_file(settings.config, "rb") as f:
            try:
                settings.idss = orjson.loads(f.read())
                # IDSs to be dynamically loaded
                set_idss_to_load(
                    [settings.idss[k]["_type"] for k in settings.idss.keys()]
                )

            except orjson.JSONDecodeError as e:
                settings.logger.error("Error parsing config file")
                settings.logger.error(e)
                exit(1)
    else:
        settings.logger.error(
            f"Could not find config file at {os.path.relpath(config_file)}"
        )
        exit(1)

    # Parse Combiner config
    if args.combiner_config:
        settings.combinerconfig = args.combiner_config

        config_file = Path(settings.combinerconfig).resolve()
        if config_file.is_file():
            with open_file(settings.combinerconfig, "rb") as f:
                try:
                    settings.combiner = orjson.loads(f.read())
                    combiner.utils.combiner_to_use_name = settings.combiner["_type"]
                except orjson.JSONDecodeError as e:
                    settings.logger.error("Error parsing config file")
                    settings.logger.error(e)
                    exit(1)
        else:
            settings.logger.error(
                f"Could not find config file at {os.path.relpath(config_file)}"
            )
            exit(1)

    else:
        settings.logger.warning("No combiner defined. Using default Any combiner!")
        settings.combiner = {"_type": "Any"}
        combiner.utils.combiner_to_use_name = settings.combiner["_type"]


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
                    f"IDS {ids._name} loaded a saved model successfully."
                )
        except NotImplementedError:
            settings.logger.info(
                f"Loading model from file not implemented for {ids._name}."
            )

    # Check if all datasets necessary for the selected IDSs are provided
    for ids in idss:
        if ids in loaded_from_file:
            continue
        if ids.requires("train.ipal") and settings.train_ipal:
            continue
        if ids.requires("train.state") and settings.train_state:
            continue

        settings.logger.error(f"Required argument: {ids._requires} for IDS {ids._name}")
        exit(1)

    # If training file is stdin, read stdin and save it to a temporary file
    # Because training multiple IIDSs on stdin is not possible since stdin can only be read once
    if settings.train_ipal == "-" and len(idss) > 1:
        settings.logger.info("Copying training stdin to temporary file.")
        tmp_ipal_file = copy_file_to_tmp_file(settings.train_ipal)
        settings.train_ipal = tmp_ipal_file
    else:
        tmp_ipal_file = None

    if settings.train_state == "-" and len(idss) > 1:
        settings.logger.info("Copying training stdin to temporary file.")
        tmp_state_file = copy_file_to_tmp_file(settings.train_state)
        settings.train_state = tmp_state_file
    else:
        tmp_state_file = None

    try:
        # Give the various IDSs the dataset they need in their learning phase
        for ids in idss:
            if ids in loaded_from_file:
                continue

            start = time.time()
            settings.logger.info(f"Training of {ids._name} started at {start}")

            ids.train(ipal=settings.train_ipal, state=settings.train_state)

            end = time.time()
            settings.logger.info(
                f"Training of {ids._name} ended at {end} ({end - start}s)"
            )

            # Try to save the trained model
            try:
                if ids.save_trained_model():
                    settings.logger.info(f"Saved trained model of {ids._name} to file.")
            except NotImplementedError:
                settings.logger.info(
                    f"Saving model to file not implemented for {ids._name}."
                )

    finally:
        # Remove temporary files
        if tmp_ipal_file is not None:
            os.remove(tmp_ipal_file)
        if tmp_state_file is not None:
            os.remove(tmp_state_file)


def train_combiner(combiner):
    # Try to load an existing model from file
    if not settings.retrain:
        try:
            if combiner.load_trained_model():
                settings.logger.info(
                    f"Combiner {combiner._name} loaded a saved model successfully."
                )
                return

        except NotImplementedError:
            settings.logger.info(
                f"Loading model from file not implemented for {combiner._name}."
            )

    # Test if training data is required and provided
    if combiner._requires_training and settings.train_combiner is None:
        settings.logger.error(
            f"Combiner {combiner._name} requires training data (--train.combiner)"
        )
        exit(1)

    # Train combiner
    start = time.time()
    settings.logger.info(f"Training of {combiner._name} combiner started at {start}")

    combiner.train(settings.train_combiner)

    end = time.time()
    settings.logger.info(
        f"Training of {combiner._name} combiner ended at {end} ({end - start}s)"
    )

    # Try to save the trained model
    try:
        if combiner.save_trained_model():
            settings.logger.info(f"Saved trained model of {combiner._name} to file.")
    except NotImplementedError:
        settings.logger.info(
            f"Saving model to file not implemented for {combiner._name}."
        )


def write_ipal_msg(msg: dict, msg_combiner, first_msg: bool = False) -> None:
    """
    Processes and writes an IPAL message, combining alerts and scores, outputting
    the final message to the appropriate file descriptor or stdout.

    :param msg:
        A dictionary representing the IPAL message.
    :param msg_combiner:
        The combiner used to combine alerts and scores for the IPAL message.
    :param first_msg:
        If True, additional configuration settings are added to the message before it is written.
    :return:
        None. The function modifies the `msg` dictionary in place and writes the resulting message to the output file
        descriptor or stdout.
    """
    alert, score, offset = msg_combiner.combine(msg["alerts"], msg["scores"])
    if offset == 0:
        msg["ids"] = alert
        msg["scores"][msg_combiner._name] = score
    else:
        msg["ids"] = False
        if "adjust" not in msg:
            msg["adjust"] = {}
        msg["adjust"][msg_combiner._name] = [[offset, alert, score]]

    if settings.output:
        if first_msg:
            msg["_iids-config"] = settings.iids_settings_to_dict()

    if settings.output == "-":
        # flushing for all output formats (not just pipes) results in drastic performance losses
        settings.outputfd.write(
            orjson.dumps(
                msg, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE
            ).decode("utf-8")
        )
        settings.outputfd.flush()
    elif settings.output:
        settings.outputfd.write(
            orjson.dumps(
                msg, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_APPEND_NEWLINE
            )
        )

    return


def read_live_ipal_msg(
    ipal_msg: dict | None, state_msg: dict | None
) -> (dict | None, dict | None, bool | None):
    """
    Reads the next IPAL or state message from live input, if available,
    and determines which message has the earlier timestamp.

    :param ipal_msg:
        The current IPAL message or None if not yet loaded.
    :param state_msg:
        The current state message or None if not yet loaded.
    :return:
        A potentially loaded IPAL message, a potentially loaded state message,
        and a boolean indicating which message is earlier.
        If no messages are loaded, the boolean is None.
    """
    if ipal_msg is None and settings.live_ipal:
        line = settings.live_ipalfd.readline()
        if line:
            ipal_msg = orjson.loads(line)

    # load a new state
    if state_msg is None and settings.live_state:
        line = settings.live_statefd.readline()
        if line:
            state_msg = json.loads(line)

    # Determine smallest timestamp ipal or state?
    if ipal_msg and state_msg:
        is_ipal_earlier = ipal_msg["timestamp"] < state_msg["timestamp"]
    elif ipal_msg:
        is_ipal_earlier = True
    elif state_msg:
        is_ipal_earlier = False
    else:  # handled all messages from files
        is_ipal_earlier = None

    return ipal_msg, state_msg, is_ipal_earlier


def process_ipal_msg(
    msg: dict, idss, msg_combiner, first_msg: bool = False, is_ipal: bool = True
) -> None:
    """
    Processes an IPAL or state message by updating alerts and scores provided by the IDSs,
    then writes the processed message to the appropriate file descriptor or stdout.

    :param msg:
        A dictionary representing the IPAL or state message to be processed.
    :param idss:
        A collection of IDSs that generate alerts and scores for the message.
    :param msg_combiner:
        A combiner used to combine the alerts and scores in the message before writing.
    :param first_msg:
        A boolean flag indicating whether this is the first message being processed. Default is False.
    :param is_ipal:
        A boolean flag indicating if the message is an IPAL message (True) or a state message (False). Default is True.
    :return:
        None. The function updates the message and writes it out.
    """
    if "scores" not in msg:
        msg["scores"] = {}
    if "alerts" not in msg:
        msg["alerts"] = {}

    required_type = "live.ipal" if is_ipal else "live.state"

    for ids in idss:
        if ids.requires(required_type):
            if is_ipal:
                alert, score = ids.new_ipal_msg(msg)
            else:
                alert, score = ids.new_state_msg(msg)
            msg["alerts"][ids._name] = alert
            msg["scores"][ids._name] = score

    write_ipal_msg(msg, msg_combiner, first_msg)

    return


def process_batched_ipal_msgs(batch: [dict], idss, msg_combiner) -> None:
    """
    Processes a batch of IPAL or state messages by updating alerts and scores provided by the IDSs,
    then writes each message to the appropriate file descriptor or stdout

    :param batch:
        A list of dictionaries, each representing an IPAL or state message in the batch.
    :param idss:
        A collection of IDSs that generate alerts and scores for the message.
    :param msg_combiner:
        A combiner used to combine the alerts and scores in the message before writing.
    :return:
        None. The function updates and writes each message in the batch.
    """
    required_type = []

    global _FIRST_BATCHED_IPAL_MSG
    global _FIRST_BATCHED_STATE_MSG

    # gather all required types to make sure we handle both
    for msg in batch:
        if "scores" not in msg:
            msg["scores"] = {}
        if "alerts" not in msg:
            msg["alerts"] = {}
        required_type.append(msg["ipal_type"])

    required_types = set(required_type)

    for ids in idss:
        # check that the ids can handle all message types in the batch
        if all(ids.requires(req_type) for req_type in required_types):
            alerts, scores = ids.new_batch(batch)
            for k, msg in enumerate(batch):
                msg["alerts"][ids._name] = alerts[k]
                msg["scores"][ids._name] = scores[k]

    for msg in batch:

        is_first_msg = False
        if _FIRST_BATCHED_IPAL_MSG and "ipal" in msg["ipal_type"]:
            is_first_msg = True
            _FIRST_BATCHED_IPAL_MSG = False
        elif _FIRST_BATCHED_STATE_MSG and "state" in msg["ipal_type"]:
            is_first_msg = True
            _FIRST_BATCHED_STATE_MSG = False
        msg.pop("ipal_type")
        write_ipal_msg(msg, msg_combiner, is_first_msg)

    return


def live_idss_batch(idss, msg_combiner, batch_size: int) -> None:
    """
    Processes live IPAL and state messages in batches by collecting messages,
    generating alerts and scores using the provided IDSs,
    and writes them in the correct order to a file descriptor or stdout.

    :param idss:
        A collection of IDSs that generate alerts and scores for the messages.
    :param msg_combiner:
        A combiner used to combine alerts and scores for each message in the batch.
    :param batch_size:
        The number of messages to collect and process in each batch.
    :return:
        None. The function continuously reads, processes, and writes batches of messages
        until no more messages are available.
    """

    assert batch_size > 0

    for ids in idss:
        if not isinstance(ids, BatchIDS):
            settings.logger.error(
                f"'{ids._name}' is not a BatchIDS, can't do batching! Aborting."
            )
            exit(1)

    # Keep track of the last state and message information. Then we are capable of delivering them in the right order.
    ipal_msg = None
    state_msg = None
    global _FIRST_BATCHED_IPAL_MSG
    global _FIRST_BATCHED_STATE_MSG

    _FIRST_BATCHED_IPAL_MSG = True
    _FIRST_BATCHED_STATE_MSG = True

    run = True

    while run:
        # List of ipal / state messages
        batch = []
        # Collect batch_size number of ipal / state messages
        for b in range(batch_size):
            # load a new ipal message
            ipal_msg, state_msg, is_ipal_earlier = read_live_ipal_msg(
                ipal_msg, state_msg
            )

            # break loop if we have no more new messages
            if is_ipal_earlier is None:
                run = False
                break

            # Process next message
            if is_ipal_earlier:
                ipal_msg["ipal_type"] = "live.ipal"
                batch.append(ipal_msg)
                ipal_msg = None
            else:
                state_msg["ipal_type"] = "live.state"
                batch.append(state_msg)
                state_msg = None

        process_batched_ipal_msgs(batch, idss, msg_combiner)


def live_idss(idss, msg_combiner) -> None:
    """
    Processes live IPAL and state messages, generating alerts and scores using the provided IDSs,
    and writes them in the correct order to a file descriptor or stdout.

    :param idss:
        A collection of IDSs that generate alerts and scores for each message.
    :param msg_combiner:
        A combiner used to combine alerts and scores for each message.
    :return:
        None. The function continuously reads, processes, and writes batches of messages
        until no more messages are available.
    """
    # Keep track of the last state and message information. Then we are capable of delivering them in the right order.
    ipal_msg = None
    state_msg = None
    first_ipal_msg = True
    first_state_msg = True

    while True:
        # load a new ipal message
        ipal_msg, state_msg, is_ipal_earlier = read_live_ipal_msg(ipal_msg, state_msg)

        # handled all messages, end loop
        if is_ipal_earlier is None:
            break

        # Process next message
        if is_ipal_earlier:
            process_ipal_msg(
                ipal_msg, idss, msg_combiner, first_msg=first_ipal_msg, is_ipal=True
            )
            first_ipal_msg = False
            ipal_msg = None
        else:
            process_ipal_msg(
                state_msg, idss, msg_combiner, first_msg=first_state_msg, is_ipal=False
            )
            first_state_msg = False
            state_msg = None


# Load data from extra config file
def load_extra_config(args):
    if args.extraconfig:
        # Load config
        econf_file = open_file(args.extraconfig, "rb")
        econf_json = orjson.loads(econf_file.read())
        # Root path
        rp = Path(os.path.abspath(args.extraconfig)).parent
        # Add idss
        for eo in econf_json["IDS"]:
            ids_paths[eo["name"]] = os.path.join(rp, eo["path"])
        # Add combiners
        for eo in econf_json["Combiner"]:
            combiner.utils.paths[eo["name"]] = os.path.join(rp, eo["path"])
        # Add preprocessors
        for eo in econf_json["Preprocessor"]:
            preprocessors.utils.preprocessor_paths[eo["name"]] = os.path.join(
                rp, eo["path"]
            )
        # Close file
        econf_file.close()
        # Reload settings
        settings.idss = {
            ids_name: {"_type": ids_name} for ids_name in list(ids_paths.keys())
        }


def main():
    # Argument parser and settings
    parser = argparse.ArgumentParser(
        prog="ipal-iids",
        description="This program contains the ipal-iids framework together with implementations of several IIDSs "
        "based on the IPAL message and state format.",
        conflict_handler="resolve",
    )
    prepare_arg_parser(parser)
    args = parser.parse_args()
    initialize_logger(args)

    # Load extra config
    load_extra_config(args)

    # Hook help to reload help strings, in order to load IDS names potentially added by the extra config
    if hasattr(args, "help"):
        prepare_arg_parser(parser)
        parser.print_help()
        exit(0)

    # Load settings
    load_settings(args)

    # Prepare idss and combiner
    idss = parse_ids_arguments()
    combiner = parse_combiner_arguments()

    try:
        # Train IDSs
        settings.logger.info("Start IDS training...")
        train_idss(idss)

        # Train combiner
        settings.logger.info("Start Combiner training...")
        train_combiner(combiner)

        # Live IDS
        settings.logger.info("Start IDS live...")

        if args.live_batch == 0:
            live_idss(idss, combiner)
        else:
            live_idss_batch(idss, combiner, args.live_batch)

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
