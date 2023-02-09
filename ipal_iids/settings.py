from io import TextIOWrapper
import logging

from ids.utils import get_all_iidss

version = "v1.4.3"

# Gzip options
compresslevel = 9  # 0 no compress, 1 large/fast, 9 small/slow

# In and output
config = None
train_ipal = None
train_state = None
train_combiner = None
live_ipal = None
live_ipalfd: TextIOWrapper
live_state = None
live_statefd: TextIOWrapper
retrain = False
output = None
outputfd: TextIOWrapper

# Logging settings
logger = logging.getLogger("ipal-iids")
log = logging.WARNING
logformat = "%(levelname)s:%(name)s:%(message)s"
logfile = None

# IDS parameters
idss = {ids._name: {"_type": ids._name} for ids in get_all_iidss().values()}

combinerconfig = None
combiner = None


def iids_settings_to_dict():
    return {
        "version": version,
        "compresslevel": compresslevel,
        "config": config,
        "combiner_config": combinerconfig,
        "idss": idss,
        "combiner": combiner,
        "train_ipal": train_ipal,
        "train_state": train_state,
        "train_combiner": train_combiner,
        "live_ipal": live_ipal,
        "live_state": live_state,
        "retrain": retrain,
        "output": output,
        "log": log,
        "logformat": logformat,
        "logfile": logfile,
    }
