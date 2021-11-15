from io import TextIOWrapper
import logging

from .ids.utils import get_all_iidss

# Gzip options
compresslevel = 9  # 0 no compress, 1 large/fast, 9 small/slow

# In and output
config: str
train_ipal = None
train_state = None
live_ipal = None
live_ipalfd: TextIOWrapper
live_state = None
live_statefd: TextIOWrapper
retrain = False
output = None
outputfd: TextIOWrapper

# Logging settings
logger = logging.getLogger("IDS")
log = logging.WARNING
logformat = "%(levelname)s:%(name)s:%(message)s"
logfile = None

# IDS parameters
idss = {ids._name: {"_type": ids._name} for ids in get_all_iidss().values()}
