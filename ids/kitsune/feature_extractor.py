import re
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

import ids.kitsune.netstat as ns


class FeatureExtractor:
    def __init__(
        self,
        features_regexp: Dict[str, Any],
        lambdas: List[float],
        max_host: int,
        max_sess: int,
        stats: List[Dict[str, Any]],
    ) -> None:
        self._feature_regexp = {}

        self._nstat = ns.netStat(stats, lambdas, max_host, max_sess)

        for feature, extraction in features_regexp.items():
            self._feature_regexp[feature] = [extraction[0], re.compile(extraction[1])]

    def extract_features(
        self, message: Dict[str, Any], timestamp_offset: Optional[float] = None
    ) -> NDArray[np.float64]:
        features: Dict[str, Any] = {"_zero": 0}
        for feature, extraction in self._feature_regexp.items():
            field = extraction[0]
            regexp: re.Pattern = extraction[1]

            if field not in message.keys():
                features[feature] = 0
                continue

            match: Optional[re.Match[str]] = regexp.search(str(message[field]))
            v_type = type(message[field])
            if match is None:
                features[feature] = 0
                continue

            features[feature] = v_type(match.group(1))
            # fake the timestamp if necessary
            if timestamp_offset is not None and feature == "timestamp":
                features[feature] += timestamp_offset

        return self._nstat.updateGetStats(features)

    def get_num_features(self) -> int:
        return len(self._nstat.getNetStatHeaders())
