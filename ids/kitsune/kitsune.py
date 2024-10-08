#!/usr/bin/env python3
from typing import Any, Dict

import orjson

from ids.ids import MetaIDS
from ids.kitsune.feature_extractor import FeatureExtractor
from ids.kitsune.KitNET.KitNET import KitNET


class Kitsune(MetaIDS):
    _name = "Kitsune"
    _description = "Kitsune 🦊"
    _requires = ["train.ipal", "live.ipal"]
    _kitsune_default_settings = {
        "offset_live_timestamps": True,
        "max_host": 10000000000,
        "max_sess": 10000000000,
        "threshold": 10,
        "max_autoencoder_size": 10,
        "fm_grace_period": 10000,
        "learning_rate": 0.1,
        "hidden_ratio": 0.75,
        "lambdas": [5, 3, 1, 0.1, 0.01],
        "features_regexp": {
            "srcMAC": [
                "src",
                r"([abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d])",
            ],
            "dstMAC": [
                "dest",
                r"([abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d]\.[abcdef\d][abcdef\d])",
            ],
            "srcIP": ["src", r"(\d+\.\d+\.\d+\.\d+)"],
            "dstIP": ["dest", r"(\d+\.\d+\.\d+\.\d+)"],
            "srcPort": ["src", r"\d+\.\d+\.\d+\.\d+:(\d+)"],
            "dstPort": ["src", r"\d+\.\d+\.\d+\.\d+:(\d+)"],
            "datagramSize": ["length", r"(.*)"],
            "timestamp": ["timestamp", r"(.*)"],
        },
        "stats": [
            # MAC.IP: Stats on src MAC-IP relationships
            {
                "name": "MIstat",
                "type": "1D",
                "features": {
                    "ID1": ["srcMAC", "srcIP"],
                    "t1": "timestamp",
                    "v1": "datagramSize",
                },
                "typediff": False,
                "limit": "mac_hostlimit",
            },
            # Host-Host BW: Stats on the dual traffic behavior between srcIP and dstIP
            {
                "name": "HHstat",
                "type": "1D2D",
                "features": {
                    "ID1": ["srcIP"],
                    "ID2": ["dstIP"],
                    "t1": "timestamp",
                    "v1": "datagramSize",
                },
                "typediff": False,
                "limit": "hostlimit",
            },
            # Host-Host jitter: Stats on the dual traffic behavior between srcIP and dstIP
            {
                "name": "HHstat_ji",
                "type": "1D",
                "features": {
                    "ID1": ["srcIP", "dstIP"],
                    "t1": "timestamp",
                    "v1": "_zero",
                },
                "typediff": True,
                "limit": "hostlimit",
            },
            # Hostport-Hostport BW: Stats on the dual traffic behavior between srcIP:srcPort and dstIP:dstPort
            {
                "name": "HpHpstat",
                "type": "1D2D",
                "features": {
                    "ID1": ["srcIP", "srcPort"],
                    "ID2": ["dstIP", "dstPort"],
                    "t1": "timestamp",
                    "v1": "datagramSize",
                },
                "typediff": False,
                "limit": "sessionlimit",
            },
        ],
    }
    _supports_preprocessor = False
    _fe: FeatureExtractor
    _detector: KitNET

    def __init__(self, name=None):
        super().__init__(name=name)
        self._add_default_settings(self._kitsune_default_settings)
        self._last_training_ts = 0.0
        self._last_training_ts_delta = 0.0
        self._ts_offset = 0.0

    def train(self, ipal=None, state=None):
        self._fe = FeatureExtractor(
            self.settings["features_regexp"],
            self.settings["lambdas"],
            self.settings["max_host"],
            self.settings["max_sess"],
            self.settings["stats"],
        )

        self._detector = KitNET(
            n=self._fe.get_num_features(),
            max_autoencoder_size=self.settings["max_autoencoder_size"],
            FM_grace_period=self.settings["fm_grace_period"],
            learning_rate=self.settings["learning_rate"],
            hidden_ratio=self.settings["hidden_ratio"],
        )

        with self._open_file(ipal) as f:
            for line in f:
                ipal_msg = orjson.loads(line)

                ts = ipal_msg["timestamp"]
                self._last_training_ts_delta = ts - self._last_training_ts
                self._last_training_ts = ts

                features = self._fe.extract_features(ipal_msg)
                self._detector.train(features)

    def new_ipal_msg(self, msg: Dict[str, Any]):
        if self.settings["offset_live_timestamps"] and self._ts_offset == 0.0:
            self._ts_offset = (
                self._last_training_ts + self._last_training_ts_delta - msg["timestamp"]
            )

        features = self._fe.extract_features(msg, timestamp_offset=self._ts_offset)
        score = self._detector.execute(features)

        return bool(score > self.settings["threshold"]), float(score)
