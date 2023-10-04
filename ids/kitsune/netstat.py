## Prep AfterImage cython package
from typing import Any, Dict, List, Union

import numpy as np
import pyximport

pyximport.install()

import ids.kitsune.afterimage as af  # noqa: E402

# import AfterImage_NDSS as af

#
# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


class netStat:
    # Datastructure for efficent network stat queries
    # HostLimit: no more that this many Host identifiers will be tracked
    # HostSimplexLimit: no more that this many outgoing channels from each host will be tracked (purged periodically)
    # Lambdas: a list of 'window sizes' (decay factors) to track for each stream. nan resolved to default [5,3,1,.1,.01]
    def __init__(
        self,
        stats: List[Dict[str, Any]],
        lambdas: List[float],
        hostlimit=255,
        hostsimplexlimit=1000,
    ):
        # Lambdas
        self._lambdas = lambdas

        # HT Limits
        self.hostlimit = hostlimit
        self.sessionlimit = (
            hostsimplexlimit * self.hostlimit * self.hostlimit
        )  # *2 since each dual creates 2 entries in memory
        self.mac_hostlimit = self.hostlimit * 10

        self._stats = []
        for stat_d in stats:
            stat = {
                "type": stat_d["type"],
                "name": stat_d["name"],
                "features": stat_d["features"],
                "typediff": stat_d["typediff"],
                "db": af.incStatDB(limit=getattr(self, stat_d["limit"])),
            }

            self._stats.append(stat)

    # FIXME: appears unused?
    def findDirection(
        self, IPtype, srcIP, dstIP, eth_src, eth_dst
    ):  # cpp: this is all given to you in the direction string of the instance (NO NEED FOR THIS FUNCTION)
        if IPtype == 0:  # is IPv4
            lstP = srcIP.rfind(".")
            src_subnet = srcIP[0:lstP:]
            lstP = dstIP.rfind(".")
            dst_subnet = dstIP[0:lstP:]
        elif IPtype == 1:  # is IPv6
            src_subnet = srcIP[0 : round(len(srcIP) / 2) :]
            dst_subnet = dstIP[0 : round(len(dstIP) / 2) :]
        else:  # no Network layer, use MACs
            src_subnet = eth_src
            dst_subnet = eth_dst

        return src_subnet, dst_subnet

    def updateGetStats(self, features: Dict[str, Any]):
        res = []

        for stat in self._stats:
            if stat["type"] == "1D":
                res_i = np.zeros((3 * len(self._lambdas)))
                id_1 = (
                    "".join([str(features[f]) for f in stat["features"]["ID1"]])
                    if len(stat["features"]["ID1"]) > 1
                    else features[stat["features"]["ID1"][0]]
                )
                for j in range(len(self._lambdas)):
                    res_i[(j * 3) : ((j + 1) * 3)] = stat["db"].update_get_1D_Stats(
                        ID=id_1,
                        t=features[stat["features"]["t1"]],
                        v=features[stat["features"]["v1"]],
                        Lambda=self._lambdas[j],
                        isTypeDiff=stat["typediff"],
                    )
                res.append(res_i)

            elif stat["type"] == "1D2D":
                res_i = np.zeros((7 * len(self._lambdas)))
                id_1 = (
                    "".join([str(features[f]) for f in stat["features"]["ID1"]])
                    if len(stat["features"]["ID1"]) > 1
                    else features[stat["features"]["ID1"][0]]
                )
                id_2 = (
                    "".join([str(features[f]) for f in stat["features"]["ID2"]])
                    if len(stat["features"]["ID2"]) > 1
                    else features[stat["features"]["ID2"][0]]
                )
                for j in range(len(self._lambdas)):
                    res_i[(j * 7) : ((j + 1) * 7)] = stat["db"].update_get_1D2D_Stats(
                        ID1=id_1,
                        ID2=id_2,
                        t1=features[stat["features"]["t1"]],
                        v1=features[stat["features"]["v1"]],
                        Lambda=self._lambdas[j],
                    )
                res.append(res_i)

            else:
                raise Exception(f"Unknown stat type for stat '{stat['name']}'")

        # concatenation of stats into one stat vector
        return np.concatenate(res)

    # FIXME: Never used for anything besides getting number of features(?)
    def getNetStatHeaders(self):
        headers = []

        for stat in self._stats:
            if stat["type"] == "1D":
                for i in range(len(self._lambdas)):
                    headers += [
                        f"{stat['name']}_{h}"
                        for h in stat["db"].getHeaders_1D(
                            Lambda=self._lambdas[i], ID=None
                        )
                    ]

            elif stat["type"] == "1D2D":
                for i in range(len(self._lambdas)):
                    headers += [
                        f"{stat['name']}_{h}"
                        for h in stat["db"].getHeaders_1D2D(
                            Lambda=self._lambdas[i], IDs=None, ver=2
                        )
                    ]

        return headers
