import numpy as np

import ids.kitsune.KitNET.corClust as CC
import ids.kitsune.KitNET.dA as AE
import ipal_iids.settings as settings

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document


class KitNET:
    # n: the number of features in your input dataset (i.e., x \in R^n)
    # m: the maximum size of any autoencoder in the ensemble layer
    # AD_grace_period: the number of instances the network will learn from before producing anomaly scores
    # FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AM_grace_period
    # learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
    # hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
    # feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
    #           where the i-th entry contains a list of the feature indices to be assigned to the i-th autoencoder in the ensemble.
    #           For example, [[2,5,3],[4,0,1],[6,7]]
    def __init__(
        self,
        n,
        FM_grace_period,
        max_autoencoder_size=10,
        learning_rate=0.1,
        hidden_ratio=0.75,
        feature_map=None,
    ):
        # Parameters:
        self.FM_grace_period = FM_grace_period
        if max_autoencoder_size <= 0:
            self.m = 1
        else:
            self.m = max_autoencoder_size
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.n = n

        # Variables
        self.n_trained = 0  # the number of training instances so far
        self.n_executed = 0  # the number of executed instances so far
        self.v = feature_map
        if self.v is None:
            settings.logger.info(
                "Feature-Mapper: train-mode, Anomaly-Detector: off-mode"
            )
        else:
            self.__createAD__()
            settings.logger.info(
                "Feature-Mapper: execute-mode, Anomaly-Detector: train-mode"
            )
        self.FM = CC.corClust(
            self.n
        )  # incremental feature clustering for the feature mapping process
        self.ensembleLayer = []
        self.outputLayer = None

    # force train KitNET on x
    # returns the anomaly score of x during training (do not use for alerting)
    def train(self, x):
        if (
            self.n_trained <= self.FM_grace_period and self.v is None
        ):  # If the FM is in train-mode, and the user has not supplied a feature mapping
            # update the incremental correlation matrix
            self.FM.update(x)
            if (
                self.n_trained == self.FM_grace_period
            ):  # If the feature mapping should be instantiated
                self.v = self.FM.cluster(self.m)
                self.__createAD__()
                settings.logger.info(
                    f"The Feature-Mapper found a mapping: {str(self.n)} features to {str(len(self.v))} autoencoders."
                )
                settings.logger.info(
                    "Feature-Mapper: execute-mode, Anomaly-Detector: train-mode"
                )
        elif self.v is not None and self.outputLayer is not None:  # train
            ## Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub instance for autoencoder 'a'
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].train(xi)
            ## OutputLayer
            self.outputLayer.train(S_l1)
        self.n_trained += 1

    # force execute KitNET on x
    def execute(self, x):
        if self.v is None or self.outputLayer is None:
            raise RuntimeError(
                "KitNET Cannot execute x, because a feature mapping has not yet been learned or provided. Try running process(x) instead."
            )
        else:
            self.n_executed += 1
            ## Ensemble Layer
            S_l1 = np.zeros(len(self.ensembleLayer))
            for a in range(len(self.ensembleLayer)):
                # make sub inst
                xi = x[self.v[a]]
                S_l1[a] = self.ensembleLayer[a].execute(xi)
            ## OutputLayer
            return self.outputLayer.execute(S_l1)

    def __createAD__(self):
        if self.v is None:
            raise RuntimeError("Cannot create AD if v is still None")
        # construct ensemble layer
        for map in self.v:
            params = AE.dA_params(
                n_visible=len(map),
                n_hidden=0,
                lr=self.lr,
                corruption_level=0,
                gracePeriod=0,
                hiddenRatio=self.hr,
            )
            self.ensembleLayer.append(AE.dA(params))

        # construct output layer
        params = AE.dA_params(
            len(self.v),
            n_hidden=0,
            lr=self.lr,
            corruption_level=0,
            gracePeriod=0,
            hiddenRatio=self.hr,
        )
        self.outputLayer = AE.dA(params)


# Copyright (c) 2017 Yisroel Mirsky
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
