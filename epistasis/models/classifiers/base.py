__description__ = \
"""
Base model for epistasis classifier models.
"""
__author__ = "Zach Sailer"

import gpmap

from epistasis.mapping import EpistasisMap
from epistasis.models.utils import arghandler
from epistasis.models.linear import EpistasisLinearRegression

from sklearn.preprocessing import binarize

import numpy as np
import pandas as pd


class EpistasisClassifierMixin:
    """
    A Mixin class for epistasis classifiers
    """
    def _fit_additive(self, X=None, y=None):
        # Construct an additive model.
        self.Additive = EpistasisLinearRegression(
            order=1, model_type=self.model_type)

        self.Additive.add_gpm(self.gpm,
                              self.genotype_column,
                              self.phenotype_column,
                              self.uncertainty_column)

        # Prepare a high-order model
        self.Additive.epistasis = EpistasisMap(
            sites=self.Additive.Xcolumns,
        )

        # Fit the additive model and infer additive phenotypes
        self.Additive.fit(X=X, y=y)
        return self

    def _fit_classifier(self, X=None, y=None):
        # This method builds x and y from data.
        add_coefs = self.Additive.epistasis.values
        add_X = self.Additive._X(data=X)

        # Project X into padd space.
        X = add_X * add_coefs

        # Label X.
        y = binarize(y.reshape(1, -1), threshold=self.threshold)[0]
        self.classes = y

        # Fit classifier.
        super().fit(X=X, y=y)
        return self

    def fit_transform(self, X=None, y=None, **kwargs):
        self.fit(X=X, y=y, **kwargs)
        ypred = self.predict(X=X)

        # Read new GenotypePhenotypeMap only taking data where ypred=1
        gpm = gpmap.read_dataframe(self.gpm.data.loc[ypred==1,:],
                                   wildtype=self.gpm.wildtype,
                                   mutations=self.gpm.mutations,
                                   site_labels=self.gpm.site_labels)

        return gpm

    def predict(self, X=None):
        Xadd = self.Additive._X(data=X)
        X = Xadd * self.Additive.epistasis.values
        return super().predict(X=X)

    def predict_transform(self, X=None, y=None):
        x = self.predict(X=X)
        y[x <= 0.5] = self.threshold
        return y

    def predict_log_proba(self, X=None):
        Xadd = self.Additive._X(data=X)
        X = Xadd * self.Additive.epistasis.values
        return super().predict_log_proba(X)

    def predict_proba(self, X=None):
        Xadd = self.Additive._X(data=X)
        X = Xadd * self.Additive.epistasis.values
        return super().predict_proba(X=X)
