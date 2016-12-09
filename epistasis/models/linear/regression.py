import numpy as _np
from sklearn.linear_model import LinearRegression as _LinearRegression

from epistasis.models.base import BaseModel as _BaseModel
from epistasis.models.base import X_fitter as X_fitter
from epistasis.models.base import X_predictor as X_predictor

class EpistasisLinearRegression(_LinearRegression, _BaseModel):
    """ Ordinary least-squares regression of epistatic interactions.
    """
    def __init__(self, order=1, model_type="global", n_jobs=1, **kwargs):
        # Set Linear Regression settings.
        self.fit_intercept = False
        self.normalize = False
        self.copy_X = False
        self.n_jobs = n_jobs
        self.set_params(model_type=model_type, order=order, **kwargs)

    @X_fitter
    def fit(self, X=None, y=None):
        # Build input linear regression.
        super(self.__class__, self).fit(X,y)

    @X_predictor
    def predict(self, X=None):
        return super(self.__class__, self).predict(X)

    @X_fitter
    def score(self, X=None, y=None):
        return super(self.__class__, self).score(X, y)
