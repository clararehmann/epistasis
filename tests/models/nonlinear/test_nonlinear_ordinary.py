# Externel imports
import pytest

import numpy as np
from gpmap import GenotypePhenotypeMap

# Module to test
import epistasis
from epistasis.models.nonlinear import EpistasisNonlinearRegression

import warnings

# Ignore fitting warnings
warnings.simplefilter("ignore", RuntimeWarning)

@pytest.fixture
def gpm(test_data):
    """
    Create a genotype-phenotype map
    """

    d = test_data[0]

    return GenotypePhenotypeMap(genotype=d["genotype"],
                                phenotype=d["phenotype"],
                                wildtype=d["wildtype"])


def function(x, A, B):
    return A * x + B


class TestEpistasisNonlinearRegression(object):

    model_type = "local"

    def test_init(self, gpm):
        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type)
        m.add_gpm(gpm)

        # Checks
        assert hasattr(m, 'parameters') is True
        assert hasattr(m, 'function') is True

        parameters = m.parameters

        # Checks
        assert 'A' in parameters
        assert 'B' in parameters

    def test_fit(self, gpm):

        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type,
                                         A=1,
                                         B=0)
        m.add_gpm(gpm)
        m.fit()

        assert True

    def test_score(self, gpm):
        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type,
                                         A=1, B=0)
        m.add_gpm(gpm)

        m.fit()
        score = m.score()
        assert 0 <= score <= 1

    def test_predict(self, gpm):
        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type,
                                         A=1, B=0)
        m.add_gpm(gpm)

        m.fit()
        y = m.predict()

        # Tests
        assert True

    def test_thetas(self, gpm):
        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type,
                                         A=1, B=0)
        m.add_gpm(gpm)
        m.fit()
        coefs = m.thetas
        # Tests
        assert len(coefs) == 6

    def test_hypothesis(self, gpm):

        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type,
                                         A=1, B=0)
        m.add_gpm(gpm)

        m.fit()
        predictions = m.hypothesis()
        # Tests
        assert True


    def test_lnlikelihood(self, gpm):
        m = EpistasisNonlinearRegression(function=function,
                                         model_type=self.model_type,
                                         A=1, B=0)
        m.add_gpm(gpm)

        m.fit()

        # Calculate lnlikelihood
        lnlike = m.lnlikelihood()
        assert lnlike.dtype == float
