# External imports
import unittest
import pytest

import numpy as np
from gpmap import GenotypePhenotypeMap

# Module to test
import epistasis
from epistasis.models.linear import EpistasisRidge


@pytest.fixture
def gpm(test_data):
    """
    Create a genotype-phenotype map
    """

    d = test_data[0]

    return GenotypePhenotypeMap(genotype=d["genotype"],
                                phenotype=d["phenotype"],
                                wildtype=d["wildtype"])


class TestEpistasisRidge(object):

    order = 3

    def test_init(self, gpm):
        model = EpistasisRidge(order=self.order, model_type="local")
        model.add_gpm(gpm)

        # Checks
        check1 = model.order
        check2 = model.model_type
        assert check1 == self.order
        assert check2 == "local"

    def test_fit(self, gpm):
        model = EpistasisRidge(order=self.order, model_type="local")
        model.add_gpm(gpm)
        model.fit()
        # Checks
        check1 = hasattr(model, "Xbuilt")
        check2 = hasattr(model, "coef_")
        check3 = hasattr(model, "epistasis")

        # Tests
        assert check1 is True
        assert check2 is True
        assert check3 is True
        assert "fit" in model.Xbuilt


    def test_predict(self, gpm):
        model = EpistasisRidge(order=self.order, model_type="local")
        model.add_gpm(gpm)
        model.fit()
        check1 = model.predict(X='fit')

        # Tests
        assert "predict" in model.Xbuilt

    def test_score(self, gpm):
        model = EpistasisRidge(order=self.order, model_type="local", alpha=0.1)
        model.add_gpm(gpm)
        model.fit()
        score = model.score()
        # Tests
        assert score >= 0
        assert score <= 1

    def test_hypothesis(self, gpm):
        model = EpistasisRidge(order=self.order, model_type="local")
        model.add_gpm(gpm)
        model.fit()
        # Checks
        check1 = model.hypothesis(thetas=model.coef_)
        # Tests
        assert True

    def test_lnlikelihood(self, gpm):
        model = EpistasisRidge(order=self.order, model_type="local")
        model.add_gpm(gpm)
        model.fit()

        # Calculate lnlikelihood
        lnlike = model.lnlikelihood()
        assert lnlike.dtype == float
