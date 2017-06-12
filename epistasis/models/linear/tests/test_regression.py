from nose import tools
from gpmap.simulate import GenotypePhenotypeSimulation


from ..regression import *


def test_EpistasisLinearRegression_initialization():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    # Checks
    check1 = model.order
    check2 = model.model_type
    tools.assert_equals(check1, 2)
    tools.assert_equals(check2, "local")


def tests_EpistasisLinearRegression_fit_sets_various_attributes():
    gpm = GenotypePhenotypeSimulation.from_length(2)
    model = EpistasisLinearRegression.from_gpm(gpm, order=2, model_type="local")
    model.fit()
    # Checks
    check1 = hasattr(model, "Xfit")
    check2 = hasattr(model, "coef_")
    check3 = hasattr(model, "epistasis")
    # Tests
    tools.assert_true(check1)
    tools.assert_true(check2)
    tools.assert_true(check3)