__description__ = \
"""
Class for generating simulated epistasis maps with options for various
distributions of values.
"""
__author__ = "Zach Sailer"

from epistasis.mapping import EpistasisMap
from numpy import random
from functools import wraps

class DistributionSimulation(EpistasisMap):
    """
    Just like an epistasis map, but with extra methods for setting epistatic
    coefficients
    """
    def __init__(self, gpm, df=None, sites=None, values=None, uncertainties=None):
        super().__init__(df=df, sites=sites, values=values, uncertainties=uncertainties)
        self._gpm = gpm

    @property
    def avail_distributions(self):
        return random.__all__

    def set_order_from_distribution(self, orders, dist="normal", **kwargs):
        """
        Sets epistatic coefficients to values drawn from a statistical
        distribution.

        Distributions are found in SciPy's `random` module. Kwargs are passed
        directly to these methods
        """
        # Get distribution
        try:
            method = getattr(random, dist)
        except AttributeError:
            err = "Distribution now found. Check the `avail_distribution` \n"
            err += "attribute for available distributions.\n"
            raise ValueError(err)

        idx = self.data.orders.isin(orders)
        self.data.loc[idx, "values"] = method(
            size=sum(idx),
            **kwargs
        )
        self._gpm.build()

    @wraps(EpistasisMap.set_values)
    def set_values(self, values, filter=None):
        super().set_values(values, filter=filter)
        self._gpm.build()
