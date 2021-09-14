__description__ = \
"""
Class for simulating genotype phenotype map with linear epistasis model.
"""
__author__ = "Zach Sailer"

from epistasis.simulate.base import BaseSimulation

import numpy as np

class LinearSimulation(BaseSimulation):
    """
    Construct an genotype-phenotype from linear building blocks and
    epistatic coefficients.

    Example
    -------
    Phenotype = b0 + b1 + b2 + b3 + b12 + b13 + b13 + b123

    Parameters
    ---------
    wildtype : str
        Wildtype genotype
    mutations : dict
        Mapping for each site to its alphabet
    order : int
        Order of epistasis in simulated genotype-phenotype map
    betas : array-like
        values of epistatic coefficients (must be positive for this function
        to work. Log is taken)
    model_type : str
        Use a local or global (i.e. Walsh space) epistasis model to construct
        phenotypes
    """

    def __init__(self, wildtype, mutations,
                 model_type='global',
                 **kwargs):

        # Construct epistasis mapping objects (empty)
        super(LinearSimulation, self).__init__(
              wildtype,
              mutations,
              **kwargs)
        self.model_type = model_type

    def build(self):
        """
        Build the phenotype map from epistatic interactions.
        """
        X = self.add_X()

        # Get model type:
        self.data.loc[:,'phenotype'] = np.dot(X, self.epistasis.values)
