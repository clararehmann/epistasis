__description__ = \
"""
Epistasis Pipeline module.
"""
__author__ = "Zach Sailer"

from epistasis.stats import pearson
from epistasis.models.base import BaseModel
from epistasis.models.utils import arghandler

import numpy as np

class EpistasisPipeline(list, BaseModel):
    """
    Construct a pipeline of epistasis models to run in series.

    This object is a subclass of a Python list. This means, all objects are
    changed in-place. This also means, you can append, prepend, remove,
    and rearrage Pipelines as you would a list.

    The models fit in order. A `fit_transform` method is called on each model
    one at a time. This method returns as transformed GenotypePhenotypeMap
    object and adds it to the next Epistasis model in the list.
    """

    @property
    def num_of_params(self):
        return sum([m.num_of_params for m in self])

    def add_gpm(self,
                gpm,
                genotype_column="genotype",
                phenotype_column=None,
                uncertainty_column=None):
        """
        Add a GenotypePhenotypeMap object to the epistasis model.

        gpm : gpmap.GenotypePhenotypeMap
            genotype phenotype map with genotypes and phenotypes
        genotype_column : str
            name of the genotype column in the gpm
        phenotype_column : str
            name of the phenotype column in the gpm. If None, take the first
            numeric column beside the genotype_column in the gpm
        uncertainty_column : str
            name of column with phenotype uncertainty in gpm. if None, make a
            column `epi_zero_uncertainty` with 1e-6*np.min(phenotype)
        """

        self._gpm = gpm
        self[0].add_gpm(gpm,
                        genotype_column,
                        phenotype_column,
                        uncertainty_column)
        return self

    @property
    def gpm(self):
        return self._gpm

    @gpm.setter
    def gpm(self, gpm):
        raise Exception("Can not add gpm directly. Use `add_gpm` method.")

    def fit(self, X=None, y=None):
        """Fit pipeline.

        Parameters
        ----------
        X : array
            array of genotypes.

        y : array
            array of phentoypes.
        """

        # Fit the first model
        model = self[0]
        gpm = model.fit_transform(X=X, y=y)

        # Then fit every model afterwards.
        for i, model in enumerate(self[1:]):

            # Get transformed gpm from previous model
            model.add_gpm(gpm,
                          self[i-1].genotype_column,
                          self[i-1].phenotype_column,
                          self[i-1].uncertainty_column)

            # Fit model.
            try:
                gpm = model.fit_transform(X=X, y=y)
            except Exception as e:
                print("Failed with {}".format(model))
                print("Input was :")
                print("X : {}".format(X),
                print("y : {}".format(y)))
                raise e

        return self

    @arghandler
    def predict(self, X=None):
        """Predict genotypes.

        Parameters
        ----------
        X : array
            array of genotypes.
        """
        # Predict from last model in the list first.
        model = self[-1]
        ypred = model.predict_transform(X=X)

        # Then work backwards predicting/transforming until the first model.
        for model in self[-2::-1]:
            ypred = model.predict_transform(X=X, y=ypred)

        # Return predictions
        return ypred

    @arghandler
    def hypothesis(self, X=None, thetas=None):
        """Compute phenotypes of genotypes from
        given an array of model parameters.

        Parameters
        ----------
        X : array
            array of genotypes.

        thetas : array
            array of model parameters.
        """
        # Flatten thetas
        t = []
        idx = 0

        # Break up thetas into lists of lists
        for m in self:
            n = m.num_of_params
            t.append(list(thetas[idx:idx + n]))
            idx += n

        # Predict from last model in the list first.
        model = self[-1]
        thetas = t[-1]
        ypred = model.hypothesis_transform(X=X, thetas=thetas)

        # Then work backwards predicting/transforming until the first model.
        for i in range(-2, -(len(self)+1), -1):
            model = self[i]
            thetas = t[i]
            ypred = model.hypothesis_transform(X=X, y=ypred, thetas=thetas)

        # Return predictions
        return ypred

    @arghandler
    def score(self, X=None, y=None):
        """Score the model (using pearson coefficient).

        Parameters
        ----------
        X : array
            array of genotypes.

        y : array
            array of phentoypes.
        """
        ypred = self.predict(X=X)
        return pearson(y, ypred)**2

    @property
    def thetas(self):
        """Flattened array of parameters in all models (ordered by models)."""
        # All parameters in order of models.
        thetas = [m.thetas for m in self]
        return np.concatenate(thetas)

    @arghandler
    def lnlike_of_data(
            self,
            X=None,
            y=None,
            yerr=None,
            thetas=None):
        """Computes likelihood of each genotype-phenotype given model.

        Parameters
        ----------
        X

        Returns
        -------
        lnlike : array
            likelihood for each of each point.
        """
        # Flatten thetas
        t = []
        idx = 0
        # Break up thetas into lists of lists
        for m in self:
            n = m.num_of_params
            t.append(list(thetas[idx:idx + n]))
            idx += n

        # Predict from last model in the list first.
        model = self[-1]
        thetas = t[-1]
        lnlike = np.zeros(len(X))
        ypred = model.hypothesis_transform(X=X, y=y, thetas=thetas)
        lnlike = model.lnlike_transform(X=X, y=ypred, yerr=yerr, lnprior=lnlike, thetas=thetas)

        # Then work backwards predicting/transforming until the first model.
        for i in range(-2, -(len(self)+1), -1):
            model = self[i]
            thetas = t[i]
            ypred = model.hypothesis_transform(X=X, y=y, thetas=thetas)
            lnlike = model.lnlike_transform(X=X, y=ypred, yerr=yerr, lnprior=lnlike, thetas=thetas)

        # Return predictions
        return lnlike

    def lnlikelihood(
            self,
            X=None,
            y=None,
            yerr=None,
            thetas=None):
        """Compute the individal log-likelihoods for each datapoint from a set
        of model parameters.

        Parameters
        ----------
        X : list
            List of genotypes to compute likelihood.

        y : ndarray
            An array of phenotypes to transform.

        yerr : ndarray
            An array of the measured phenotypes standard deviations.

        thetas : ndarray
            array of model parameters. See thetas property for specifics.

        Returns
        -------
        lnlike : float
            log-likelihood of the model parameters.
        """
        lnlike = np.sum(self.lnlike_of_data(X=X, y=y, yerr=yerr, thetas=thetas))
        # If log-likelihood is infinite, set to negative infinity.
        if np.isinf(lnlike) or np.isnan(lnlike):
            return -np.inf
        return lnlike

    # -----------------------------------------------------------
    # Argument handlers.
    # -----------------------------------------------------------

    def _X(self, data=None, method=None):
        """Handle the X argument in this model."""
        X = data
        # If X is None, see if we saved an array.
        if X is None:
            return np.array(self.gpm.data.loc[:,self.genotype_column])
        return X

    def _y(self, data=None, method=None):
        """Handle y arguments in this model."""
        y = data

        if y is None:
            return np.array(self.gpm.data.loc[:,self.phenotype_column])
        return y

    def _yerr(self, data=None, method=None):
        """Handle yerr argument in this model."""
        yerr = data
        if yerr is None:
            return np.array(self.gpm.data.loc[:,self.uncertainty_column])
        return yerr

    def _thetas(self, data=None, method=None):
        """Handle yerr argument in this model."""
        thetas = data
        if thetas is None:
            return self.thetas
        return thetas
