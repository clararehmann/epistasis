__description__ = \
"""
Functions for generating epistatic matrices. This is a much slower python
implementation of the functionality impelmented in matrix_cython.pyx.
"""
__author__ = "Zach Sailer"

import numpy as np
import pandas as pd

import warnings

import epistasis.matrix.matrix_python as py

# Try importing model matrix builder from cython extension for speed up.
try:
    import epistasis.matrix.matrix_cython as cy
except ImportError:
    cy = None
    warnings.warn("Could not load cython extension, build_model_matrix'.\n")


def _encode_vectors(binary_genotypes, model_type='global'):
    """
    Encode a set of binary genotypes is input vectors for the given model.

    For the global model, genotypes are mapped as follows:
        [0,0,0] -> [1,1,1,1]
        [1,0,0] -> [1,-1,1,1]
        [1,1,1] -> [1,-1,-1,1] etc.

    For the local model, genotypes are mapped as follows:
        [0,0,0] -> [1,0,0,0]
        [1,0,0] -> [1,1,0,0]
        [1,1,1] -> [1,1,1,1] etc.

    Parameters
    ----------
    binary_genotypes : list or array
        List of genotypes in their binary representation

    model_type : string
        Type of epistasis model (global/Hadamard, local/Biochemical).

    Returns
    -------
    X : numpy.ndarray
        2D numpy array (num_genotypes by num_sites + 1) of floats that encodes
        each genotype.
    """
    # Initialize vector container
    vectors = []

    # Handle a global model
    if model_type == 'global':

        for i, genotype in enumerate(binary_genotypes):
            vector = np.array([0] + list(genotype), dtype=float)
            vector[vector==1] = -1
            vector[vector==0] = 1
            vectors.append(vector)

    # Handle a local model.
    elif model_type == 'local':

        for i, genotype in enumerate(binary_genotypes):
            vector = np.array([1] + list(genotype), dtype=float)
            vectors.append(vector)

    # Don't understand the model
    else:
        Exception("Unrecognized model type.")

    return np.array(vectors)


def get_model_matrix(binary_genotypes, sites, model_type='global',use_cython=True):
    """
    Get a model matrix for a given set of genotypes and coefficients.

    Parameters
    ----------
    binary_genotypes : list or array
        List of genotypes in their binary representation

    sites : list
        List of epistatic interaction sites.

    model_type : string
        Type of epistasis model (global/Hadamard, local/Biochemical).

    use_cython : bool
        use the cython build_matrix calculation

    Returns
    -------
    X : numpy.ndarray
        array encoding which parameters are needed to describe which genotype
    """
    # Convert sites to array of arrays
    sites = np.array([np.array(s) for s in sites],dtype=object)

    # Encode genotypes
    encoding_vectors = _encode_vectors(binary_genotypes, model_type=model_type)

    if use_cython and cy is None:
        warnings.warn("Could not load cython extension for build_model_matrix\n")
        use_cython = False

    if use_cython:
        return cy.build_model_matrix(encoding_vectors,sites)
    else:
        return py.build_model_matrix(encoding_vectors,sites)
