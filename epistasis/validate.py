import numpy as np
import pandas as pd
from .stats import split_gpm, pearson


def k_fold(gpm,
           model,
           k=10,
           genotype_column="genotype",
           phenotype_column="phenotype"):
    """
    Cross-validation using K-fold validation on a model.

    Parameters
    ----------
    gpm : gpmap.GenotypePhenotypeMap
        genotype phenotype map with data to fit
    model : epistasis model
        epistasis model to test
    k : int (Default 10)
        break data in to k subsets and fit each one independently
    genotype_column : str (Default: genotype)
        column in gpm.data to use for gentotype
    phenotype_column : str (Default : phenotype)
        column in gpm.data to use for phenotype

    Returns
    -------
    scores : list of floats
        k pearson coefficients for k-size test sets calculated using individually
        trained models
    """
    # Get index.
    idx = np.copy(gpm.index)

    # Shuffle index
    np.random.shuffle(idx)

    # Get subsets.
    subsets = np.array_split(idx, k)
    subsets_idx = np.arange(len(subsets))

    # Do k-fold
    scores = []
    for i in range(k):
        # Split index into train/test subsets
        train_idx = np.concatenate(np.delete(subsets, i))
        test_idx = subsets[i]

        # Split genotype-phenotype map
        train, test = split_gpm(gpm, idx=train_idx)

        # Fit model.
        model.add_gpm(train,
                      genotype_column=genotype_column,
                      phenotype_column=phenotype_column)
        model.fit()

        # Score validation set
        pobs = np.array(test.gpm.loc[:,phenotype_column])
        pred = model.predict(X=np.array(test.gpm.loc[:,genotype_column]))

        score = pearson(pobs, pred)**2
        scores.append(score)

    return scores


def holdout(gpm,
            model,
            size=1,
            repeat=1,
            genotype_column="genotype",
            phenotype_column="phenotype"):
    """
    Validate a model by holding-out parts of the data.

    Parameters
    ----------
    gpm : gpmap.GenotypePhenotypeMap
        genotype phenotype map with data to fit
    model : epistasis model
        epistasis model to test
    size : int
        how many observations to hold out when testing
    repeat : int
        how many times to repeat hodling out data
    genotype_column : str (Default: genotype)
        column in gpm.data to use for gentotype
    phenotype_column : str (Default : phenotype)
        column in gpm.data to use for phenotype

    Returns
    -------
    train_scores, test_scores : (list,list)
        repeat-length lists of pearson coefficients for training and test sets
    """

    train_scores = []
    test_scores = []

    model.add_gpm(gpm,
                  genotype_column=genotype_column,
                  phenotype_column=phenotype_column)
    X = model._X()

    for i in range(repeat):
        # Get index.
        idx = np.copy(gpm.index)

        # Shuffle index
        np.random.shuffle(idx)

        # Split model matriay to cross validate).
        train_idx = idx[:size]
        test_idx = idx[size:]
        train_X = X[train_idx, :]
        test_X = X[test_idx, :]

        # Train the model
        train_pheno = gpm.data.loc[train_idx,gpm.phenotype_column]
        model.fit(X=train_X, y=train_pheno)

        train_p = model.predict(X=train_X)
        train_s = pearson(train_p, train_pheno)**2
        train_scores.append(train_s)

        # Test the model
        test_pheno = gpm.data.loc[test_idx,gpm.phenotype_column]
        test_p = model.predict(X=test_X)
        test_s = pearson(test_p, test_pheno)**2
        test_scores.append(test_s)


    return train_scores, test_scores
