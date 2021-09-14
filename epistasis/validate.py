__description__ = \
"""
Functions for validating model fits of epistasis models.
"""
__author__ = "Zach Sailer"


from .stats import pearson
import gpmap
import numpy as np

def split_data(data, idx=None, nobs=None, fraction=None):
    """
    Split DataFrame into two sets, a training and a test set.

    Parameters
    ----------
    data : pandas.DataFrame
        full dataset to split.

    idx : list
        List of indices to include in training set

    nobs : int
        number of observations in training. If nobs is given, fraction is
        ignored.

    fraction : float
        fraction in training set.

    Returns
    -------
    train_set : pandas.DataFrame
        training set.

    test_set : pandas.DataFrame
        test set.
    """
    if idx:

        train_idx = set(idx)
        total_idx = set(data.index)
        test_idx = total_idx.difference(train_idx)

        train_idx = sorted(list(train_idx))
        test_idx = sorted(list(test_idx))

    elif nobs:
        length = len(data)

        # Shuffle the indices
        index = np.arange(0, length, dtype=int)
        np.random.shuffle(index)

        train_idx = index[:nobs]
        test_idx = index[nobs:]

    elif fraction:

        if fraction is None:
            raise Exception("nobs or fraction must be given")

        elif 0 < fraction > 1.0:
            raise Exception("fraction is invalid.")

        else:
            length = len(data)
            nobs = int(length * fraction)

        # Shuffle the indices
        index = np.arange(0, length, dtype=int)
        np.random.shuffle(index)

        train_idx = index[:nobs]
        test_idx = index[nobs:]

    # Split data.
    train_set = data.iloc[train_idx]
    test_set = data.iloc[test_idx]

    return train_set, test_set


def split_gpm(gpm, idx=None, nobs=None, fraction=None):
    """
    Split GenotypePhenotypeMap into two sets, a training and a test set.

    Parameters
    ----------
    gpm : gpmap.GenotypePhenotypeMap
        GenotypePhenotypeMap with data to split

    idx : list
        List of indices to include in training set

    nobs : int
        number of observations in training.

    fraction : float
        fraction in training set.

    Returns
    -------
    train_gpm : GenotypePhenotypeMap
        training set.

    test_gpm : GenotypePhenotypeMap
        test set.
    """
    train, test = split_data(gpm.data, idx=idx, nobs=nobs, fraction=fraction)


    # Create two new GenotypePhenotypeMaps given test and train pandas df
    train_gpm = gpmap.read_dataframe(train,
                                     wildtype=gpm.wildtype,
                                     mutations=gpm.mutations,
                                     site_labels=gpm.site_labels)

    test_gpm =  gpmap.read_dataframe(test,
                                     wildtype=gpm.wildtype,
                                     mutations=gpm.mutations,
                                     site_labels=gpm.site_labels)

    return train_gpm, test_gpm


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
