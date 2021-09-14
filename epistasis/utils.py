# __description__ = \
# """
# Useful generic functions for package.
# """

# XX DETRIOUS
# class DocstringMeta(abc.ABCMeta):
#     """
#     Metaclass that allows docstring 'inheritance'
#
#     Idea taken from this thread:
#     https://github.com/sphinx-doc/sphinx/issues/3140
#     """
#     def __new__(mcls, classname, bases, cls_dict):
#         # Create a new class as expected.
#         cls = abc.ABCMeta.__new__(mcls, classname, bases, cls_dict)
#
#         # Get order of inheritance
#         mro = cls.__mro__[1:]
#
#         # Iterate through items in class.
#         for name, member in cls_dict.iteritems():
#
#             # If the item does not have a docstring, add the base class docstring.
#             if not getattr(member, '__doc__'):
#                 for base in mro:
#                     try:
#                         member.__doc__ = getattr(base, name).__doc__
#                         break
#                     except AttributeError:
#                         pass
#         return cls

# def extract_mutations_from_genotypes(genotypes):
#     """
#     Given a list of genotypes, infer a mutations dictionary.
#     """
#     genotypes_grid = [list(g) for g in genotypes]
#     genotypes_array = np.array(genotypes_grid)
#     (n_genotypes, n_sites) = genotypes_array.shape
#     mutations = dict([(i, None) for i in range(n_sites)])
#     for i in range(n_sites):
#         unique = list(np.unique(genotypes_array[:, i]))
#         if len(unique) != 1:
#             mutations[i] = unique
#     return mutations

# -------------------------------------------------------
# Miscellaneous Python functions for random task
# -------------------------------------------------------

# from gpmap.utils import genotypes_to_binary
#
# from .mapping import encoding_to_sites
# from epistasis.matrix import get_model_matrix
#
#
# def genotypes_to_X(genotypes, gpm, order=1, model_type='global'):
#     """
#     Build an X matrix for a list of genotypes.
#
#     Parameters
#     ----------
#     genotypes : list-like
#         list of genotypes matching genotypes seen in gpm
#     gpm : gpmap.GenotypePhenotypeMap
#         genotype phenotype map that has an encoding table for converting the
#         genotypes to binary
#     order : int
#         order of epistasis for generating the X matrix.
#     model_type : str
#         should be 'global' or 'local', indicating what reference state to use
#         for the epistasis mode.
#
#     Returns
#     -------
#     X : np.ndarray
#         binary array indicating which epistatic coefficients should be applied
#         to which genotype.
#     """
#     # But a sites list.
#     sites = encoding_to_sites(
#         order,
#         gpm.encoding_table
#     )
#     binary = genotypes_to_binary(genotypes, gpm.encoding_table)
#
#     # X matrix
#     X = get_model_matrix(binary, sites, model_type=model_type)
#
#     return X
