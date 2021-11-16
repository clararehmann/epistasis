
import numpy as np

def build_model_matrix(encoding_vectors, sites):
    """
    Build model matrix.

    This can be made faster by cython.
    """
    # get dimensions of matrix
    sites = [list(s) for s in sites]
    n, m = len(encoding_vectors), len(sites)

    matrix = np.ones((n,m), dtype=int)

    # Interate over rows
    for i in range(n):
        vec = encoding_vectors[i]

        # Iterate over cols
        for j in range(m):
            # Get sites in this coefficient column
            players = sites[j]

            # Multiply these elements
            matrix[i,j] = np.prod(vec[players])

    return matrix
