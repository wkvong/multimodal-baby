import numpy as np
import scipy.stats

def cosine_matrix(A):
    """The cosine similarity matrix of A.
    A: np.ndarray of shape [N, D] containing N vectors of dimension D
    return: np.ndarray of shape [N, N] where (i, j)-th value is the cosine of
        A[i] and A[j]
    """
    normalized_A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return normalized_A @ normalized_A.T

def cosine_dissim_matrix(A):
    """Compute dissimilarity by (1 - cos(A[i], A[j])) / 2
    """
    return (1. - cosine_matrix(A)) / 2.

def strict_upper_tri_items(A):
    """Return all items in the strict upper triauglar matrix of A in a 1-D
    np.ndarray
    A: 2-D np.ndarray
    return: 1-D np.ndarray
    """
    return A[np.triu_indices(A.shape[0], k=1, m=A.shape[1])]

def rsa_of_dissim_matrices(A, B):
    """Representational Similarity Analysis (RSA; Kriegeskorte et al. 2008)
    A, B: np.ndarray of shape [N, N]
    """
    spearmanr_result = scipy.stats.spearmanr(
        strict_upper_tri_items(A),
        strict_upper_tri_items(B),
    )
    return spearmanr_result.correlation

def rsa(A, B, dissim_matrix_fn=cosine_dissim_matrix):
    """Representational Similarity Analysis (RSA; Kriegeskorte et al. 2008)
    A: np.ndarray of shape [N, D_A]
    B: np.ndarray of shape [N, D_B]
    return: float
    """
    return rsa_sim_matrix(dissim_matrix_fn(A), dissim_matrix_fn(B))
