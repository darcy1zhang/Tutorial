import numpy as np
from scipy.linalg import svd

def SVD(a):
    """
    Description:
        Computes the singular value decomposition of a matrix.

    Args:
        a: Input matrix.

    Returns:
        A structure containing the 3 matrices of the singular value decomposition of the input.
    """

    U, S, V = svd(a, full_matrices=True)

    return U, S, V


if __name__ == "__main__":
    input_matrix = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])


    U, S, V = SVD(input_matrix)


    print("U Matrix:")
    print(U)
    print("\nS Matrix (Singular Values):")
    print(S)
    print("\nV Matrix (Conjugate transpose of V):")
    print(V)
