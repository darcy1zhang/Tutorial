import numpy as np

def gsampleOne(node, scale, np_size):
    """
    Description:
        Generate a sampled identity matrix.

    Params:
        node: Location of the reconstruction Gabor functions.
        scale: Scale of the Gabor functions.
        np: Size of the reconstructed signal.

    Returns:
        Diagonal of the "sampled" Q1 term as a 1D vector.
    """
    dia = np.zeros(np_size)


    for j in range(np_size):
        tmp1 = (j - node) / scale
        tmp1 = np.exp(-(tmp1 * tmp1))
        dia[j] = np.sum(tmp1)

    return dia

if __name__ == "__main__":
    node = np.array([0, 1, 2, 3])
    scale = 0.5
    np_size = 5
    dia = gsampleOne(node, scale, np_size)
    print(dia)