from mpmath import mp, matrix
import numpy as np
import ml_dtypes

from src.core.format import Takum

mp.prec = 113  # type: ignore
from math import ceil
from random import randint

def generate_sparse_matrix(m: int, n: int):
    if (m < 2 and n < 2):
        raise ValueError('m (rows) and n (columns) cannot be both less than 2.')
    if m < 1 or n < 1:
        raise ValueError('Neither m (rows) nor n (columns) can be less than 1.')

    total_non_zeros = ceil(m * n / 10)
    matrix = mp.zeros(m, n)

    for _ in range(total_non_zeros):
        matrix[randint(0,m-1), randint(0,n-1)] = mp.rand()

    return matrix


def generate_sparse_array(n: int):
    if n < 2:
        raise ValueError('n (length) must be at least 2.')

    array = mp.zeros(n, 1)

    for _ in range(ceil(n/10)):
        array[randint(0,n-1), 0] = mp.rand()

    return array


def float128_to_takum(matrix: matrix, takum_bits: int):
    takum_matrix = [[Takum.from_mpf(matrix[i, j], takum_bits) for j in range(matrix.cols)] for i in range(matrix.rows)]
    return takum_matrix

def fp128_downcasting(matrix: matrix, fp_bits: int):
    if fp_bits != 8 and fp_bits != 16 and fp_bits != 32:
        raise ValueError('fp_bits must be either 8, 16 or 32.')

    float64_matrix = np.array(matrix.tolist(), dtype=float)

    if fp_bits == 32:
        return float64_matrix.astype(np.float32)
    if fp_bits == 16:
        return float64_matrix.astype(np.float16)
    if fp_bits == 8:
        return float64_matrix.astype(ml_dtypes.float8_e4m3fn)