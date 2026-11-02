from data_gen import float128_to_takum, fp128_downcasting, generate_sparse_array, generate_sparse_matrix
from src.simd.vector import vtdot
import numpy as np

def run_benchmark(m: int, n: int, total_bits: int):
    original_matrix = generate_sparse_matrix(m, n)
    original_array = generate_sparse_array(n)

    reference_result = original_matrix * original_array

    takum_matrix = float128_to_takum(original_matrix, total_bits)
    takum_array = float128_to_takum(original_array, total_bits)
    takum_array = [row[0] for row in takum_array]

    fp_matrix = fp128_downcasting(original_matrix, total_bits)
    fp_array = fp128_downcasting(original_array, total_bits)

    takum_result = [[vtdot(row, takum_array)] for row in takum_matrix]
    fp_result = np.dot(fp_matrix, fp_array)

    return reference_result, takum_result, fp_result