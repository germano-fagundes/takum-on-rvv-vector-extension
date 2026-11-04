from mpmath import mp, matrix
import numpy as np

from src.core.format import Takum
mp.prec = 113 # type: ignore

def compare(reference_result: matrix, takum_result: list[list[Takum]], fp_result: np.ndarray):
    ref_1d = [reference_result[i, 0] for i in range(reference_result.rows)]
    
    fp_1d = np.array(fp_result).flatten().tolist()
    upcasted_fp = [mp.mpf(val) for val in fp_1d]
    
    takum_1d = [row[0] for row in takum_result]
    upcasted_takum = [t.to_mpf() for t in takum_1d]

    relative_error_fp = [
        [abs(reference_result[row, col] - upcasted_fp[row][col]) /
         abs(reference_result[row, col]) if reference_result[row, col] != 0
         else mp.mpf(0.0) for col in range(reference_result.cols)]
         for row in range(reference_result.rows)]
    
    relative_error_takum = [
        [abs(reference_result[row, col] - upcasted_takum[row][col]) /
         abs(reference_result[row, col]) if reference_result[row, col] != 0
         else mp.mpf(0.0) for col in range(reference_result.cols)]
         for row in range(reference_result.rows)]
    
    return relative_error_fp, relative_error_takum