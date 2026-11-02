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