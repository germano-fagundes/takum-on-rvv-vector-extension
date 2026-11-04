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

    relative_error_fp = []
    relative_error_takum = []

    for i in range(len(ref_1d)):
        true_val = ref_1d[i]
        
        if true_val != 0:
            err_fp = abs(true_val - upcasted_fp[i]) / abs(true_val)
            err_tak = abs(true_val - upcasted_takum[i]) / abs(true_val)
        else:
            err_fp = mp.mpf(0.0)
            err_tak = mp.mpf(0.0)
            
        relative_error_fp.append(err_fp)
        relative_error_takum.append(err_tak)
        
    return relative_error_fp, relative_error_takum