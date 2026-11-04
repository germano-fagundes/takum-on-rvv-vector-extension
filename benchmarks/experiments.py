import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from random import randint
import math

# Adiciona o diretório raiz ao path para os imports funcionarem se rodar da pasta benchmarks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from benchmarks.data_gen import float128_to_takum, fp128_downcasting, generate_sparse_matrix, generate_sparse_array
from benchmarks.comparison import compare
from src.simd.vector import vtdot

def generate_dynamic_sparse_matrix(m: int, n: int):
    total_non_zeros = math.ceil(m * n / 10)
    matrix = mp.zeros(m, n)
    for _ in range(total_non_zeros):
        scale = mp.power(10, randint(-4, 4))
        matrix[randint(0, m-1), randint(0, n-1)] = mp.rand() * scale
    return matrix

def generate_dynamic_sparse_array(n: int):
    array = mp.zeros(n, 1)
    for _ in range(math.ceil(n/10)):
        scale = mp.power(10, randint(-4, 4))
        array[randint(0, n-1), 0] = mp.rand() * scale
    return array

def run_trial(mat_orig, arr_orig, bits):
    ref_result = mat_orig * arr_orig
    
    mat_takum = float128_to_takum(mat_orig, bits)
    arr_takum = float128_to_takum(arr_orig, bits)
    
    mat_fp = fp128_downcasting(mat_orig, bits)
    arr_fp = fp128_downcasting(arr_orig, bits)
    
    takum_vec_1d = [row[0] for row in arr_takum]
    takum_result = [[vtdot(row, takum_vec_1d)] for row in mat_takum]
    
    fp_result = np.dot(mat_fp, arr_fp)
    
    err_fp, err_takum = compare(ref_result, takum_result, fp_result)
    
    # Extração de estatísticas (filtrando zeros perfeitos da matriz esparsa)
    fp_valid = [float(e) for e in err_fp if e != 0 and not mp.isnan(e) and not mp.isinf(e)]
    tak_valid = [float(e) for e in err_takum if e != 0]
    
    fp_mean = np.mean(fp_valid) if fp_valid else 0.0
    fp_max = np.max(fp_valid) if fp_valid else 0.0
    
    tak_mean = np.mean(tak_valid) if tak_valid else 0.0
    tak_max = np.max(tak_valid) if tak_valid else 0.0
    
    fp_failures = sum(1 for e in err_fp if mp.isnan(e) or mp.isinf(e))
    
    return fp_mean, fp_max, tak_mean, tak_max, fp_failures

def plot_results(title, labels, fp_m, tak_m, fp_p, tak_p):
    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(x - width/2, fp_m, width, label='IEEE 754 (FP)')
    ax1.bar(x + width/2, tak_m, width, label='Takum')
    ax1.set_ylabel('Erro Relativo Médio')
    ax1.set_title('Erro Médio (Escala Log)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_yscale('log')
    ax1.legend()

    ax2.bar(x - width/2, fp_p, width, label='IEEE 754 (FP)')
    ax2.bar(x + width/2, tak_p, width, label='Takum')
    ax2.set_ylabel('Erro Relativo Máximo')
    ax2.set_title('Pico de Erro (Escala Log)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_yscale('log')
    ax2.legend()

    plt.suptitle(title)
    plt.tight_layout()
    safe_title = title.replace(':', '').replace('/', '_').replace(' ', '_')
    plt.savefig(f"{safe_title}.png", dpi=300)
    plt.close()

def run_experiments():
    print("Iniciando Experimento 1: Uniforme...")
    mat_uni = generate_sparse_matrix(50, 50)
    arr_uni = generate_sparse_array(50)
    
    fp_m16, fp_p16, tk_m16, tk_p16, _ = run_trial(mat_uni, arr_uni, 16)
    fp_m32, fp_p32, tk_m32, tk_p32, _ = run_trial(mat_uni, arr_uni, 32)
    
    plot_results('Exp 1: Distribuição Uniforme (16/32 bits)', ['16 bits', '32 bits'], 
                 [fp_m16, fp_m32], [tk_m16, tk_m32],
                 [fp_p16, fp_p32], [tk_p16, tk_p32])
    
    print("\nIniciando Experimento 2: Estresse Dinâmico...")
    mat_dyn = generate_dynamic_sparse_matrix(50, 50)
    arr_dyn = generate_dynamic_sparse_array(50)
    
    fp_m8, fp_p8, tk_m8, tk_p8, f8 = run_trial(mat_dyn, arr_dyn, 8)
    fp_m16d, fp_p16d, tk_m16d, tk_p16d, f16 = run_trial(mat_dyn, arr_dyn, 16)
    
    print(f"-> FP8 sofreu Overflow/Underflow em {f8} valores (NaN/Inf)")
    print(f"-> FP16 sofreu Overflow/Underflow em {f16} valores (NaN/Inf)")
    
    plot_results('Exp 2: Estresse Dinâmico (8/16 bits)', ['8 bits', '16 bits'], 
                 [fp_m8, fp_m16d], [tk_m8, tk_m16d],
                 [fp_p8, fp_p16d], [tk_p8, tk_p16d])

if __name__ == '__main__':
    run_experiments()
