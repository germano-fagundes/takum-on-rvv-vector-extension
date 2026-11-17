import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp
from random import randint
import math

# Adiciona o diretório raiz ao path para os imports funcionarem se rodar da pasta benchmarks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.simd.simulator import RVVSimulator
from benchmarks.data_gen import float128_to_takum, fp128_downcasting, generate_sparse_matrix, generate_sparse_array
from benchmarks.comparison import compare

def generate_dynamic_sparse_matrix(m: int, n: int):
    total_non_zeros = math.ceil(m * n / 10)
    matrix = mp.zeros(m, n)
    for _ in range(total_non_zeros):
        scale = mp.power(10, randint(-6, 6))
        matrix[randint(0, m-1), randint(0, n-1)] = mp.rand() * scale
    return matrix

def generate_dynamic_sparse_array(n: int):
    array = mp.zeros(n, 1)
    for _ in range(math.ceil(n/10)):
        scale = mp.power(10, randint(-6, 6))
        array[randint(0, n-1), 0] = mp.rand() * scale
    return array

def run_trial(mat_orig, arr_orig, bits):
    ref_result = mat_orig * arr_orig

    mat_takum = float128_to_takum(mat_orig, bits)
    arr_takum = float128_to_takum(arr_orig, bits)
    takum_vec_1d = [row[0] for row in arr_takum]

    mat_fp = fp128_downcasting(mat_orig, bits)
    arr_fp = fp128_downcasting(arr_orig, bits).flatten()

    # Instancia os dois simuladores com a mesma configuração física de hardware
    sim_takum = RVVSimulator(arch="takum", vlen_bits=256, datapath_bits=64)
    sim_fp = RVVSimulator(arch="ieee754", vlen_bits=256, datapath_bits=64)

    # Executa y = A * x através das sequências de instruções de cada ISA
    takum_result = sim_takum.matvec(mat_takum, takum_vec_1d, sew=bits)
    fp_result = sim_fp.matvec(mat_fp, arr_fp, sew=bits)

    err_fp, err_takum = compare(ref_result, takum_result, fp_result)

    fp_valid = [1.0 if (mp.isnan(e) or mp.isinf(e)) else float(e) for e in err_fp if e != 0]
    tak_valid = [1.0 if (mp.isnan(e) or mp.isinf(e)) else float(e) for e in err_takum if e != 0]

    fp_mean = np.mean(fp_valid) if fp_valid else 0.0
    fp_max = np.max(fp_valid) if fp_valid else 0.0
    tak_mean = np.mean(tak_valid) if tak_valid else 0.0
    tak_max = np.max(tak_valid) if tak_valid else 0.0

    # Telemetria extraída diretamente dos simuladores para o artigo IEEE
    stats = {
        "takum_insts": sim_takum.instruction_count,
        "fp_insts": sim_fp.instruction_count,
        "inst_reduction_pct": (1 - sim_takum.instruction_count / sim_fp.instruction_count) * 100,
        "takum_cycles": sim_takum.cycle_count,
        "fp_cycles": sim_fp.cycle_count,
        "fp_exceptions": sim_fp.exception_events,
        "fp_fflags": dict(sim_fp.fflags),
    }

    return fp_mean, fp_max, tak_mean, tak_max, stats

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

    plt.tight_layout()
    directory, filename = title.split('/', 1) if '/' in title else ('', title)
    plt.suptitle(filename)
    safe_name = filename.replace(':', '').replace('/', '_').replace(' ', '_')
    out_path = os.path.join(directory, f"{safe_name}.png") if directory else f"{safe_name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()

def print_stats_table(exp_name: str, results: dict[int, dict]):
    print(f"\n=== Telemetria Arquitetural: {exp_name} ===")
    print(f"{'Bits':<6} | {'Insts (T/FP)':<15} | {'Redução':<9} | {'Ciclos (T/FP)':<15} | {'Falhas FP':<10} | {'Flags CSR (fflags)'}")
    print("-" * 88)
    for bits, st in results.items():
        inst_str = f"{st['takum_insts']}/{st['fp_insts']}"
        cyc_str = f"{st['takum_cycles']}/{st['fp_cycles']}"
        fail_str = f"{st['fp_exceptions']}/50"
        print(f"{bits:<6} | {inst_str:<15} | {st['inst_reduction_pct']:>7.2f}% | {cyc_str:<15} | {fail_str:<10} | {st['fp_fflags']}")

def run_experiments():
    os.makedirs("results", exist_ok=True)

    print("Iniciando Experimento 1: Distribuição Uniforme (8/16/32 bits)...")
    mat_uni = generate_sparse_matrix(50, 50)
    arr_uni = generate_sparse_array(50)

    fp_m8u, fp_p8u, tk_m8u, tk_p8u, st8_u = run_trial(mat_uni, arr_uni, 8)
    fp_m16u, fp_p16u, tk_m16u, tk_p16u, st16_u = run_trial(mat_uni, arr_uni, 16)
    fp_m32u, fp_p32u, tk_m32u, tk_p32u, st32_u = run_trial(mat_uni, arr_uni, 32)

    print_stats_table("Exp 1 - Uniforme", {8: st8_u, 16: st16_u, 32: st32_u})
    plot_results(
        'results/Exp 1: Distribuição Uniforme (8/16/32 bits)',
        ['8 bits', '16 bits', '32 bits'],
        [fp_m8u, fp_m16u, fp_m32u], [tk_m8u, tk_m16u, tk_m32u],
        [fp_p8u, fp_p16u, fp_p32u], [tk_p8u, tk_p16u, tk_p32u]
    )

    print("\nIniciando Experimento 2: Estresse Dinâmico (8/16/32 bits)...")
    mat_dyn = generate_dynamic_sparse_matrix(50, 50)
    arr_dyn = generate_dynamic_sparse_array(50)

    fp_m8d, fp_p8d, tk_m8d, tk_p8d, st8_d = run_trial(mat_dyn, arr_dyn, 8)
    fp_m16d, fp_p16d, tk_m16d, tk_p16d, st16_d = run_trial(mat_dyn, arr_dyn, 16)
    fp_m32d, fp_p32d, tk_m32d, tk_p32d, st32_d = run_trial(mat_dyn, arr_dyn, 32)

    print_stats_table("Exp 2 - Estresse Dinâmico", {8: st8_d, 16: st16_d, 32: st32_d})
    plot_results(
        'results/Exp 2: Estresse Dinâmico (8/16/32 bits)',
        ['8 bits', '16 bits', '32 bits'],
        [fp_m8d, fp_m16d, fp_m32d], [tk_m8d, tk_m16d, tk_m32d],
        [fp_p8d, fp_p16d, fp_p32d], [tk_p8d, tk_p16d, tk_p32d]
    )
