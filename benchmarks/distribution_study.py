import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpmath import mp

from benchmarks.data_gen import (
    float128_to_takum,
    fp128_downcasting,
    generate_sparse_matrix,
    generate_sparse_array,
)
from benchmarks.comparison import compare
from benchmarks.experiments import generate_dynamic_sparse_matrix, generate_dynamic_sparse_array
from src.simd.simulator import RVVSimulator


def collect_error_distribution(scenario: str, bits: int, num_seeds: int = 30, size: int = 50):
    all_fp_errors = []
    all_takum_errors = []

    for seed in range(num_seeds):
        random.seed(seed)
        mp.dps = 34  # Mantém mp.prec = 113

        if scenario == "uniform":
            mat = generate_sparse_matrix(size, size)
            arr = generate_sparse_array(size)
        else:
            mat = generate_dynamic_sparse_matrix(size, size)
            arr = generate_dynamic_sparse_array(size)

        ref = mat * arr
        mat_tk = float128_to_takum(mat, bits)
        vec_tk = [row[0] for row in float128_to_takum(arr, bits)]
        mat_fp = fp128_downcasting(mat, bits)
        vec_fp = fp128_downcasting(arr, bits).flatten()

        sim_tk = RVVSimulator(arch="takum")
        sim_fp = RVVSimulator(arch="ieee754")

        res_tk = sim_tk.matvec(mat_tk, vec_tk, sew=bits)
        res_fp = sim_fp.matvec(mat_fp, vec_fp, sew=bits)

        err_fp, err_tk = compare(ref, res_tk, res_fp)

        all_fp_errors.extend([1.0 if (mp.isnan(e) or mp.isinf(e)) else float(e) for e in err_fp if e != 0])
        all_takum_errors.extend([1.0 if (mp.isnan(e) or mp.isinf(e)) else float(e) for e in err_tk if e != 0])

    return np.array(all_fp_errors), np.array(all_takum_errors)


def run_distribution_study(num_seeds: int = 30):
    os.makedirs("results", exist_ok=True)
    widths = [8, 16, 32]

    for scenario, label in [("uniform", "Distribuição Uniforme"), ("dynamic", "Estresse Dinâmico")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        print(f"\n=== Estatísticas de Distribuição ({label} | {num_seeds} rodadas) ===")

        for idx, bits in enumerate(widths):
            fp_errs, tk_errs = collect_error_distribution(scenario, bits, num_seeds=num_seeds)

            # Evita zeros exatos no eixo log da CDF
            fp_pos = np.clip(fp_errs, 1e-12, 1.0)
            tk_pos = np.clip(tk_errs, 1e-12, 1.0)

            print(
                f"[{bits} bits] IEEE 754 -> Mediana: {np.median(fp_pos):.2e} | Média: {np.mean(fp_pos):.2e} | DP: {np.std(fp_pos):.2e}"
            )
            print(
                f"[{bits} bits] Takum    -> Mediana: {np.median(tk_pos):.2e} | Média: {np.mean(tk_pos):.2e} | DP: {np.std(tk_pos):.2e}"
            )

            # Plota CDF empírica (quanto mais à esquerda, menor o erro)
            ax = axes[idx]
            ax.plot(np.sort(fp_pos), np.linspace(0, 1, len(fp_pos)), label="IEEE 754 (FP)", linewidth=2)
            ax.plot(np.sort(tk_pos), np.linspace(0, 1, len(tk_pos)), label="Takum", linewidth=2)
            ax.set_xscale("log")
            ax.set_title(f"{bits} bits")
            ax.set_xlabel("Erro Relativo (Escala Log)")
            if idx == 0:
                ax.set_ylabel("Probabilidade Acumulada (CDF)")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            ax.legend()

        plt.suptitle(f"Distribuição Acumulada do Erro Relativo - {label}")
        plt.tight_layout()
        plt.savefig(f"results/CDF_{scenario}.png", dpi=300)
        plt.close()