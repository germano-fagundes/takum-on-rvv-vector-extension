from mpmath import mp
from benchmarks.data_gen import float128_to_takum
from src.core.format import Takum
from src.simd.simulator import RVVSimulator


def run_ml_inference_kernel(dim: int = 32):
    """Simula uma camada de inferência ML (MatVec + Bias + LayerNorm + ReLU + Quantização T16->T8)."""
    W_mp = mp.matrix([[mp.rand() - 0.5 for _ in range(dim)] for _ in range(dim)])
    x_mp = mp.matrix([[mp.rand() for _ in range(1)] for _ in range(dim)])
    b_mp = [mp.rand() * 0.1 for _ in range(dim)]
    mean_mp = [mp.mpf(0.05) for _ in range(dim)]
    gamma_mp = [mp.mpf(1.2) for _ in range(dim)]

    # Referência Float128
    z_ref = [sum(W_mp[i, j] * x_mp[j, 0] for j in range(dim)) + b_mp[i] for i in range(dim)]
    norm_ref = [max(mp.mpf(0), (z_ref[i] - mean_mp[i]) * gamma_mp[i]) for i in range(dim)]

    # Execução via Zvtakum (T16 -> T8)
    sim = RVVSimulator(arch="takum", vlen_bits=256, datapath_bits=64)
    W_t16 = float128_to_takum(W_mp, 16)
    x_t16 = [row[0] for row in float128_to_takum(x_mp, 16)]
    b_t16 = [Takum.from_mpf(v, 16) for v in b_mp]
    mean_t16 = [Takum.from_mpf(v, 16) for v in mean_mp]
    gamma_t16 = [Takum.from_mpf(v, 16) for v in gamma_mp]

    # 1. vtdot.vv (projeção linear)
    z_t16 = [row[0] for row in sim.matvec(W_t16, x_t16, sew=16)]
    # 2. vtadd.vv (soma de bias)
    zb_t16 = sim.vtadd_vv(z_t16, b_t16)
    # 3. vtsub.vv e vtmul.vv (normalização afim)
    centered_t16 = sim.vtsub_vv(zb_t16, mean_t16)
    scaled_t16 = sim.vtmul_vv(centered_t16, gamma_t16)
    # 4. ReLU + vncvt.t.t (requantização T16 -> T8)
    relu_t16 = [Takum(0, 0, 0, 0, 16, 0) if val.signal == 1 else val for val in scaled_t16]
    out_t8 = sim.vncvt_tt(relu_t16, new_total_bits=8)

    errors = [
        abs(norm_ref[i] - out_t8[i].to_mpf()) / abs(norm_ref[i])
        for i in range(dim) if norm_ref[i] != 0
    ]

    print("\n=== Kernel de Inferência ML (Zvtakum T16 -> T8) ===")
    print(f"Instruções emitidas: {sim.instruction_breakdown}")
    print(f"Total de instruções: {sim.instruction_count} | Ciclos: {sim.cycle_count}")
    print(f"Erro relativo médio pós-requantização T8: {float(sum(errors)/len(errors)):.4f}")