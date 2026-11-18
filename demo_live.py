import time
import warnings
import numpy as np
import ml_dtypes
from mpmath import mp

# Precisão quádrupla (IEEE 754 binary128) como gabarito
mp.prec = 113  # type: ignore

from src.core.format import Takum
from src.simd.simulator import RVVSimulator

# Cores ANSI para destaque visual no terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def step_print(msg: str, delay: float = 0.30):
    print(msg)
    time.sleep(delay)


def to_takum_vec(values: list, bits: int) -> list[Takum]:
    return [Takum.from_mpf(mp.mpf(str(v)), bits) for v in values]


def to_ieee_vec(values: list, bits: int) -> np.ndarray:
    arr64 = np.array([float(v) for v in values], dtype=float)
    if bits == 8:
        return arr64.astype(ml_dtypes.float8_e4m3fn)
    if bits == 16:
        return arr64.astype(np.float16)
    if bits == 32:
        return arr64.astype(np.float32)
    raise ValueError("bits deve ser 8, 16 ou 32.")


def run_live_dot_demo(title: str, vec_a: list, vec_b: list, bits: int = 16):
    print(f"\n{BOLD}{CYAN}{'=' * 82}{RESET}")
    print(f"{BOLD}{CYAN}  {title}  |  Largura: {bits} bits (T{bits} vs FP{bits}){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 82}{RESET}")
    time.sleep(0.4)

    # 1. Gabarito Exato Float128
    ref_a = [mp.mpf(str(x)) for x in vec_a]
    ref_b = [mp.mpf(str(x)) for x in vec_b]
    ref_dot = sum((a * b for a, b in zip(ref_a, ref_b)), mp.mpf(0))

    step_print(f"\n{BOLD}[1] Vetores de Entrada (AVL = {len(vec_a)}):{RESET}")
    step_print(f"    A = {vec_a}", 0.2)
    step_print(f"    B = {vec_b}", 0.2)
    step_print(f"    Gabarito Exato (Float128 / 113-bit): {GREEN}{mp.nstr(ref_dot, 8)}{RESET}", 0.35)

    # 2. Codificação Takum Bit a Bit via extract_fields()
    t_a = to_takum_vec(vec_a, bits)
    t_b = to_takum_vec(vec_b, bits)

    step_print(f"\n{BOLD}[2] Codificação Takum T{bits} do Vetor A (extract_fields):{RESET}")
    for orig, tk in zip(vec_a, t_a):
        f = tk.extract_fields()
        step_print(
            f"    {str(orig):>8} -> S={f['S']} D={f['D']} R='{f['R']}' C='{f['C']}' F='{f['F']}' "
            f"| Reconstruído: {mp.nstr(tk.to_mpf(), 6)}",
            0.22,
        )

    # 3. Execução no RVVSimulator (Zvtakum vs IEEE 754)
    step_print(f"\n{BOLD}[3] Simulação Arquitetural RVV (VLEN=256b, Datapath=64b):{RESET}")

    # --- Zvtakum ---
    sim_tk = RVVSimulator(arch="takum", vlen_bits=256, datapath_bits=64)
    t_res = sim_tk.dot_product_takum(t_a, t_b)
    t_val = t_res.to_mpf()
    t_err = abs(ref_dot - t_val) / abs(ref_dot) if ref_dot != 0 else mp.mpf(0)

    step_print(f"    {GREEN}>>> Zvtakum (T{bits}):{RESET}", 0.2)
    step_print(f"        Breakdown ISA : {sim_tk.instruction_breakdown}", 0.25)
    step_print(f"        Desempenho    : {sim_tk.instruction_count} instruções | {sim_tk.cycle_count} ciclos", 0.25)
    step_print(f"        Estado CSR    : 0 bits (Stateless / Zero Exceptions)", 0.25)
    step_print(
        f"        Resultado     : {mp.nstr(t_val, 8)}  |  Erro Relativo: {GREEN}{mp.nstr(t_err, 3)}{RESET}",
        0.4,
    )

    # --- IEEE 754 ---
    sim_fp = RVVSimulator(arch="ieee754", vlen_bits=256, datapath_bits=64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        fp_a = to_ieee_vec(vec_a, bits)
        fp_b = to_ieee_vec(vec_b, bits)
        fp_res = sim_fp.dot_product_ieee(fp_a, fp_b, sew=bits)

    fp_mpf = mp.mpf(float(fp_res))
    has_exc = bool(mp.isnan(fp_mpf) or mp.isinf(fp_mpf) or sim_fp.exception_events > 0)
    fp_err = 1.0 if has_exc else (float(abs(ref_dot - fp_mpf) / abs(ref_dot)) if ref_dot != 0 else 0.0)

    color_fp = RED if has_exc else YELLOW
    step_print(f"\n    {color_fp}>>> IEEE 754 (FP{bits}):{RESET}", 0.2)
    step_print(f"        Breakdown ISA : {sim_fp.instruction_breakdown}", 0.25)
    step_print(f"        Desempenho    : {sim_fp.instruction_count} instruções | {sim_fp.cycle_count} ciclos", 0.25)
    step_print(f"        Flags CSR     : {color_fp}{sim_fp.fflags}{RESET}", 0.25)
    step_print(
        f"        Resultado     : {fp_res}  |  Erro Relativo: {color_fp}{fp_err:.2e}{RESET}",
        0.45,
    )


if __name__ == "__main__":
    print(f"{BOLD}=== DEMONSTRAÇÃO FUNCIONAL: EXTENSÃO VETORIAL RISC-V Zvtakum ==={RESET}")
    input("\nPressione [ENTER] para iniciar o Cenário 1 (Valores Nominais em 16 bits)...")

    run_live_dot_demo(
        title="CENÁRIO 1: Precisão Próxima à Unidade",
        vec_a=[0.75, -12.5, 1.5, 0.25],
        vec_b=[2.0, 0.25, -4.0, 8.0],
        bits=16,
    )

    input("\nPressione [ENTER] para iniciar o Cenário 2 (Estresse Dinâmico em 16 bits)...")

    run_live_dot_demo(
        title="CENÁRIO 2: Estresse de Alcance Dinâmico (10^-5 a 10^5)",
        vec_a=[1e5, 1e-5, 1.0, 5e4],
        vec_b=[2.0, 1e-2, 3.5, 2.0],
        bits=16,
    )

    input("\nPressione [ENTER] para iniciar o Cenário 3 (Economia de Instruções em 8 bits)...")

    run_live_dot_demo(
        title="CENÁRIO 3: Baixa Precisão (T8 vs FP8 E4M3FN)",
        vec_a=[0.75, 1.25, -2.0, 0.25],
        vec_b=[1.0, 2.0, 0.25, 4.0],
        bits=8,
    )

    print(f"\n{BOLD}{GREEN}Demonstração concluída com sucesso!{RESET}")