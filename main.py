import argparse
from mpmath import mp

from src.core.format import Takum
from benchmarks.experiments import run_experiments
from benchmarks.distribution_study import run_distribution_study
from benchmarks.ml_kernel import run_ml_inference_kernel


def demo_decoder():
    print("=== Demonstração do Decodificador Takum (S, D, R, C, F) ===")
    samples = [mp.mpf("0.75"), mp.mpf("-12.5"), mp.mpf("1e-5"), mp.mpf("1e5")]
    for val in samples:
        for bits in (8, 16, 32):
            t = Takum.from_mpf(val, bits)
            fields = t.extract_fields()
            print(f"Valor: {float(val):>10.4g} | T{bits:<2} -> {fields} | Reconstruído: {float(t.to_mpf()):.6g}")
        print("-" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Simulador Funcional RVV com Extensão Aritmética Takum (Zvtakum)"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "demo", "exp", "dist", "ml"],
        default="all",
        help="Modo de execução: demo (decodificador), exp (barras), dist (CDFs), ml (kernel ML) ou all.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Número de rodadas aleatórias para o estudo de distribuição (padrão: 30).",
    )
    args = parser.parse_args()

    if args.mode in ("all", "demo"):
        demo_decoder()
    if args.mode in ("all", "exp"):
        run_experiments()
    if args.mode in ("all", "dist"):
        run_distribution_study(num_seeds=args.seeds)
    if args.mode in ("all", "ml"):
        run_ml_inference_kernel()


if __name__ == "__main__":
    main()