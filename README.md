# Extensão Vetorial RISC-V com Aritmética Takum (`Zvtakum`)

Simulador funcional em nível de instrução e ciclo para avaliação do formato numérico de precisão afunilada **Takum** (`T8`, `T16` e `T32`) integrado a um subconjunto da Extensão Vetorial RISC-V (**RVV**), comparado ao padrão **IEEE 754** (`FP8 E4M3FN`, `FP16` e `FP32`).

Projeto desenvolvido para a disciplina de **Arquitetura e Organização de Computadores (AOC)** — Departamento de Engenharia Eletrônica (DELT / UFMG).

---

## Arquitetura do Repositório

```text
takum-on-rvv-vector-extension/
├── src/
│   ├── core/
│   │   └── format.py          # Codificação/decodificação Takum (S, D, R, C, F) e conversão mpf
│   ├── alu/
│   │   └── scalar.py          # Operações aritméticas escalares (add, sub, mul) com RNTE
│   └── simd/
│       ├── vector.py          # Instruções vetoriais Zvtakum (vtadd, vtsub, vtmul, vtdot, vncvt)
│       └── simulator.py       # Simulador RVV (strip-mining, contagem de ciclos e CSRs/fflags)
├── benchmarks/
│   ├── data_gen.py            # Geração de matrizes/vetores esparsos e downcasting Float128
│   ├── comparison.py          # Cálculo de erro relativo contra referência Float128 (mpmath)
│   ├── evaluation.py          # Rotina base de avaliação matriz-vetor
│   ├── experiments.py         # Experimentos comparativos (Uniforme vs. Estresse Dinâmico)
│   ├── distribution_study.py  # Análise estatística multi-semente e curvas CDF
│   └── ml_kernel.py           # Kernel de inferência ML quantizada (MatVec + Bias + Norm + T16->T8)
├── tests/
│   ├── test_format.py         # Testes unitários do formato Takum e arredondamento RNTE
│   └── test_alu_and_simd.py   # Testes da ALU escalar, instruções Zvtakum e simulador RVV
├── results/                   # Gráficos PNG gerados pelos benchmarks
└── main.py                    # Ponto de entrada CLI unificado
```

---

## Instruções Suportadas (`Zvtakum`)

| Instrução Proposta | Operação | Equivalente RVV Padrão |
| :--- | :--- | :--- |
| `vtadd.vv` | Adição vetorial elemento a elemento em Takum | `vfadd.vv` |
| `vtsub.vv` | Subtração vetorial elemento a elemento em Takum | `vfsub.vv` |
| `vtmul.vv` | Multiplicação vetorial elemento a elemento em Takum | `vfmul.vv` |
| `vtdot.vv` | Produto escalar vetorial com acúmulo exato (*Quire*) | `vdot.*` / `vfdot.*` |
| `vncvt.t.t` | Conversão com redução de largura (ex.: `T16` $\rightarrow$ `T8`) | `vfncvt.*` |

---

## Requisitos e Instalação

Requer Python 3.10+ e as bibliotecas listadas abaixo:

```bash
pip install numpy matplotlib mpmath ml-dtypes
```

---

## Execução

### 1. Executar todos os experimentos e demonstrações
Gera os gráficos comparativos em `results/` e exibe no terminal as tabelas de contagem de instruções, ciclos e flags CSR (`fflags`):

```bash
python main.py --mode all
```

### 2. Modos individuais via CLI
* **Decodificador explícito (`S, D, R, C, F`):**
  ```bash
  python main.py --mode demo
  ```
* **Multiplicação matriz-vetor esparsa ($y = A \cdot x$):**
  ```bash
  python main.py --mode exp
  ```
* **Distribuição estatística acumulada (CDF) em 30 sementes:**
  ```bash
  python main.py --mode dist --seeds 30
  ```
* **Kernel de inferência de Machine Learning (`T16` $\rightarrow$ `T8`):**
  ```bash
  python main.py --mode ml
  ```

### 3. Executar a suíte de testes unitários
```bash
python -m unittest discover -s tests -v
```