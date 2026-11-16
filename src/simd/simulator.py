import math
import numpy as np
from src.core.format import Takum
from src.simd.vector import vtadd, vtsub, vtmul, vtdot, vncvt
from typing import overload

class RVVSimulator:
    def __init__(self, arch: str = "takum", vlen_bits: int = 256, datapath_bits: int = 64):
        if arch not in ("takum", "ieee754"):
            raise ValueError("arch must be 'takum' or 'ieee754'")

        self.arch = arch
        self.vlen_bits = vlen_bits          # Largura de cada registrador vetorial (ex: 256 bits)
        self.datapath_bits = datapath_bits  # Largura física da ALU vetorial (ex: 64 bits)

        # Registradores de Controle e Status (CSRs)
        self.vl = 0                         # Vector Length (elementos ativos no bloco atual)
        self.sew = 32                       # Selected Element Width (8, 16 ou 32 bits)
        self.vtype: dict = {}

        # CSRs de ponto flutuante (exigidos apenas pelo padrão IEEE 754)
        self.frm = 0                        # Rounding Mode (3 bits em fcsr)
        self.fflags = {"NV": 0, "DZ": 0, "OF": 0, "UF": 0, "NX": 0}  # 5 bits de flags

        # Contadores de telemetria arquitetural
        self.instruction_count = 0
        self.instruction_breakdown: dict[str, int] = {}
        self.cycle_count = 0
        self.exception_events = 0

    def reset_metrics(self):
        self.instruction_count = 0
        self.instruction_breakdown.clear()
        self.cycle_count = 0
        self.exception_events = 0
        for k in self.fflags:
            self.fflags[k] = 0

    def _record_inst(self, mnemonic: str, extra_cycles: int = 0):
        self.instruction_count += 1
        self.instruction_breakdown[mnemonic] = self.instruction_breakdown.get(mnemonic, 0) + 1

        # O número de lanes paralelas escala inversamente com a largura do operando (SEW)
        lanes = max(1, self.datapath_bits // self.sew)
        compute_cycles = math.ceil(self.vl / lanes) if self.vl > 0 else 1
        self.cycle_count += compute_cycles + extra_cycles

    def _record_mem(self, mnemonic: str):
        self.instruction_count += 1
        self.instruction_breakdown[mnemonic] = self.instruction_breakdown.get(mnemonic, 0) + 1

        # Largura de barramento de memória igual à largura do datapath
        lanes = max(1, self.datapath_bits // self.sew)
        mem_cycles = math.ceil(self.vl / lanes) if self.vl > 0 else 1
        self.cycle_count += mem_cycles

    def vsetvli(self, avl: int, sew: int, fmt_variant: str = "standard") -> int:
        """Configura vl e vtype para o bloco de strip-mining atual."""
        self.sew = sew
        vlmax = self.vlen_bits // sew
        self.vl = min(avl, vlmax)

        if self.arch == "takum":
            # Decodificação unificada de até 12 bits: vtype precisa apenas de SEW
            self.vtype = {"sew": sew}
        else:
            # IEEE 754 exige distinguir variantes incompatíveis (ex: E4M3 vs E5M2, FP16 vs BF16)
            self.vtype = {"sew": sew, "variant": fmt_variant, "frm": self.frm}

        self.instruction_count += 1
        self.instruction_breakdown["vsetvli"] = self.instruction_breakdown.get("vsetvli", 0) + 1
        self.cycle_count += 1
        return self.vl

    # -------------------------------------------------------------------------
    # Instruções vetoriais elementares (Tabela 1: Zvtakum vs. RVV tradicional)
    # -------------------------------------------------------------------------
    def vtadd_vv(self, a: list[Takum], b: list[Takum]) -> list[Takum]:
        self._record_inst("vtadd.vv" if self.arch == "takum" else "vfadd.vv")
        return vtadd(a, b)

    def vtsub_vv(self, a: list[Takum], b: list[Takum]) -> list[Takum]:
        self._record_inst("vtsub.vv" if self.arch == "takum" else "vfsub.vv")
        return vtsub(a, b)

    def vtmul_vv(self, a: list[Takum], b: list[Takum]) -> list[Takum]:
        self._record_inst("vtmul.vv" if self.arch == "takum" else "vfmul.vv")
        return vtmul(a, b)

    def vncvt_tt(self, vec: list[Takum], new_total_bits: int) -> list[Takum]:
        self._record_inst("vncvt.t.t" if self.arch == "takum" else "vfncvt.f.f.w")
        return [vncvt(item, new_total_bits) for item in vec]

    # -------------------------------------------------------------------------
    # Sequências reais de Produto Escalar (vtdot.vv vs. vfdot/vfwmacc)
    # -------------------------------------------------------------------------
    def dot_product_takum(self, row: list[Takum], vec: list[Takum]) -> Takum:
        """Executa o produto escalar de uma linha via strip-mining em Zvtakum."""
        avl = len(row)
        sew = row[0].total_bits
        offset = 0

        while offset < avl:
            vl = self.vsetvli(avl - offset, sew)
            self._record_mem(f"vle{sew}.v")   # Carga de trecho da linha A[i, :]
            self._record_mem(f"vle{sew}.v")   # Carga de trecho do vetor x
            self._record_inst("vtdot.vv")     # MAC exato direto no registrador Quire
            offset += vl

        # O Quire acumula todas as fatias sem arredondamento intermediário e converte uma única vez
        return vtdot(row, vec)

    def dot_product_ieee(self, row: np.ndarray, vec: np.ndarray, sew: int):
        """Executa a sequência equivalente RVV IEEE 754 (FP8, FP16 ou FP32)."""
        avl = len(row)
        variant = "e4m3fn" if sew == 8 else ("binary16" if sew == 16 else "binary32")
        offset = 0

        while offset < avl:
            vl = self.vsetvli(avl - offset, sew, fmt_variant=variant)
            self._record_mem(f"vle{sew}.v")   # Carga de trecho da linha A[i, :]
            self._record_mem(f"vle{sew}.v")   # Carga de trecho do vetor x

            if sew == 8:
                # FP8 não possui FMA nativo direto sem promoção: exige widening explícito
                self._record_inst("vfwcvt.f.f.v")   # Promove fatia de A (8b -> 16b)
                self._record_inst("vfwcvt.f.f.v")   # Promove fatia de x (8b -> 16b)
                self._record_inst("vfwmacc.vv")     # Multiplica e acumula em 32b
            elif sew == 16:
                # FP16 utiliza widening MAC para reduzir risco de overflow no acumulador
                self._record_inst("vfwmacc.vv")     # Multiplica 16b e acumula em 32b
            else:
                # FP32 acumula na própria largura
                self._record_inst("vfmacc.vv")

            offset += vl

        # Redução final do vetor acumulador e narrowing de volta para o formato de destino
        if sew < 32:
            self._record_inst("vfncvt.f.f.w")
            if sew == 8:
                self._record_inst("vfncvt.f.f.w")

        result = np.dot(row, vec)
        self._update_ieee_fflags(row, vec, result)
        return result

    def _update_ieee_fflags(self, row: np.ndarray, vec: np.ndarray, result):
        """Atualiza as flags de exceção (fflags) no CSR fcsr do IEEE 754."""
        val = float(result)
        if np.isnan(val):
            self.fflags["NV"] += 1
            self.exception_events += 1
        elif np.isinf(val):
            self.fflags["OF"] += 1
            self.fflags["NX"] += 1
            self.exception_events += 1
        elif val == 0.0 and np.any((row != 0) & (vec != 0)):
            self.fflags["UF"] += 1
            self.fflags["NX"] += 1
            self.exception_events += 1

    # -------------------------------------------------------------------------
    # Operação de nível superior: Multiplicação Matriz-Vetor (y = A * x)
    # -------------------------------------------------------------------------
    @overload
    def matvec(self, matrix: list[list[Takum]], vector: list[Takum], sew: int) -> list[list[Takum]]: ...

    @overload
    def matvec(self, matrix: np.ndarray, vector: np.ndarray, sew: int) -> np.ndarray: ...
    
    def matvec(
        self,
        matrix: list[list[Takum]] | np.ndarray,
        vector: list[Takum] | np.ndarray,
        sew: int,
    ) -> list[list[Takum]] | np.ndarray:
        if self.arch == "takum":
            assert isinstance(matrix, list) and isinstance(vector, list)
            return [[self.dot_product_takum(row, vector)] for row in matrix]
        else:
            assert isinstance(matrix, np.ndarray) and isinstance(vector, np.ndarray)
            results = [self.dot_product_ieee(matrix[i], vector, sew) for i in range(matrix.shape[0])]
            return np.array(results, dtype=matrix.dtype)