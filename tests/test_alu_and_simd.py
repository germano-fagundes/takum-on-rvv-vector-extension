import unittest
from mpmath import mp
import numpy as np

from src.core.format import Takum, UnpackedFormat
from src.alu.scalar import add, sub, mul
from src.simd.vector import vtadd, vtsub, vtmul, vtdot, vncvt
from src.simd.simulator import RVVSimulator

mp.prec = 113  # type: ignore


class TestScalarALU(unittest.TestCase):
    def test_scalar_add_sub_mul_different_exponents(self):
        a = UnpackedFormat.from_takum(Takum.from_mpf(mp.mpf("10.5"), 32))
        b = UnpackedFormat.from_takum(Takum.from_mpf(mp.mpf("0.25"), 32))

        res_add = Takum.from_unpacked(add(a, b), 32)
        res_sub = Takum.from_unpacked(sub(a, b), 32)
        res_mul = Takum.from_unpacked(mul(a, b), 32)

        self.assertAlmostEqual(float(res_add.to_mpf()), 10.75, places=6)
        self.assertAlmostEqual(float(res_sub.to_mpf()), 10.25, places=6)
        self.assertAlmostEqual(float(res_mul.to_mpf()), 2.625, places=6)

    def test_scalar_cancellation_to_zero(self):
        a = UnpackedFormat.from_takum(Takum.from_mpf(mp.mpf("7.25"), 16))
        res = Takum.from_unpacked(sub(a, a), 16)
        self.assertEqual(res.to_mpf(), mp.mpf(0))


class TestZvtakumSIMDAndSimulator(unittest.TestCase):
    def setUp(self):
        self.vec_a = [Takum.from_mpf(mp.mpf(v), 16) for v in ("1.5", "-2.0", "4.0", "0.25")]
        self.vec_b = [Takum.from_mpf(mp.mpf(v), 16) for v in ("2.0", "3.0", "-0.5", "4.0")]

    def test_vector_elementwise_ops(self):
        self.assertEqual([float(x.to_mpf()) for x in vtadd(self.vec_a, self.vec_b)], [3.5, 1.0, 3.5, 4.25])
        self.assertEqual([float(x.to_mpf()) for x in vtsub(self.vec_a, self.vec_b)], [-0.5, -5.0, 4.5, -3.75])
        self.assertEqual([float(x.to_mpf()) for x in vtmul(self.vec_a, self.vec_b)], [3.0, -6.0, -2.0, 1.0])

    def test_vtdot_quire_accumulation(self):
        dot_res = vtdot(self.vec_a, self.vec_b)
        self.assertEqual(float(dot_res.to_mpf()), -4.0)

    def test_vncvt_narrowing_t16_to_t8(self):
        t16 = Takum.from_mpf(mp.mpf("3.25"), 16)
        t8 = vncvt(t16, 8)
        self.assertEqual(t8.total_bits, 8)
        self.assertAlmostEqual(float(t8.to_mpf()), 3.25, places=1)

        with self.assertRaises(ValueError):
            vncvt(t16, 32)

    def test_rvv_simulator_strip_mining_and_csrs(self):
        sim_tk = RVVSimulator(arch="takum", vlen_bits=256, datapath_bits=64)
        sim_fp = RVVSimulator(arch="ieee754", vlen_bits=256, datapath_bits=64)

        row_tk = [Takum.from_mpf(mp.mpf("1.0"), 16) for _ in range(20)]
        res_tk = sim_tk.dot_product_takum(row_tk, row_tk)
        self.assertEqual(float(res_tk.to_mpf()), 20.0)
        self.assertEqual(sim_tk.instruction_breakdown["vsetvli"], 2)
        self.assertEqual(sim_tk.instruction_breakdown["vtdot.vv"], 2)

        row_fp = np.full(20, 60000.0, dtype=np.float16)
        sim_fp.dot_product_ieee(row_fp, row_fp, sew=16)
        self.assertGreater(sim_fp.exception_events, 0)
        self.assertGreater(sim_fp.fflags["OF"], 0)


if __name__ == "__main__":
    unittest.main()