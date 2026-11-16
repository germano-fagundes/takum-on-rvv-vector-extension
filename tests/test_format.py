import unittest
from mpmath import mp

from src.core.format import Takum, UnpackedFormat

mp.prec = 113  # type: ignore


class TestTakumFormat(unittest.TestCase):
    def test_zero_representation(self):
        for bits in (8, 16, 32):
            t = Takum.from_mpf(mp.mpf(0), bits)
            self.assertEqual((t.signal, t.direction, t.regime, t.exponent, t.mantissa), (0, 0, 0, 0, 0))
            self.assertEqual(t.to_mpf(), mp.mpf(0))

    def test_powers_of_two_exact(self):
        for exp in (-15, -4, -1, 0, 1, 5, 14, 50):
            val = mp.power(2, exp)
            for bits in (16, 32):
                t = Takum.from_mpf(val, bits)
                self.assertEqual(t.to_mpf(), val, f"Falha em 2^{exp} para T{bits}")

    def test_negative_values_and_extract_fields(self):
        t = Takum.from_mpf(mp.mpf("-12.5"), 16)
        fields = t.extract_fields()

        self.assertEqual(fields["S"], 1)
        self.assertEqual(fields["D"], 1)
        self.assertEqual(len(str(fields["R"])), 3)
        self.assertEqual(len(str(fields["C"])), t.regime)
        self.assertEqual(len(str(fields["F"])), 16 - 5 - t.regime)
        self.assertAlmostEqual(float(t.to_mpf()), -12.5, places=3)

    def test_from_string_roundtrip(self):
        samples = [mp.mpf("0.75"), mp.mpf("-0.125"), mp.mpf("42.0"), mp.mpf("1e-4")]
        for val in samples:
            for bits in (8, 16, 32):
                t_orig = Takum.from_mpf(val, bits)
                f = t_orig.extract_fields()
                bitstring = f"{f['S']}{f['D']}{f['R']}{f['C']}{f['F']}"
                self.assertEqual(len(bitstring), bits)

                t_parsed = Takum.from_string(bitstring, bits)
                self.assertEqual(t_parsed.to_mpf(), t_orig.to_mpf())

    def test_t8_high_regime_truncation(self):
        t8 = Takum.from_mpf(mp.power(2, 40), 8)
        self.assertGreater(float(t8.to_mpf()), 0.0)

    def test_rnte_tie_to_even_rounding(self):
        # 1.00_100 (empate com mantissa par) -> mantém 1.00 (1.0)
        u_even = UnpackedFormat(signal=0, exponent=0, mantissa_grs=0b100_100, mantissa_length=2)
        self.assertEqual(Takum.from_unpacked(u_even, 8).to_mpf(), mp.mpf("1.0"))

        # 1.01_100 (empate com mantissa ímpar) -> arredonda para 1.10 (1.5)
        u_odd = UnpackedFormat(signal=0, exponent=0, mantissa_grs=0b101_100, mantissa_length=2)
        self.assertEqual(Takum.from_unpacked(u_odd, 8).to_mpf(), mp.mpf("1.5"))


if __name__ == "__main__":
    unittest.main()