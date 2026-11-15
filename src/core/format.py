from mpmath import mp, mpf, log, floor

mp.prec = 113  # type: ignore


class Takum:
    def __init__(self, signal: int, direction: int, regime: int, mantissa: int, total_bits: int, exponent: int | None = None):
        self.signal = signal
        self.direction = direction
        self.regime = regime
        self.exponent = exponent if exponent is not None else 0
        self.mantissa = mantissa
        self.total_bits = total_bits

    def extract_fields(self) -> dict[str, str | int]:
        """Extração explícita de S, D, R, C e F conforme Fase 1."""
        r_bits = self.regime if self.direction == 1 else (7 - self.regime)
        avail_c = max(0, min(self.regime, self.total_bits - 5))
        mantissa_len = max(0, self.total_bits - 5 - self.regime)
        return {
            'S': self.signal,
            'D': self.direction,
            'R': format(r_bits, '03b'),
            'C': format(self.exponent, f'0{avail_c}b') if avail_c > 0 else '',
            'F': format(self.mantissa, f'0{mantissa_len}b') if mantissa_len > 0 else '',
        }

    @classmethod
    def from_string(cls, takum: str, total_bits: int):
        signal = int(takum[0], 2)
        direction = int(takum[1], 2)
        raw_r = int(takum[2:5], 2)
        regime = raw_r if direction == 1 else (7 - raw_r)

        avail_c = max(0, min(regime, total_bits - 5))
        if avail_c == 0:
            exponent = 0
        else:
            c_val = int(takum[5 : 5 + avail_c], 2)
            exponent = c_val << (regime - avail_c)

        mantissa_start = 5 + avail_c
        mantissa = int(takum[mantissa_start:], 2) if mantissa_start < total_bits else 0
        return cls(signal, direction, regime, mantissa, total_bits, exponent)

    @classmethod
    def encode_exponent(cls, unbiased_exp: int) -> tuple[int, int, int]:
        """Converte expoente sem viés em (direction, regime r, characteristic c)."""
        if unbiased_exp >= 0:
            direction = 1
            k = min(unbiased_exp, 254)
        else:
            direction = 0
            k = min(-unbiased_exp - 1, 254)

        if k == 0:
            return direction, 0, 0

        for r in range(1, 8):
            base = (1 << r) - 1
            max_k = (1 << (r + 1)) - 2
            if k <= max_k:
                offset = k - base
                c = offset if direction == 1 else ((1 << r) - 1 - offset)
                return direction, r, c

        return direction, 7, ((1 << 7) - 1 if direction == 1 else 0)

    @classmethod
    def from_unpacked(cls, unpacked: 'UnpackedFormat', total_bits: int):
        if unpacked.mantissa_grs == 0:
            return cls(0, 0, 0, 0, total_bits, 0)

        grs_bits = unpacked.mantissa_grs & 0b111
        final_mantissa = unpacked.mantissa_grs >> 3
        normalized_exponent = unpacked.exponent

        if grs_bits > 0b100 or (grs_bits == 0b100 and (final_mantissa % 2) == 1):
            final_mantissa += 1
            if final_mantissa & (final_mantissa - 1) == 0:
                final_mantissa >>= 1
                normalized_exponent += 1

        result_direction, result_regime, result_c = cls.encode_exponent(normalized_exponent)
        target_mantissa_length = total_bits - 5 - result_regime

        if target_mantissa_length <= 0:
            avail_c = max(0, total_bits - 5)
            drop_c = result_regime - avail_c
            result_c = (result_c >> drop_c) << drop_c if drop_c > 0 else result_c
            final_mantissa = 0
        else:
            final_length = final_mantissa.bit_length()
            shift_amount = max(0, final_length - 1)
            mask_remove_hidden = (1 << shift_amount) - 1
            final_mantissa &= mask_remove_hidden

            if shift_amount > target_mantissa_length:
                drop = shift_amount - target_mantissa_length
                rem = final_mantissa & ((1 << drop) - 1)
                final_mantissa >>= drop
                half = 1 << (drop - 1)
                if rem > half or (rem == half and (final_mantissa & 1) == 1):
                    final_mantissa += 1
                    if final_mantissa == (1 << target_mantissa_length):
                        final_mantissa = 0
                        normalized_exponent += 1
                        result_direction, result_regime, result_c = cls.encode_exponent(normalized_exponent)
            elif shift_amount < target_mantissa_length:
                final_mantissa <<= (target_mantissa_length - shift_amount)

        return cls(unpacked.signal, result_direction, result_regime, final_mantissa, total_bits, result_c)

    def to_mpf(self) -> mpf:
        if self.signal == 0 and self.direction == 0 and self.regime == 0 and self.exponent == 0 and self.mantissa == 0:
            return mp.mpf(0.0)

        exponent = UnpackedFormat.get_unbiased_exponent(self)
        mantissa_length = self.total_bits - 5 - self.regime

        if mantissa_length <= 0:
            fraction = mp.mpf(0.0)
        else:
            fraction = mp.fdiv(self.mantissa, mp.power(2, mantissa_length))

        significand = mp.fadd(1.0, fraction)
        mpf_result = mp.fmul(significand, mp.power(2, exponent))

        if self.signal == 1:
            mpf_result = mp.fneg(mpf_result)

        return mpf_result  # type: ignore

    @classmethod
    def from_mpf(cls, n: mpf, total_bits: int):
        if n == 0:
            return Takum(0, 0, 0, 0, total_bits, 0)

        signal = 0 if n > 0 else 1
        module = abs(n)
        exponent = int(floor(log(module, 2)))

        direction, regime, physical_c = cls.encode_exponent(exponent)
        mantissa_length = total_bits - 5 - regime

        if mantissa_length <= 0:
            avail_c = max(0, total_bits - 5)
            drop_c = regime - avail_c
            physical_c = (physical_c >> drop_c) << drop_c if drop_c > 0 else physical_c
            final_mantissa = 0
        else:
            normalization = (module / mp.power(2, exponent)) - 1
            raw_slicing = normalization * mp.power(2, mantissa_length + 3)
            bin_slicing = int(raw_slicing)

            grs_bits = bin_slicing & 0b111
            final_mantissa = bin_slicing >> 3

            if grs_bits > 0b100 or (grs_bits == 0b100 and (final_mantissa % 2) == 1):
                final_mantissa += 1
                if final_mantissa == (1 << mantissa_length):
                    final_mantissa = 0
                    exponent += 1
                    direction, regime, physical_c = cls.encode_exponent(exponent)

        return cls(signal, direction, regime, final_mantissa, total_bits, physical_c)


class UnpackedFormat:
    def __init__(self, signal: int, exponent: int, mantissa_grs: int, mantissa_length: int):
        self.signal = signal
        self.exponent = exponent
        self.mantissa_grs = mantissa_grs
        self.mantissa_length = mantissa_length

    @classmethod
    def from_takum(cls, takum: Takum):
        signal = takum.signal
        exponent = cls.get_unbiased_exponent(takum)
        mantissa_grs, mantissa_length = cls.get_full_mantissa(takum)
        return cls(signal, exponent, mantissa_grs, mantissa_length)

    @staticmethod
    def get_unbiased_exponent(takum: Takum) -> int:
        c = takum.exponent if takum.exponent is not None else 0
        r = takum.regime
        d = takum.direction

        if d == 1:
            return 0 if r == 0 else ((1 << r) - 1) + c
        else:
            return -1 if r == 0 else -(1 << (r + 1)) + 1 + c

    @staticmethod
    def get_full_mantissa(takum: Takum) -> tuple[int, int]:
        mantissa_length = max(0, takum.total_bits - (5 + takum.regime))
        true_significand = (1 << mantissa_length) | takum.mantissa
        grs_added = true_significand << 3
        return grs_added, mantissa_length

    def copy(self):
        return UnpackedFormat(self.signal, self.exponent, self.mantissa_grs, self.mantissa_length)