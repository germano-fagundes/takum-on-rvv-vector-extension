class Takum:
    def __init__(self, signal: int, direction: int, regime: int, mantissa: int, total_bits: int, exponent: int|None=None):
        self.signal = signal
        self.direction = direction
        self.regime = regime
        self.exponent = exponent
        self.mantissa = mantissa
        self.total_bits = total_bits
        
    @classmethod
    def from_string(cls, takum: str, total_bits: int):
        signal = int(takum[0], 2)
        direction = int(takum[1], 2)
        regime = int(takum[2:5], 2)
        exponent = None if regime == 0 else int(takum[5:(5+regime)], 2)
        mantissa = int(takum[(5+regime):], 2)
        return cls(signal, direction, regime, mantissa, total_bits, exponent)

    @classmethod
    def from_unpacked(cls, unpacked: 'UnpackedFormat', total_bits: int): # type: ignore
        if unpacked.mantissa_grs == 0:
            return cls(0, 0, 0, 0, total_bits, 0)
            
        grs_bits = unpacked.mantissa_grs & 0b111 
        final_mantissa = unpacked.mantissa_grs >> 3
        normalized_exponent = unpacked.exponent
        
        if grs_bits >= 0b100:
            final_mantissa += 1
            if final_mantissa & (final_mantissa - 1) == 0:
                final_mantissa >>= 1
                normalized_exponent += 1
                
        result_direction = 1 if normalized_exponent >= 0 else 0
        result_regime = 0
        result_exponent = 0

        abs_exp = abs(normalized_exponent)
        if abs_exp > 0:
            for r in range(1, 8):
                base_scale = (1 << (r + 1)) - 2
                max_exp_value = (1 << r) - 1
                max_regime_capacity = base_scale + max_exp_value
                
                if abs_exp <= max_regime_capacity:
                    result_regime = r
                    result_exponent = abs(abs_exp - base_scale)
                    break
        
        final_length = final_mantissa.bit_length()
        shift_amount = max(0, final_length - 1)
        mask_remove_hidden = (1 << shift_amount) - 1
        final_mantissa &= mask_remove_hidden

        return cls(unpacked.signal, result_direction, result_regime, final_mantissa, total_bits, result_exponent)


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
    def get_unbiased_exponent(takum: Takum):
        e = takum.exponent
        r = takum.regime
        d = takum.direction

        base_scale = 0 if r == 0 else (1 << (r + 1)) - 2

        if d == 0:
            base_scale = -base_scale - 1
            unbiased_exponent = base_scale - e if e is not None else base_scale
        else:
            unbiased_exponent = base_scale + e if e is not None else base_scale

        return unbiased_exponent

    @staticmethod
    def get_full_mantissa(takum: Takum) -> tuple[int, int]:
        mantissa_length = takum.total_bits - (5 + takum.regime)
        
        if takum.mantissa == 0:
            return 0, mantissa_length

        true_significand = (1 << mantissa_length) | takum.mantissa

        # adds bits G, R and S
        grs_added = true_significand << 3
        return grs_added, mantissa_length

    def copy(self):
        return UnpackedFormat(self.signal, self.exponent, self.mantissa_grs, self.mantissa_length)