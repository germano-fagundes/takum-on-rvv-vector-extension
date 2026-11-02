from mpmath import mp, mpf, log, floor

mp.prec = 113  # type: ignore

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
    def from_unpacked(cls, unpacked: 'UnpackedFormat', total_bits: int):
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
        result_regime, result_exponent = cls.calculate_regime(abs(normalized_exponent))
        
        final_length = final_mantissa.bit_length()
        shift_amount = max(0, final_length - 1)
        mask_remove_hidden = (1 << shift_amount) - 1
        final_mantissa &= mask_remove_hidden

        return cls(unpacked.signal, result_direction, result_regime, final_mantissa, total_bits, result_exponent)

    def to_mpf(self) -> mpf:
        if self.regime == 0 and self.direction == 0 and self.exponent == 0 and self.mantissa == 0:
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
            
        return mpf_result # type: ignore

    @classmethod
    def calculate_regime(cls, abs_exp: int) -> tuple[int, int]:
        if abs_exp == 0:
            return 0, 0
            
        for r in range(1, 8):
            base_scale = (1 << (r + 1)) - 2
            max_exp_value = (1 << r) - 1
            max_regime_capacity = base_scale + max_exp_value
            
            if abs_exp <= max_regime_capacity:
                physical_exponent = abs_exp - base_scale
                return r, physical_exponent
                
        max_exp_value_7 = (1 << 7) - 1
        return 7, max_exp_value_7

    @classmethod
    def from_mpf(cls, n: mpf, total_bits: int):
        if n == 0:
            return Takum(0, 0, 0, 0, total_bits, 0)

        signal = 0 if n > 0 else 1
        module = abs(n)
        exponent = int(floor(log(module, 2)))
        direction = 1 if exponent >= 0 else 0

        abs_exp = abs(exponent)
        regime, physical_exp = cls.calculate_regime(abs_exp)
        mantissa_length = total_bits - 5 - regime

        if mantissa_length < 0:
            regime = total_bits - 5
            physical_exp = (1 << regime) - 1
            final_mantissa = 0
        else:
            normalization = (module / mp.power(2, exponent)) - 1
            raw_slicing = normalization * mp.power(2, mantissa_length + 3)
            bin_slicing = int(raw_slicing)

            grs_bits = bin_slicing & 0b111 
            final_mantissa = bin_slicing >> 3
            
            if grs_bits >= 0b100:
                final_mantissa += 1
                if final_mantissa & (final_mantissa - 1) == 0:
                    final_mantissa >>= 1
                    physical_exp += 1

        return cls(signal, direction, regime, final_mantissa, total_bits, physical_exp)

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