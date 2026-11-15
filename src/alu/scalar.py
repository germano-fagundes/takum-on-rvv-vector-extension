from src.core.format import UnpackedFormat

def add(a: UnpackedFormat, b: UnpackedFormat) -> UnpackedFormat:
    is_a_bigger = a.exponent > b.exponent
    bigger = a.copy() if is_a_bigger else b.copy()
    smaller = b.copy() if is_a_bigger else a.copy()
    
    delta = bigger.exponent - smaller.exponent

    if bigger.exponent > smaller.exponent:
        mask = (1 << delta) - 1
        dropped_bits = smaller.mantissa_grs & mask
        smaller.mantissa_grs >>= delta
        half_way = 1 << (delta - 1)
        if dropped_bits > half_way:
            smaller.mantissa_grs += 1
        elif dropped_bits == half_way:
            if (smaller.mantissa_grs & 1) == 1:
                smaller.mantissa_grs += 1

    if a.signal == b.signal:
        result_mantissa = a.mantissa_grs + b.mantissa_grs
        result_signal = a.signal
    elif a.mantissa_grs > b.mantissa_grs:
        result_mantissa = a.mantissa_grs - b.mantissa_grs
        result_signal = a.signal
    elif b.mantissa_grs > a.mantissa_grs:
        result_mantissa = b.mantissa_grs - a.mantissa_grs
        result_signal = b.signal
    else:
        return UnpackedFormat(0, 0, 0, a.mantissa_length)

    hidden_bit_position = a.mantissa_length + 3
    current_length = result_mantissa.bit_length()
    shift_delta = current_length - 1 - hidden_bit_position

    normalized_mantissa = result_mantissa
    normalized_exponent = bigger.exponent

    if shift_delta > 0:
        bits_to_drop_mask = (1 << shift_delta) - 1 
        dropped_bits = result_mantissa & bits_to_drop_mask
        normalized_mantissa = result_mantissa >> shift_delta
        half_way = 1 << (shift_delta - 1)
        if dropped_bits > half_way:
            normalized_mantissa += 1
        elif dropped_bits == half_way:
            if (normalized_mantissa & 1) == 1:
                normalized_mantissa += 1
        normalized_exponent += shift_delta
    elif shift_delta < 0:
        shift_amount = abs(shift_delta)
        normalized_mantissa = result_mantissa << shift_amount
        normalized_exponent -= shift_amount

    return UnpackedFormat(result_signal, normalized_exponent, normalized_mantissa, a.mantissa_length)


def sub(a: UnpackedFormat, b: UnpackedFormat) -> UnpackedFormat:
    simetric_b = b.copy()
    simetric_b.signal ^= 1
    return add(a, simetric_b)


def mul(a: UnpackedFormat, b: UnpackedFormat) -> UnpackedFormat:
    if a.mantissa_grs == 0 or b.mantissa_grs == 0:
        return UnpackedFormat(0, 0, 0, max(a.mantissa_length, b.mantissa_length))

    result_signal = a.signal ^ b.signal
    
    # Produto exato sem truncamento imediato
    raw_mantissa = a.mantissa_grs * b.mantissa_grs
    current_length = raw_mantissa.bit_length()
    
    # O tamanho da mantissa resultante é a soma dos comprimentos originais + bits GRS
    new_mantissa_length = a.mantissa_length + b.mantissa_length + 3
    
    normalized_exponent = (a.exponent + b.exponent + current_length - a.mantissa_length - b.mantissa_length - 7)
    
    # Retorna o formato desempacotado com a mantissa crua e tamanho expandido
    return UnpackedFormat(result_signal, normalized_exponent, raw_mantissa, new_mantissa_length)