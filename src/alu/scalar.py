from src.core.format import UnpackedFormat

def add(a: UnpackedFormat, b: UnpackedFormat) -> UnpackedFormat:
    if a.mantissa_grs == 0:
        return b.copy()
    if b.mantissa_grs == 0:
        return a.copy()

    # Igualar largura base das mantissas antes de alinhar expoentes
    target_len = max(a.mantissa_length, b.mantissa_length)
    op_a = a.copy()
    op_b = b.copy()
    if op_a.mantissa_length < target_len:
        op_a.mantissa_grs <<= (target_len - op_a.mantissa_length)
        op_a.mantissa_length = target_len
    if op_b.mantissa_length < target_len:
        op_b.mantissa_grs <<= (target_len - op_b.mantissa_length)
        op_b.mantissa_length = target_len

    is_a_bigger = (op_a.exponent > op_b.exponent) or (
        op_a.exponent == op_b.exponent and op_a.mantissa_grs >= op_b.mantissa_grs
    )
    bigger = op_a if is_a_bigger else op_b
    smaller = op_b if is_a_bigger else op_a

    delta = bigger.exponent - smaller.exponent

    if delta > 0:
        mask = (1 << delta) - 1
        dropped_bits = smaller.mantissa_grs & mask
        smaller.mantissa_grs >>= delta
        half_way = 1 << (delta - 1)
        if dropped_bits > half_way or (dropped_bits == half_way and (smaller.mantissa_grs & 1) == 1):
            smaller.mantissa_grs += 1

    if bigger.signal == smaller.signal:
        result_mantissa = bigger.mantissa_grs + smaller.mantissa_grs
        result_signal = bigger.signal
    elif bigger.mantissa_grs > smaller.mantissa_grs:
        result_mantissa = bigger.mantissa_grs - smaller.mantissa_grs
        result_signal = bigger.signal
    else:
        return UnpackedFormat(0, 0, 0, target_len)

    hidden_bit_position = target_len + 3
    current_length = result_mantissa.bit_length()
    shift_delta = current_length - 1 - hidden_bit_position

    normalized_mantissa = result_mantissa
    normalized_exponent = bigger.exponent

    if shift_delta > 0:
        bits_to_drop_mask = (1 << shift_delta) - 1
        dropped_bits = result_mantissa & bits_to_drop_mask
        normalized_mantissa = result_mantissa >> shift_delta
        half_way = 1 << (shift_delta - 1)
        if dropped_bits > half_way or (dropped_bits == half_way and (normalized_mantissa & 1) == 1):
            normalized_mantissa += 1
            if normalized_mantissa.bit_length() - 1 > hidden_bit_position:
                normalized_mantissa >>= 1
                normalized_exponent += 1
        normalized_exponent += shift_delta
    elif shift_delta < 0:
        shift_amount = abs(shift_delta)
        normalized_mantissa = result_mantissa << shift_amount
        normalized_exponent -= shift_amount

    return UnpackedFormat(result_signal, normalized_exponent, normalized_mantissa, target_len)


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