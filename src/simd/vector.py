from src.core.format import Takum, UnpackedFormat
from src.alu.scalar import add, sub, mul

def vtsetup(a: list[Takum], b: list[Takum]):
    if not a:
        return [], [], 0

    unpacked_a: list[UnpackedFormat] = []
    unpacked_b: list[UnpackedFormat] = []
    
    for i in range(len(a)):
        unpacked_a.append(UnpackedFormat.from_takum(a[i]))
        unpacked_b.append(UnpackedFormat.from_takum(b[i]))
    
    total_bits = a[0].total_bits

    return unpacked_a, unpacked_b, total_bits


def vtadd(a: list[Takum], b: list[Takum]):
    if len(a) != len(b):
        raise ValueError('The arrays to be summed must have the same length.')

    unpacked_a, unpacked_b, total_bits = vtsetup(a, b)
    return [Takum.from_unpacked(add(ua, ub), total_bits) for ua, ub in zip(unpacked_a, unpacked_b)]


def vtsub(a: list[Takum], b: list[Takum]):
    if len(a) != len(b):
        raise ValueError('The arrays to be subtracted must have the same length.')

    unpacked_a, unpacked_b, total_bits = vtsetup(a, b)
    return [Takum.from_unpacked(sub(ua, ub), total_bits) for ua, ub in zip(unpacked_a, unpacked_b)]


def vtmul(a: list[Takum], b: list[Takum]):
    if len(a) != len(b):
        raise ValueError('The arrays to be multiplied must have the same length.')

    unpacked_a, unpacked_b, total_bits = vtsetup(a, b)
    return [Takum.from_unpacked(mul(ua, ub), total_bits) for ua, ub in zip(unpacked_a, unpacked_b)]


def vtdot(a: list[Takum], b: list[Takum]):
    if len(a) != len(b):
        raise ValueError('The arrays to be MAC`d must have the same length.')
    if not a:
        raise ValueError('Vector operations cannot be performed on empty arrays.')

    unpacked_a, unpacked_b, total_bits = vtsetup(a, b)
    acc = UnpackedFormat(0, 0, 0, unpacked_a[0].mantissa_length)

    for ua, ub in zip(unpacked_a, unpacked_b):
        product = mul(ua, ub)
        acc = add(acc, product)

    return Takum.from_unpacked(acc, total_bits)


def vncvt(takum: Takum, new_total_bits: int):
    bits_to_drop = takum.total_bits - new_total_bits
    if bits_to_drop < 0 or new_total_bits < 8:
        raise ValueError('The new_total_bits must be at least 8 and less than the older total bits.')
    if bits_to_drop == 0:
        return takum
    
    unpacked = UnpackedFormat.from_takum(takum)

    bits_to_drop_mask = (1 << bits_to_drop) - 1 
    dropped_bits = unpacked.mantissa_grs & bits_to_drop_mask

    unpacked.mantissa_grs >>= bits_to_drop
    if dropped_bits > 0:
        unpacked.mantissa_grs |= 1

    unpacked.mantissa_length -= bits_to_drop

    return Takum.from_unpacked(unpacked, new_total_bits)