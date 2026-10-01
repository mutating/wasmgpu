// Integer and IEEE-754 operations. vec2 words are little endian.
// Floating point is implemented with integers, including subnormals and f64.
// All arithmetic in this file executes on the GPU.
fn add64(a: vec2u, b: vec2u) -> vec2u {
    let lo = a.x + b.x;
    return vec2u(lo, a.y + b.y + u32(lo < a.x));
}
fn neg64(a: vec2u) -> vec2u { return add64(~a, vec2u(1u, 0u)); }
fn sub64(a: vec2u, b: vec2u) -> vec2u { return add64(a, neg64(b)); }
fn zero64(a: vec2u) -> bool { return (a.x | a.y) == 0u; }
fn lt64(a: vec2u, b: vec2u) -> bool { return a.y < b.y || (a.y == b.y && a.x < b.x); }
fn eq64(a: vec2u, b: vec2u) -> bool { return all(a == b); }
fn shl64(a: vec2u, n: u32) -> vec2u {
    if n == 0u { return a; }
    if n >= 64u { return vec2u(0u); }
    if n >= 32u { return vec2u(0u, a.x << (n - 32u)); }
    return vec2u(a.x << n, (a.y << n) | (a.x >> (32u - n)));
}
fn shr64(a: vec2u, n: u32) -> vec2u {
    if n == 0u { return a; }
    if n >= 64u { return vec2u(0u); }
    if n >= 32u { return vec2u(a.y >> (n - 32u), 0u); }
    return vec2u((a.x >> n) | (a.y << (32u - n)), a.y >> n);
}
fn jam64(a: vec2u, n: u32) -> vec2u {
    let shifted = shr64(a, n);
    return shifted | vec2u(u32(!eq64(shl64(shifted, n), a)), 0u);
}
fn bit64(a: vec2u, n: i32) -> u32 {
    if n < 0 || n >= 64 { return 0u; }
    return shr64(a, u32(n)).x & 1u;
}
fn clz64(a: vec2u) -> u32 {
    if a.y != 0u { return countLeadingZeros(a.y); }
    return 32u + countLeadingZeros(a.x);
}
fn mul32(a: u32, b: u32) -> vec2u {
    let a0 = a & 65535u; let a1 = a >> 16u;
    let b0 = b & 65535u; let b1 = b >> 16u;
    let p0 = a0 * b0;
    let p1 = a1 * b0 + (p0 >> 16u);
    let p2 = a0 * b1 + (p1 & 65535u);
    return vec2u((p0 & 65535u) | (p2 << 16u), a1 * b1 + (p1 >> 16u) + (p2 >> 16u));
}
fn mul64(a: vec2u, b: vec2u) -> vec2u {
    let low = mul32(a.x, b.x);
    return vec2u(low.x, low.y + a.x * b.y + a.y * b.x);
}
struct Division { quotient: vec2u, remainder: vec2u }
fn div64(a: vec2u, b: vec2u) -> Division {
    var q = vec2u(0u); var rem = vec2u(0u);
    for (var i = 63; i >= 0; i -= 1) {
        let carry = rem.y >> 31u;
        rem = shl64(rem, 1u) | vec2u(bit64(a, i), 0u);
        if carry != 0u || !lt64(rem, b) {
            rem = sub64(rem, b);
            q |= shl64(vec2u(1u, 0u), u32(i));
        }
    }
    return Division(q, rem);
}
fn sar64(a: vec2u, n: u32) -> vec2u {
    let shift = n & 63u;
    if shift == 0u { return a; }
    let sign = a.y >> 31u;
    return shr64(a, shift) | select(vec2u(0u), shl64(vec2u(0xffffffffu), 64u - shift), sign != 0u);
}
fn add128(a: vec4u, b: vec4u) -> vec4u {
    let lo = add64(a.xy, b.xy);
    let hi = add64(add64(a.zw, b.zw), vec2u(u32(lt64(lo, a.xy)), 0u));
    return vec4u(lo, hi);
}
fn mul128(a: vec2u, b: vec2u) -> vec4u {
    let p0 = mul32(a.x, b.x); let p1 = mul32(a.x, b.y);
    let p2 = mul32(a.y, b.x); let p3 = mul32(a.y, b.y);
    return add128(add128(vec4u(p0, p3), vec4u(0u, p1, 0u)), vec4u(0u, p2, 0u));
}
struct SoftFloat { sign: u32, exponent: i32, sig: vec2u, kind: u32 }
fn unpack_float(bits: vec2u, single: bool) -> SoftFloat {
    var sign = bits.y >> 31u;
    var exponent = i32((bits.y >> 20u) & 2047u);
    var sig = vec2u(bits.x, bits.y & 0xfffffu);
    var max_exp = 2047; var bias = 1023;
    if single {
        sign = bits.x >> 31u; exponent = i32((bits.x >> 23u) & 255u);
        sig = shl64(vec2u(bits.x & 0x7fffffu, 0u), 29u);
        max_exp = 255; bias = 127;
    }
    if exponent == max_exp { return SoftFloat(sign, 0, sig, select(2u, 1u, zero64(sig))); }
    if exponent == 0 {
        exponent = 1 - bias;
        if !zero64(sig) {
            let shift = clz64(sig) - 11u;
            sig = shl64(sig, shift); exponent -= i32(shift);
        }
    } else {
        exponent -= bias; sig.y |= 0x100000u;
    }
    return SoftFloat(sign, exponent, sig, 0u);
}
fn float_special(sign: u32, nan: bool, single: bool) -> vec2u {
    if single { return vec2u((sign << 31u) | 0x7f800000u | select(0u, 0x400000u, nan), 0u); }
    return vec2u(0u, (sign << 31u) | 0x7ff00000u | select(0u, 0x80000u, nan));
}
fn float_zero(sign: u32, single: bool) -> vec2u {
    if single { return vec2u(sign << 31u, 0u); }
    return vec2u(0u, sign << 31u);
}
fn pack_float(sign: u32, original_exp: i32, original_sig: vec2u, single: bool) -> vec2u {
    if zero64(original_sig) { return float_zero(sign, single); }
    var sig = original_sig; var exponent = original_exp;
    while sig.y >= 0x1000000u { sig = jam64(sig, 1u); exponent += 1; }
    while sig.y < 0x800000u { sig = shl64(sig, 1u); exponent -= 1; }
    let min_exp = select(-1022, -126, single); let max_exp = select(1023, 127, single);
    if exponent < min_exp { sig = jam64(sig, u32(min_exp - exponent)); exponent = min_exp; }
    if single { sig = jam64(sig, 29u); }
    let round = sig.x & 7u;
    sig = shr64(sig, 3u);
    if round > 4u || (round == 4u && (sig.x & 1u) != 0u) { sig = add64(sig, vec2u(1u, 0u)); }
    let mantissa_bits = select(52u, 23u, single);
    if bit64(sig, i32(mantissa_bits + 1u)) != 0u { sig = shr64(sig, 1u); exponent += 1; }
    if exponent > max_exp { return float_special(sign, false, single); }
    let encoded_exp = select(0u, u32(exponent + max_exp), bit64(sig, i32(mantissa_bits)) != 0u);
    if single { return vec2u((sign << 31u) | (encoded_exp << 23u) | (sig.x & 0x7fffffu), 0u); }
    return vec2u(sig.x, (sign << 31u) | (encoded_exp << 20u) | (sig.y & 0xfffffu));
}
fn soft_add(x: vec2u, y: vec2u, subtract: bool, single: bool) -> vec2u {
    var a = unpack_float(x, single); var b = unpack_float(y, single);
    if subtract { b.sign ^= 1u; }
    if a.kind == 2u || b.kind == 2u { return float_special(0u, true, single); }
    if a.kind == 1u {
        return float_special(a.sign, b.kind == 1u && b.sign != a.sign, single);
    }
    if b.kind == 1u { return float_special(b.sign, false, single); }
    if zero64(a.sig) && zero64(b.sig) { return float_zero(a.sign & b.sign, single); }
    if zero64(a.sig) { return pack_float(b.sign, b.exponent, shl64(b.sig, 3u), single); }
    if zero64(b.sig) { return pack_float(a.sign, a.exponent, shl64(a.sig, 3u), single); }
    if a.exponent < b.exponent || (a.exponent == b.exponent && lt64(a.sig, b.sig)) {
        let swap = a; a = b; b = swap;
    }
    let aa = shl64(a.sig, 3u); let bb = jam64(shl64(b.sig, 3u), u32(a.exponent - b.exponent));
    if a.sign == b.sign { return pack_float(a.sign, a.exponent, add64(aa, bb), single); }
    let difference = sub64(aa, bb);
    return pack_float(select(a.sign, 0u, zero64(difference)), a.exponent, difference, single);
}
fn soft_mul(x: vec2u, y: vec2u, single: bool) -> vec2u {
    let a = unpack_float(x, single); let b = unpack_float(y, single); let sign = a.sign ^ b.sign;
    if a.kind == 2u || b.kind == 2u { return float_special(0u, true, single); }
    if a.kind == 1u || b.kind == 1u {
        return float_special(sign, (a.kind == 0u && zero64(a.sig)) || (b.kind == 0u && zero64(b.sig)), single);
    }
    let product = mul128(a.sig, b.sig);
    // Product / 2^49 gives a significand with three rounding bits.
    let sig = vec2u((product.y >> 17u) | (product.z << 15u), (product.z >> 17u) | (product.w << 15u));
    let sticky = u32(product.x != 0u || (product.y & 0x1ffffu) != 0u);
    return pack_float(sign, a.exponent + b.exponent, sig | vec2u(sticky, 0u), single);
}
fn soft_div(x: vec2u, y: vec2u, single: bool) -> vec2u {
    let a = unpack_float(x, single); let b = unpack_float(y, single); let sign = a.sign ^ b.sign;
    if a.kind == 2u || b.kind == 2u || (a.kind == 1u && b.kind == 1u) || (a.kind == 0u && b.kind == 0u && zero64(a.sig) && zero64(b.sig)) {
        return float_special(0u, true, single);
    }
    if a.kind == 1u || (b.kind == 0u && zero64(b.sig)) { return float_special(sign, false, single); }
    if b.kind == 1u || zero64(a.sig) { return float_zero(sign, single); }
    var rem = a.sig; var quotient = vec2u(0u);
    for (var i = 0u; i < 57u; i += 1u) {
        quotient = shl64(quotient, 1u);
        if !lt64(rem, b.sig) { rem = sub64(rem, b.sig); quotient.x |= 1u; }
        rem = shl64(rem, 1u);
    }
    quotient.x |= u32(!zero64(rem));
    return pack_float(sign, a.exponent - b.exponent - 1, quotient, single);
}
fn soft_sqrt(x: vec2u, single: bool) -> vec2u {
    let a = unpack_float(x, single);
    if a.kind == 2u { return float_special(0u, true, single); }
    if a.kind == 0u && zero64(a.sig) { return x; }
    if a.sign != 0u { return float_special(0u, true, single); }
    if a.kind == 1u { return x; }
    let shift = 58 + (a.exponent & 1);
    var root = vec2u(0u); var rem = vec2u(0u);
    for (var i = 55; i >= 0; i -= 1) {
        let pair = (bit64(a.sig, i * 2 + 1 - shift) << 1u) | bit64(a.sig, i * 2 - shift);
        rem = shl64(rem, 2u) | vec2u(pair, 0u);
        let test = shl64(root, 2u) | vec2u(1u, 0u);
        root = shl64(root, 1u);
        if !lt64(rem, test) { rem = sub64(rem, test); root.x |= 1u; }
    }
    root.x |= u32(!zero64(rem));
    return pack_float(0u, a.exponent >> 1, root, single);
}
// Comparison: 0 equal, 1 less, 2 greater, 3 unordered.
fn float_compare(x: vec2u, y: vec2u, single: bool) -> u32 {
    let a = unpack_float(x, single); let b = unpack_float(y, single);
    if a.kind == 2u || b.kind == 2u { return 3u; }
    if a.kind == 0u && b.kind == 0u && zero64(a.sig) && zero64(b.sig) { return 0u; }
    if eq64(x, y) { return 0u; }
    if a.sign != b.sign { return select(2u, 1u, a.sign != 0u); }
    return select(2u, 1u, lt64(x, y) != (a.sign != 0u));
}
fn soft_round(x: vec2u, mode: u32, single: bool) -> vec2u {
    let a = unpack_float(x, single);
    if a.kind == 2u { return float_special(0u, true, single); }
    if a.kind == 1u || zero64(a.sig) || a.exponent >= 52 { return x; }
    let shift = u32(max(0, 52 - a.exponent));
    var integer = shr64(a.sig, shift);
    let remainder = sub64(a.sig, shl64(integer, shift));
    if !zero64(remainder) {
        if (mode == 0u && a.sign == 0u) || (mode == 1u && a.sign != 0u) {
            integer = add64(integer, vec2u(1u, 0u));
        } else if mode == 3u && shift <= 53u {
            let half = shl64(vec2u(1u, 0u), shift - 1u);
            if lt64(half, remainder) || (eq64(half, remainder) && (integer.x & 1u) != 0u) {
                integer = add64(integer, vec2u(1u, 0u));
            }
        }
    }
    return pack_float(a.sign, 52, shl64(integer, 3u), single);
}
fn integer_to_float(x: vec2u, is_signed: bool, wide: bool, single: bool) -> vec2u {
    var integer = x; var sign = 0u;
    if wide {
        if is_signed && (x.y >> 31u) != 0u { sign = 1u; integer = neg64(x); }
    } else {
        integer.y = 0u;
        if is_signed && (x.x >> 31u) != 0u { sign = 1u; integer.x = 0u - x.x; }
    }
    if zero64(integer) { return float_zero(0u, single); }
    let top = 63u - clz64(integer);
    var extended = vec2u(0u);
    if top > 55u { extended = jam64(integer, top - 55u); }
    else { extended = shl64(integer, 55u - top); }
    return pack_float(sign, i32(top), extended, single);
}
fn convert_float(x: vec2u, from_single: bool) -> vec2u {
    let a = unpack_float(x, from_single);
    if a.kind != 0u { return float_special(a.sign, a.kind == 2u, !from_single); }
    return pack_float(a.sign, a.exponent, shl64(a.sig, 3u), !from_single);
}
struct IntConversion { value: vec2u, trap: u32 }
fn float_to_integer(x: vec2u, single: bool, wide: bool, is_signed: bool, saturate: bool) -> IntConversion {
    let a = unpack_float(x, single);
    let size = select(32u, 64u, wide);
    let max_unsigned = select(vec2u(0xffffffffu, 0u), vec2u(0xffffffffu), wide);
    let minimum = shl64(vec2u(1u, 0u), size - 1u);
    let maximum = select(max_unsigned, sub64(minimum, vec2u(1u, 0u)), is_signed);
    if a.kind == 2u { return IntConversion(vec2u(0u), select(6u, 0u, saturate)); }
    var magnitude = vec2u(0u);
    if a.exponent >= 52 { magnitude = shl64(a.sig, u32(a.exponent - 52)); }
    else { magnitude = shr64(a.sig, u32(52 - a.exponent)); }
    var invalid = a.kind == 1u || a.exponent >= i32(size);
    if is_signed { invalid = invalid || lt64(select(maximum, minimum, a.sign != 0u), magnitude); }
    else { invalid = invalid || (a.sign != 0u && !zero64(magnitude)); }
    if invalid {
        if !saturate { return IntConversion(vec2u(0u), 5u); }
        return IntConversion(select(maximum, select(vec2u(0u), minimum, is_signed), a.sign != 0u), 0u);
    }
    var result = select(magnitude, neg64(magnitude), a.sign != 0u);
    if !wide { result.y = 0u; }
    return IntConversion(result, 0u);
}
