struct NumericResult { value: vec2u, trap: u32 }
fn numeric(op: u32, a: vec2u, b: vec2u) -> NumericResult {
    var result = vec2u(0u);
    var trap = 0u;
    let ai = bitcast<i32>(a.x); let bi = bitcast<i32>(b.x);
    let shift = b.x & 31u;
    switch op {
        case 0x45u: { result.x = u32(a.x == 0u); }
        case 0x46u: { result.x = u32(a.x == b.x); }
        case 0x47u: { result.x = u32(a.x != b.x); }
        case 0x48u: { result.x = u32(ai < bi); }
        case 0x49u: { result.x = u32(a.x < b.x); }
        case 0x4au: { result.x = u32(ai > bi); }
        case 0x4bu: { result.x = u32(a.x > b.x); }
        case 0x4cu: { result.x = u32(ai <= bi); }
        case 0x4du: { result.x = u32(a.x <= b.x); }
        case 0x4eu: { result.x = u32(ai >= bi); }
        case 0x4fu: { result.x = u32(a.x >= b.x); }
        case 0x50u: { result.x = u32(zero64(a)); }
        case 0x51u: { result.x = u32(eq64(a, b)); }
        case 0x52u: { result.x = u32(!eq64(a, b)); }
        case 0x53u, 0x55u, 0x57u, 0x59u: {
            let less = lt64(a ^ vec2u(0u, 0x80000000u), b ^ vec2u(0u, 0x80000000u));
            let equal = eq64(a, b);
            switch op {
                case 0x53u: { result.x = u32(less); }
                case 0x55u: { result.x = u32(!less && !equal); }
                case 0x57u: { result.x = u32(less || equal); }
                default: { result.x = u32(!less); }
            }
        }
        case 0x54u: { result.x = u32(lt64(a, b)); }
        case 0x56u: { result.x = u32(lt64(b, a)); }
        case 0x58u: { result.x = u32(!lt64(b, a)); }
        case 0x5au: { result.x = u32(!lt64(a, b)); }
        case 0x67u: { result.x = countLeadingZeros(a.x); }
        case 0x68u: { result.x = countTrailingZeros(a.x); }
        case 0x69u: { result.x = countOneBits(a.x); }
        case 0x6au: { result.x = a.x + b.x; }
        case 0x6bu: { result.x = a.x - b.x; }
        case 0x6cu: { result.x = a.x * b.x; }
        case 0x6du: {
            if b.x == 0u { trap = 4u; }
            else if a.x == 0x80000000u && b.x == 0xffffffffu { trap = 5u; }
            else { result.x = bitcast<u32>(ai / bi); }
        }
        case 0x6eu: { if b.x == 0u { trap = 4u; } else { result.x = a.x / b.x; } }
        case 0x6fu: { if b.x == 0u { trap = 4u; } else { result.x = bitcast<u32>(ai % bi); } }
        case 0x70u: { if b.x == 0u { trap = 4u; } else { result.x = a.x % b.x; } }
        case 0x71u: { result.x = a.x & b.x; }
        case 0x72u: { result.x = a.x | b.x; }
        case 0x73u: { result.x = a.x ^ b.x; }
        case 0x74u: { result.x = a.x << shift; }
        case 0x75u: { result.x = bitcast<u32>(ai >> shift); }
        case 0x76u: { result.x = a.x >> shift; }
        case 0x77u: { result.x = (a.x << shift) | (a.x >> ((32u - shift) & 31u)); }
        case 0x78u: { result.x = (a.x >> shift) | (a.x << ((32u - shift) & 31u)); }
        case 0x79u: { result.x = clz64(a); }
        case 0x7au: { result.x = select(countTrailingZeros(a.x), 32u + countTrailingZeros(a.y), a.x == 0u); }
        case 0x7bu: { result.x = countOneBits(a.x) + countOneBits(a.y); }
        case 0x7cu: { result = add64(a, b); }
        case 0x7du: { result = sub64(a, b); }
        case 0x7eu: { result = mul64(a, b); }
        case 0x7fu, 0x80u, 0x81u, 0x82u: {
            let is_signed = op == 0x7fu || op == 0x81u;
            let remainder = op >= 0x81u;
            if zero64(b) { trap = 4u; }
            else if is_signed && !remainder && eq64(a, vec2u(0u, 0x80000000u)) && all(b == vec2u(0xffffffffu)) { trap = 5u; }
            else {
                let aneg = is_signed && (a.y >> 31u) != 0u; let bneg = is_signed && (b.y >> 31u) != 0u;
                let division = div64(select(a, neg64(a), aneg), select(b, neg64(b), bneg));
                result = select(division.quotient, division.remainder, remainder);
                if select(aneg != bneg, aneg, remainder) { result = neg64(result); }
            }
        }
        case 0x83u: { result = a & b; }
        case 0x84u: { result = a | b; }
        case 0x85u: { result = a ^ b; }
        case 0x86u: { result = shl64(a, b.x & 63u); }
        case 0x87u: { result = sar64(a, b.x); }
        case 0x88u: { result = shr64(a, b.x & 63u); }
        case 0x89u: { result = shl64(a, b.x & 63u) | shr64(a, (64u - b.x) & 63u); }
        case 0x8au: { result = shr64(a, b.x & 63u) | shl64(a, (64u - b.x) & 63u); }
        case 0xa7u: { result.x = a.x; }
        case 0xacu: { result = vec2u(a.x, select(0u, 0xffffffffu, ai < 0)); }
        case 0xadu: { result.x = a.x; }
        case 0xb6u: { result = convert_float(a, false); }
        case 0xbbu: { result = convert_float(a, true); }
        case 0xbcu, 0xbeu: { result.x = a.x; }
        case 0xbdu, 0xbfu: { result = a; }
        case 0xc0u: { result.x = bitcast<u32>(bitcast<i32>(a.x << 24u) >> 24); }
        case 0xc1u: { result.x = bitcast<u32>(bitcast<i32>(a.x << 16u) >> 16); }
        case 0xc2u, 0xc3u, 0xc4u: {
            let bits = select(select(32u, 16u, op == 0xc3u), 8u, op == 0xc2u);
            result = sar64(shl64(a, 64u - bits), 64u - bits);
        }
        default: {
            if op >= 0x5bu && op <= 0x66u {
                let single = op <= 0x60u;
                let compare = float_compare(a, b, single);
                switch (op - select(0x61u, 0x5bu, single)) {
                    case 0u: { result.x = u32(compare == 0u); }
                    case 1u: { result.x = u32(compare != 0u); }
                    case 2u: { result.x = u32(compare == 1u); }
                    case 3u: { result.x = u32(compare == 2u); }
                    case 4u: { result.x = u32(compare <= 1u); }
                    default: { result.x = u32(compare == 0u || compare == 2u); }
                }
            } else if op >= 0x8bu && op <= 0xa6u {
                let single = op <= 0x98u;
                let kind = op - select(0x99u, 0x8bu, single);
                let mask = select(vec2u(0u, 0x80000000u), vec2u(0x80000000u, 0u), single);
                switch kind {
                    case 0u: { result = a & ~mask; }
                    case 1u: { result = a ^ mask; }
                    case 2u, 3u, 4u, 5u: { result = soft_round(a, kind - 2u, single); }
                    case 6u: { result = soft_sqrt(a, single); }
                    case 7u: { result = soft_add(a, b, false, single); }
                    case 8u: { result = soft_add(a, b, true, single); }
                    case 9u: { result = soft_mul(a, b, single); }
                    case 10u: { result = soft_div(a, b, single); }
                    case 11u, 12u: {
                        let compare = float_compare(a, b, single);
                        if compare == 3u { result = float_special(0u, true, single); }
                        else if compare == 0u { result = select(a & b, a | b, kind == 11u); }
                        else { result = select(b, a, compare == select(2u, 1u, kind == 11u)); }
                    }
                    default: { result = (a & ~mask) | (b & mask); }
                }
            } else if (op >= 0xb2u && op <= 0xb5u) || (op >= 0xb7u && op <= 0xbau) {
                let single = op <= 0xb5u;
                let kind = op - select(0xb7u, 0xb2u, single);
                result = integer_to_float(a, (kind & 1u) == 0u, kind >= 2u, single);
            } else {
                var single = true; var wide = false; var is_signed = true; var saturate = false;
                if op >= 0xfc00u && op <= 0xfc07u {
                    let kind = op - 0xfc00u;
                    single = (kind & 3u) < 2u; wide = kind >= 4u; is_signed = (kind & 1u) == 0u; saturate = true;
                } else if op >= 0xa8u && op <= 0xabu {
                    single = op <= 0xa9u; is_signed = (op & 1u) == 0u;
                } else if op >= 0xaeu && op <= 0xb1u {
                    single = op <= 0xafu; is_signed = (op & 1u) == 0u; wide = true;
                } else { return NumericResult(vec2u(0u), 11u); }
                let conversion = float_to_integer(a, single, wide, is_signed, saturate);
                result = conversion.value; trap = conversion.trap;
            }
        }
    }
    return NumericResult(result, trap);
}
