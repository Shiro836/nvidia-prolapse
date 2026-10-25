#!/usr/bin/env python3
"""Texture header / sampler pool check of an nvcap push capture (REC_POOL records, capture.rs):
nvcap_pools.py CAP [--dump]

Per submission: the pools it binds (3D subchannel 0, compute 1), how many 32-B entries are
non-zero, and every texture handle the batch loads into a 256-B handle window (nvrm's driver
cbuf handle table: LOAD_CONSTANT_BUFFER under SET_CONSTANT_BUFFER_SELECTOR_A size 0x100) whose
header (handle & 0xfffff) or sampler (handle >> 20) entry is all zeros in the captured pool.
A batch whose draws sample zero headers prints BAD; --dump prints the referenced entries.
"""
import os, struct, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvcap

LOAD_OFFSET = 0x238c  # LOAD_CONSTANT_BUFFER_OFFSET (cl9097.h:3606); values follow at LOAD_CONSTANT_BUFFER(i)
SELECTOR_A = 0x2380  # SET_CONSTANT_BUFFER_SELECTOR_A (cl9097.h:3597)


def handles_of(submit):
    """(window address, offset, value) for every dword loaded into a 0x100-byte window."""
    out = []
    sel = (0, 0)
    for (_a, _sz, _fl, words) in submit['pushes']:
        if not words:
            continue
        i, n = 0, len(words)
        while i < n:
            h = words[i]
            op, cnt, sc, m = h >> 29, (h >> 16) & 0x1fff, (h >> 13) & 7, (h & 0xfff) << 2
            i += 1
            if op == 4 or op not in (1, 3, 5):
                continue
            vals = words[i:i + cnt]
            i += cnt
            if sc != 0:
                continue
            if m == SELECTOR_A and op == 1 and len(vals) >= 3:
                sel = (vals[0], (vals[1] & 0x1ffffff) << 32 | vals[2])
            elif m == LOAD_OFFSET and op == 5 and vals and sel[0] == 0x100:
                off = vals[0]
                for k, v in enumerate(vals[1:]):
                    out.append((sel[1], off + 4 * k, v))
    return out


def main():
    cap = nvcap.parse(nvcap.read(sys.argv[1]))
    dump = '--dump' in sys.argv
    by_tok = {}
    for p in cap['pools']:
        by_tok.setdefault(p['tok'], []).append(p)
    bad_batches = 0
    for si, s in enumerate(cap['submits']):
        pools = {(p['sc'], p['kind']): p for p in by_tok.get(s['tok'], [])}
        if not pools:
            continue
        line = [f"submit {si} tok {s['tok']}"]
        for (sc, kind), p in sorted(pools.items()):
            d = p['data']
            nz = sum(1 for e in range(len(d) // 32) if any(d[32 * e:32 * e + 32]))
            line.append(f"{'TIC' if kind == 1 else 'TSC'}{sc}@{p['addr']:#x} {nz}/{p['entries']} non-zero"
                        + ('' if d else ' (not CPU-mapped)'))
        print('  '.join(line))
        tic, tsc = pools.get((0, 1)), pools.get((0, 2))
        if not tic or not tic['data']:
            continue
        bad = []
        seen = set()
        for (win, off, h) in handles_of(s):
            hi, si_ = h & 0xfffff, h >> 20
            if h in seen:
                continue
            seen.add(h)
            te = tic['data'][32 * hi:32 * hi + 32] if 32 * hi + 32 <= len(tic['data']) else b''
            se = tsc['data'][32 * si_:32 * si_ + 32] if tsc and 32 * si_ + 32 <= len(tsc['data']) else b''
            zero_t = not any(te)
            zero_s = bool(tsc and tsc['data']) and not any(se)
            if dump:
                print(f"    handle {h:#010x} (win {win:#x}+{off:#x}): TIC[{hi:#x}] "
                      f"{' '.join(f'{w:08x}' for w in struct.unpack('<8I', te)) if len(te) == 32 else 'out of range'}")
            if h and (zero_t or zero_s):
                bad.append((h, zero_t, zero_s))
        if bad:
            bad_batches += 1
            print(f"  BAD: {len(bad)} handles reference zero entries: "
                  + ', '.join(f"{h:#x}{' TIC' if t else ''}{' TSC' if z else ''}" for h, t, z in bad[:16]))
    print(f"{bad_batches} batches with handles on zero pool entries")
    return 1 if bad_batches else 0


if __name__ == '__main__':
    sys.exit(main())
