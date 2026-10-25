#!/usr/bin/env python3
"""Ordered method dump of an nvcap capture (Rust RM capture hook, push mode): nvcap_seq.py CAP [submit_index...]"""
import sys, os, struct
os.environ.setdefault('NV_CLASS_HEADERS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../build/mesa-rm-source/src/nouveau/headers/nvidia/classes/'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvcap
cap = nvcap.parse(nvcap.read(sys.argv[1]))
sel = set(int(x) for x in sys.argv[2:])
for si, s in enumerate(cap['submits']):
    print(f"=== submit {si} ctx {s['ctx']:#x} tok {s['tok']} waits {s['waits']} pushes {len(s['pushes'])}")
    if sel and si not in sel:
        continue
    for (a, sz, fl, words) in s['pushes']:
        print(f"  -- push {a:#x} {sz}B flags {fl:#x}")
        if not words: continue
        i = 0; n = len(words)
        while i < n:
            h = words[i]
            if h == 0: i += 1; continue
            op = h >> 29; cnt = (h >> 16) & 0x1fff; sc = (h >> 13) & 7; m = (h & 0xfff) << 2
            if op == 4: vals = [cnt]; i += 1; kind = 'IMM'
            elif op in (1, 3, 5): vals = list(words[i+1:i+1+cnt]); i += 1 + cnt; kind = {1:'INC',3:'NON',5:'ONE'}[op]
            else: print(f"    raw op{op} {h:#x}"); i += 1; continue
            if m < 0x100: nm = 'HOST.' + nvcap.HOST.get(m, f'{m:#x}')
            else:
                cname, table = nvcap.SUBC.get(sc, (f'sc{sc}', {}))
                nm = cname + '.' + table.get(m, f'{m:#06x}')
            if kind == 'NON' and len(vals) > 4:
                print(f"    {nm} [{kind} x{len(vals)}] {' '.join(f'{v:#x}' for v in vals[:4])} ...")
            else:
                print(f"    {nm} [{kind}] {' '.join(f'{v:#x}' for v in vals[:12])}{' ...' if len(vals)>12 else ''}")
