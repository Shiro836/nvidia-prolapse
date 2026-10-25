#!/usr/bin/env python3
"""Per-kick CPU/GPU timeline of a capture: nvcap_timeline.py CAP.bin [frame_index] [nframes]
CPU submit time is mapped onto the GPU clock with the START record's (cpu_ns, gpu_ns) pair."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nvcap
cap = nvcap.parse(nvcap.read(sys.argv[1]))
fi = int(sys.argv[2]) if len(sys.argv) > 2 else 10
nf = int(sys.argv[3]) if len(sys.argv) > 3 else 2
cpu0, gpu0 = cap['start'][0], cap['start'][1]
marks = sorted(m['cpu'] for m in cap['marks'])
a, b = marks[fi], marks[fi + nf]
prev_end = None
print(f'{"kick":>4} {"cpu_submit":>10} {"gpu_start":>10} {"gpu_end":>10} {"busy":>7} {"gap_before":>10} {"start-submit":>12} bytes')
for i, s in enumerate(cap['submits']):
    if not (a <= s['cpu'] < b):
        continue
    t = cap['times'].get((s['ctx'], s['tok']))
    sub = s['cpu'] - cpu0
    for m in marks:
        if prev_end is not None and s['cpu'] > m > prev_cpu:
            print('   --- present (fence export) at cpu', (m - cpu0) / 1e6)
    if t:
        st, en = t[0] - gpu0, t[1] - gpu0
        gap = (st - prev_end) if prev_end is not None else 0
        busy = en - max(st, prev_end or st)
        print(f'{i:4d} {sub/1e6:10.3f} {st/1e6:10.3f} {en/1e6:10.3f} {busy/1e6:7.3f} {gap/1e6:10.3f} {(st-sub)/1e6:12.3f} {sum(p[1] for p in s["pushes"])}')
        prev_end = max(en, prev_end or en)
    prev_cpu = s['cpu']
