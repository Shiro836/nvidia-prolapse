#!/usr/bin/env python3
"""Per-draw state table (targets, depth, blend, FS texture handles) from nvcap_seq.py output: nvcap_draws.py SEQ.txt [submit]"""
import sys, re
want = int(sys.argv[2]) if len(sys.argv) > 2 else None
st = {}; sub = -1; n = 0
keys = ['SET_DEPTH_TEST','SET_DEPTH_FUNC','SET_DEPTH_WRITE','SET_STENCIL_TEST','SET_COLOR_TARGET_A(0)','SET_ZT_A','SET_ZT_SELECT','SET_CT_WRITE(0)','SET_BLEND(0)','tex','CLEAR_SURFACE','SET_RENDER_ENABLE_C']
for line in open(sys.argv[1]):
    if line.startswith('=== submit'):
        sub = int(line.split()[2]); n = 0; st = {}; continue
    if want is not None and sub != want: continue
    m = re.match(r'\s+3D\.(\S+) \[\w+(?: x\d+)?\] ?(.*)', line)
    if not m: continue
    nm, vals = m.group(1), m.group(2).split()
    if nm in keys: st[nm] = ' '.join(vals[:3])
    if nm == 'LOAD_CONSTANT_BUFFER_OFFSET' and st.get('sel','').startswith('0x100 '): st['tex'] = ' '.join(vals[:3])
    if nm == 'SET_CONSTANT_BUFFER_SELECTOR_A': st['sel'] = ' '.join(vals[:3])
    if nm == 'CLEAR_SURFACE':
        print(f"s{sub} CLEAR {vals[0]} ct={st.get('SET_COLOR_TARGET_A(0)')} zt={st.get('SET_ZT_A')}")
    if nm == 'BEGIN':
        n += 1
        print(f"s{sub} d{n} op={vals[0]} ct={st.get('SET_COLOR_TARGET_A(0)','?').split()[1] if 'SET_COLOR_TARGET_A(0)' in st else '?'} zsel={st.get('SET_ZT_SELECT')} ztest={st.get('SET_DEPTH_TEST')} zfunc={st.get('SET_DEPTH_FUNC')} zw={st.get('SET_DEPTH_WRITE')} blend={st.get('SET_BLEND(0)')} ctw={st.get('SET_CT_WRITE(0)')} tex={st.get('tex')}")
