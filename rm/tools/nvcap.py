#!/usr/bin/env python3
"""Decode a Rust RM perf capture (native/src/capture.rs) into per-frame tables.

usage: nvcap.py CAP.bin [--driver nvk|nvrm] [--json OUT] [--frames-skip N] [--top N]

File: magic b"NVCAP01\\0", then records (u32 tag, u32 len, payload):
  1 START   cpu_ns u64, gpu_ns u64, pid u32, push u32
  2 SUBMIT  cpu_ns u64, context u64, token u64, waits u32, foreign u32, signals u32,
            pushes u32, then per push: address u64, size u32, flags u32, nbytes u32, bytes
  3 TIME    context u64, token u64, start_ns u64, end_ns u64   (GPU PTIMER)
  4 MARK    cpu_ns u64, context u64, token u64, kind u32 (1 context fence, 2 sync-file
            export, 3 opaque export), pad u32
  5 END     cpu_ns u64, reason u32, pad u32
  6 CONTEXT context u64, engines u32, channel class u32
  7 POOL    token u64, address u64, subchannel u32, kind u32 (1 texture header pool,
            2 sampler pool), entries u32, nbytes u32, bytes (pool contents as the CPU wrote
            them, read when the batch binding it was submitted; push mode)

Frames = intervals between consecutive MARK records (fence exports = presents).
Method names come from the pinned Mesa class headers (NVIDIA open headers).
"""
import collections, json, re, struct, sys, os

HDR = os.environ.get('NV_CLASS_HEADERS', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../../build/mesa-nak-source/src/nouveau/headers/nvidia/classes/'))


def names(f, pfx):
    d = {}
    arr = []
    try:
        lines = open(HDR + f).read().split('\n')
    except OSError:
        return d
    for l in lines:
        m = re.match(r'#define ' + pfx + r'(\w+)\s+\(?(0x[0-9a-fA-F]+)\)?\s*$', l)
        if m and not re.search(r'(_V|_VALUE)$|_V_', m[1]) and int(m[2], 16) % 4 == 0 and int(m[2], 16) < 0x8000 \
                and (len(m[2]) == 6 or len(m[2]) == 10):
            d.setdefault(int(m[2], 16), m[1])
        m = re.match(r'#define ' + pfx + r'(\w+)\((\w+)\)\s+\((0x[0-9a-fA-F]+)\+\(\w+\)\*(\d+)\)', l)
        if m:
            arr.append((int(m[3], 16), int(m[4]), m[1]))
    bases = sorted(set(d) | {b for b, _, _ in arr})
    for b, st, nm in arr:
        # expand up to the first method block at or beyond the second element
        later = [x for x in bases if x >= b + st]
        limit = min(later) if later else b + 256 * st
        j = 0
        while b + j * st < limit and j < 256:
            d.setdefault(b + j * st, f"{nm}({j})")
            j += 1
    return d


CLS3D = names('clce97.h', 'NVCE97_')
for f, p in (('clcb97.h', 'NVCB97_'), ('clc797.h', 'NVC797_'), ('clc597.h', 'NVC597_'), ('cl9097.h', 'NV9097_')):
    for k, v in names(f, p).items():
        CLS3D.setdefault(k, v)
CLSCOMP = names('clcec0.h', 'NVCEC0_')
CLS2D = names('cl902d.h', 'NV902D_')
CLSCOPY = names('clc9b5.h', 'NVC9B5_')
CLSI2M = names('cla040.h', 'NVA040_')
HOST = names('clc86f.h', 'NVC86F_')
# Subchannels as bound by NVK (nvk_queue.c) and nvrm (nvrm_screen.c): 0 3D, 1 compute,
# 2 inline-to-memory, 3 2D, 4 copy.
SUBC = {0: ('3D', CLS3D), 1: ('COMP', CLSCOMP), 2: ('I2M', CLSI2M), 3: ('2D', CLS2D), 4: ('CE', CLSCOPY)}

NVK_MME = ['SELECT_CB0', 'BIND_CBUF_DESC', 'CLEAR', 'BIND_IB', 'BIND_VB', 'SET_VB_ENABLES', 'SET_VB_STRIDE',
           'SET_TESS_PARAMS', 'SET_SHADING_RATE_CONTROL', 'SET_ANTI_ALIAS', 'DRAW', 'DRAW_INDEXED',
           'DRAW_INDIRECT', 'DRAW_INDEXED_INDIRECT', 'DRAW_INDIRECT_COUNT', 'DRAW_INDEXED_INDIRECT_COUNT',
           'ADD_CS_INVOCATIONS', 'DISPATCH_INDIRECT', 'WRITE_CS_INVOCATIONS', 'XFB_COUNTER_LOAD',
           'XFB_DRAW_INDIRECT', 'SET_PRIV_REG', 'SET_WRITE_MASK', 'SET_CONSERVATIVE_RASTER_STATE',
           'SET_VIEWPORT_MIN_MAX_Z', 'SET_Z_CLAMP']  # nvk_mme.h enum nvk_mme (pinned Mesa)
NVRM_MME = ['DRAW_INDIRECT', 'DRAW_INDEXED_INDIRECT', 'DISPATCH_INDIRECT', 'SET_PRIV_REG', 'DRAW',
            'DRAW_INDEXED']  # nvrm_priv.h


def read(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'NVCAP01\0', 'not a capture'
    off = 8
    recs = []
    while off + 8 <= len(data):
        tag, ln = struct.unpack_from('<II', data, off)
        off += 8
        if off + ln > len(data):
            break
        recs.append((tag, data[off:off + ln]))
        off += ln
    return recs


def parse(recs):
    out = {'contexts': {}, 'submits': [], 'times': {}, 'marks': [], 'start': None, 'end': None,
           'pools': []}
    for tag, p in recs:
        if tag == 1:
            out['start'] = struct.unpack_from('<QQII', p)
        elif tag == 6:
            c, e, k = struct.unpack_from('<QII', p)
            out['contexts'][c] = (e, k)
        elif tag == 2:
            cpu, ctx, tok, w, f, s, n = struct.unpack_from('<QQQIIII', p)
            o = 40
            pushes = []
            for _ in range(n):
                a, sz, fl, nb = struct.unpack_from('<QIII', p, o)
                o += 20
                words = struct.unpack_from(f'<{nb // 4}I', p, o) if nb else None
                o += nb
                pushes.append((a, sz, fl, words))
            out['submits'].append(dict(cpu=cpu, ctx=ctx, tok=tok, waits=w, foreign=f, signals=s, pushes=pushes))
        elif tag == 3:
            c, t, s, e = struct.unpack_from('<QQQQ', p)
            out['times'][(c, t)] = (s, e)
        elif tag == 4:
            cpu, c, t, k, _ = struct.unpack_from('<QQQII', p)
            out['marks'].append(dict(cpu=cpu, ctx=c, tok=t, kind=k))
        elif tag == 5:
            out['end'] = struct.unpack_from('<QI', p)
        elif tag == 7:
            tok, addr, sc, kind, entries, nb = struct.unpack_from('<QQIIII', p)
            out['pools'].append(dict(tok=tok, addr=addr, sc=sc, kind=kind, entries=entries,
                                     data=bytes(p[32:32 + nb])))
    return out


def decode_words(words, mme_names, stats, push_flags):
    """Walk Volta+ pushbuffer methods; update a Counter `stats`."""
    i = 0
    n_words = len(words)
    last = {}
    while i < n_words:
        h = words[i]
        if h == 0:
            i += 1
            continue
        op = h >> 29
        cnt = (h >> 16) & 0x1fff
        sc = (h >> 13) & 7
        m = (h & 0xfff) << 2
        if op == 4:  # IMMD
            vals = [cnt]
            i += 1
            kind = 'IMM'
        elif op in (1, 3, 5):
            vals = words[i + 1:i + 1 + cnt]
            i += 1 + cnt
            kind = {1: 'INC', 3: 'NON', 5: 'ONE'}[op]
        else:
            stats['raw_op%d' % op] += 1
            i += 1
            continue
        for k, v in enumerate(vals):
            mm = m if kind in ('NON', 'IMM') else (m if k == 0 else m + 4) if kind == 'ONE' else m + 4 * k
            if mm < 0x100:
                nm = 'HOST.' + HOST.get(mm, f'{mm:#x}')
            else:
                cname, table = SUBC.get(sc, (f'sc{sc}', {}))
                nm = cname + '.' + table.get(mm, f'{mm:#06x}')
            stats['m:' + nm] += 1
            stats['method_writes'] += 1
            classify(nm, mm, v, sc, last, stats, mme_names)


def classify(nm, mm, v, sc, last, st, mme_names):
    base = nm.split('.', 1)[1]
    cls = nm.split('.', 1)[0]
    if base.startswith('CALL_MME_MACRO('):
        idx = int(base[len('CALL_MME_MACRO('):-1])
        mname = mme_names[idx] if idx < len(mme_names) else f'#{idx}'
        st['mme:' + mname] += 1
        if mname.startswith('DRAW'):
            st['draws'] += 1
        return
    if base.startswith('CALL_MME_DATA'):
        return
    if cls == '3D' and base in ('END', 'DRAW_VERTEX_ARRAY_BEGIN_END', 'DRAW_INDEX_BUFFER_BEGIN_END',
                                'DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_FIRST', 'DRAW_VERTEX_ARRAY_BEGIN_END_INSTANCE_SUBSEQUENT',
                                'DRAW_INDEX_BUFFER_BEGIN_END_INSTANCE_FIRST', 'DRAW_INDEX_BUFFER_BEGIN_END_INSTANCE_SUBSEQUENT',
                                'DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_FIRST', 'DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_FIRST',
                                'DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_FIRST', 'DRAW_INDEX_BUFFER32_BEGIN_END_INSTANCE_SUBSEQUENT',
                                'DRAW_INDEX_BUFFER16_BEGIN_END_INSTANCE_SUBSEQUENT', 'DRAW_INDEX_BUFFER8_BEGIN_END_INSTANCE_SUBSEQUENT'):
        st['draws'] += 1
    if cls == 'COMP' and base == 'SEND_SIGNALING_PCAS2_B' or (cls == 'COMP' and base.startswith('SEND_PCAS')):
        st['dispatches'] += 1
    if 'WAIT_FOR_IDLE' in base or base == 'WFI' or base == 'SET_REFERENCE':
        st['wfi:' + nm] += 1
        st['wfi'] += 1
    if cls == 'HOST' and base == 'SEM_EXECUTE':
        st['host_sem_acquire' if (v & 0x1f) in (2, 3, 4) else 'host_sem_release'] += 1
    if cls == '3D' and base == 'SET_REPORT_SEMAPHORE_D':
        st['3d_report_sem'] += 1
    if 'INVALIDATE' in base or 'FLUSH' in base or base.startswith('MEM_OP'):
        st['inv:' + nm] += 1
        st['invalidates'] += 1
    if cls == 'CE':
        if base in ('LINE_LENGTH_IN', 'LINE_COUNT'):
            last[base] = v
        if base == 'LAUNCH_DMA':
            st['ce_launch'] += 1
            remap = 0
            st['ce_bytes'] += last.get('LINE_LENGTH_IN', 0) * max(1, last.get('LINE_COUNT', 1))
    if base in ('LINE_LENGTH_IN',) and cls in ('3D', 'COMP', 'I2M'):
        last['i2m_len'] = v
        last['i2m_cnt'] = 1
    if base == 'LINE_COUNT' and cls in ('3D', 'COMP', 'I2M'):
        last['i2m_cnt'] = v
    if base == 'LAUNCH_DMA' and cls in ('3D', 'COMP', 'I2M'):
        st['i2m_launch'] += 1
        st['i2m_bytes'] += last.get('i2m_len', 0) * max(1, last.get('i2m_cnt', 1))
    if base == 'LOAD_INLINE_DATA':
        st['inline_data_words'] += 1
    if base.startswith('LOAD_CONSTANT_BUFFER(') or base == 'LOAD_CONSTANT_BUFFER':
        st['cbuf_load_words'] += 1
    if base.startswith('LOAD_CONSTANT_BUFFER_OFFSET'):
        st['cbuf_loads'] += 1
    if base.startswith('SET_CONSTANT_BUFFER_SELECTOR_A'):
        st['cbuf_select'] += 1
    if base.startswith('BIND_GROUP_CONSTANT_BUFFER'):
        st['cbuf_bind'] += 1
    if base.startswith('SET_PIPELINE_PROGRAM') or base.startswith('SET_PIPELINE_SHADER') or base.startswith('SET_PIPELINE_PROGRAM_ADDRESS'):
        st['shader_set'] += 1
    if base.startswith('SET_TEX_HEADER_POOL') or base.startswith('SET_TEX_SAMPLER_POOL'):
        st['tic_tsc_pool'] += 1
    if base.startswith('SET_COLOR_TARGET_A(') or base == 'SET_ZT_A':
        st['rt_bind'] += 1
    if base == 'SET_CT_SELECT':
        st['rt_select'] += 1
    if base.startswith('CLEAR_SURFACE'):
        st['clears'] += 1
    if cls == 'COMP' and base.startswith('SEND_PCAS_A'):
        st['dispatches'] += 1


def frames_of(cap):
    marks = sorted(cap['marks'], key=lambda m: m['cpu'])
    # one present = one mark; nvrm's context fence and NVK's sync-file export
    bounds = [m['cpu'] for m in marks]
    frames = []
    for a, b in zip(bounds, bounds[1:]):
        frames.append((a, b))
    return frames


def gpu_union(intervals):
    intervals = sorted(intervals)
    busy = 0
    gaps = []
    cur_s = cur_e = None
    for s, e in intervals:
        if cur_e is None:
            cur_s, cur_e = s, e
        elif s > cur_e:
            busy += cur_e - cur_s
            gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy, gaps


def main():
    args = sys.argv[1:]
    path = args[0]
    driver = 'nvk'
    skip = 2
    top = 40
    jout = None
    if '--driver' in args:
        driver = args[args.index('--driver') + 1]
    if '--frames-skip' in args:
        skip = int(args[args.index('--frames-skip') + 1])
    if '--top' in args:
        top = int(args[args.index('--top') + 1])
    if '--json' in args:
        jout = args[args.index('--json') + 1]
    mme_names = NVK_MME if driver == 'nvk' else NVRM_MME
    cap = parse(read(path))
    frames = frames_of(cap)
    frames = frames[skip:-1] if len(frames) > skip + 1 else frames
    if not frames and '--per-submit' in args:
        frames = [(s['cpu'], s['cpu'] + 1) for s in cap['submits']]
    elif not frames:  # no presents (host tests): the whole capture is one interval
        frames = [(0, 1 << 63)]
    print(f'# {path}: driver {driver}, contexts {cap["contexts"]}, submits {len(cap["submits"])}, '
          f'marks {len(cap["marks"])} (kinds {collections.Counter(m["kind"] for m in cap["marks"])}), '
          f'times {len(cap["times"])}, frames analysed {len(frames)}')
    per_frame = []
    for (a, b) in frames:
        st = collections.Counter()
        subs = [s for s in cap['submits'] if a <= s['cpu'] < b]
        ivs = []
        sub_times = []
        for s in subs:
            st['kicks'] += 1
            st['kicks_ctx%d' % s['ctx']] += 1
            st['gpfifo_entries'] += len(s['pushes'])
            st['waits'] += s['waits'] + s['foreign']
            st['signals'] += s['signals']
            for (addr, sz, fl, words) in s['pushes']:
                st['push_bytes'] += sz
                if words is None:
                    st['uncaptured_pushes'] += 1
                    st['uncaptured_bytes'] += sz
                    continue
                decode_words(words, mme_names, st, fl)
            t = cap['times'].get((s['ctx'], s['tok']))
            if t:
                ivs.append(t)
                sub_times.append((t, s))
        busy, gaps = gpu_union(ivs)
        st['gpu_busy_ns'] = busy
        st['gpu_gap_ns'] = sum(gaps)
        st['gpu_gaps'] = len(gaps)
        if ivs:
            st['gpu_span_ns'] = max(e for s, e in ivs) - min(s for s, e in ivs)
        st['cpu_frame_ns'] = b - a
        st['timed_kicks'] = len(ivs)
        per_frame.append(st)
    if not per_frame:
        print('no frames')
        return
    keys = sorted(set().union(*[set(f) for f in per_frame]))
    avg = {k: sum(f[k] for f in per_frame) / len(per_frame) for k in keys}
    summary = {k: avg[k] for k in keys if not k.startswith('m:')}
    for k in sorted(summary):
        v = summary[k]
        if k.endswith('_ns'):
            print(f'{k:50s} {v / 1e6:12.3f} ms')
        else:
            print(f'{k:50s} {v:12.1f}')
    print(f'# top {top} methods per frame')
    meth = sorted(((avg[k], k) for k in keys if k.startswith('m:')), reverse=True)[:top]
    for v, k in meth:
        print(f'  {k[2:]:60s} {v:10.1f}')
    # GPU time per submission over the whole capture (sorted by end), attributed
    allt = sorted((cap['times'][(s['ctx'], s['tok'])], s) for s in cap['submits'] if (s['ctx'], s['tok']) in cap['times'])
    prev_end = 0
    durs = []
    for (st_, en), s in allt:
        d = en - max(st_, prev_end)
        prev_end = max(prev_end, en)
        durs.append(d)
    if jout:
        json.dump({'driver': driver, 'frames': len(per_frame), 'avg': avg,
                   'per_frame': [dict(f) for f in per_frame]}, open(jout, 'w'), indent=1)


if __name__ == '__main__':
    main()
