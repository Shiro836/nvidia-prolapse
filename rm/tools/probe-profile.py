#!/usr/bin/env python3
"""Probe a tag's public OGKM headers for every layout/constant in a compat_data.rs profile.

usage: probe-profile.py <compat_data.rs> <profile, e.g. P20> <ogkm checkout> <outdir>
Writes probe.c, probe-m32.tsv, probe-m64.tsv, expected.tsv, missing.txt into outdir.
"""
import re, subprocess, sys
from pathlib import Path

data, prof, tree, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
src = data.read_text()

layouts = {}
for m in re.finditer(r'static (L\d+): Layout = Layout \{ name: "([^"]+)", size: (\d+), fields: &\[\n(.*?)\] \};', src, re.S):
    fields = re.findall(r'Field \{ name: "([^"]+)", offset: (\d+), len: (\d+) \}', m.group(4))
    layouts[m.group(1)] = (m.group(2), int(m.group(3)), [(f, int(o), int(n)) for f, o, n in fields])
pm = re.search(r'static %s: Profile = Profile \{ layouts: &\[\n(.*?)\], constants: &\[\n(.*?)\] \};' % prof, src, re.S)
entries = re.findall(r'\("([^"]+)", &(L\d+)\)', pm.group(1))
consts = [(n, int(v)) for n, v in re.findall(r'\("([^"]+)", (\d+)\)', pm.group(2))]

INCLUDES = ['src/common/sdk/nvidia/inc', 'src/common/inc', 'src/nvidia/arch/nvalloc/unix/include',
            'kernel-open/common/inc', 'kernel-open/nvidia-drm', 'src/nvidia-modeset/kapi/interface',
            'src/nvidia-modeset/interface', 'src/common/unix/common/inc']

# expected TSV, in the probe's own order
exp = []
for name, lid in entries:
    lname, size, fields = layouts[lid]
    assert lname == name
    exp.append(f'{name}\tsize={size}')
    exp += [f'{name}.{f}\toff={o} len={n}' for f, o, n in fields]
exp += [f'{n}\t0x{v:x}' for n, v in consts]
(out / 'expected.tsv').write_text('\n'.join(exp) + '\n')

def find_header(sym):
    pat = re.compile(r'(typedef\s+struct\s+\w*\s*\{[^}]*?|struct\s+%s\s*\{|#\s*define\s+%s\b|\b%s\s*[;,=])' % (sym, sym, sym))
    for inc in INCLUDES:
        for h in sorted((tree / inc).rglob('*.h')):
            t = h.read_text(errors='replace')
            if sym in t and (re.search(r'\b%s\b' % sym, t)):
                if re.search(r'#\s*define\s+%s\b' % sym, t) or re.search(r'\}\s*%s\s*;' % sym, t) or \
                   re.search(r'struct\s+%s\s*\{' % sym, t) or re.search(r'\b%s\s*(=|,)' % sym, t) or \
                   re.search(r'typedef\s+[\w\s]+\b%s\s*;' % sym, t):
                    return str(h.relative_to(tree / inc))
    return None

def build(skip_consts):
    headers = []
    for name, _ in entries:
        h = find_header(name)
        if h and h not in headers: headers.append(h)
    for n, _ in consts:
        if n in skip_consts: continue
        h = find_header(n)
        if h and h not in headers: headers.append(h)
    body = ['#include <stdio.h>', '#include <stddef.h>', '#include <stdint.h>', '#include <sys/ioctl.h>',
            '#include <nvtypes.h>', '#include <nvmisc.h>', '#include <nvstatus.h>']
    body += ['#include <%s>' % h for h in headers]
    body += ['#define S(T) printf(#T "\\tsize=%zu\\n",sizeof(T))',
             '#define F(T,n,f) printf(#T "." n "\\toff=%zu len=%zu\\n",offsetof(T,f),sizeof(((T*)0)->f))',
             '#define K(k) printf(#k "\\t0x%llx\\n",(unsigned long long)(k))', 'int main(void) {']
    for name, lid in entries:
        typ = 'struct ' + name if name.startswith(('drm_', 'NvKmsKapiPriv')) else name
        body.append(f'S({typ});'.replace('S(struct ', 'S(struct '))
        for f, _, _ in layouts[lid][2]:
            fexpr = f
            body.append(f'F({typ},"{f}",{fexpr});')
    for n, _ in consts:
        if n not in skip_consts: body.append(f'K({n});')
    body.append('return 0; }')
    return headers, '\n'.join(body) + '\n'

skip = set()
for attempt in range(12):
    headers, code = build(skip)
    (out / 'probe.c').write_text(code)
    r = subprocess.run(['cc', '-m64', '-std=gnu11', '-w', *[f'-I{tree / p}' for p in INCLUDES],
                        str(out / 'probe.c'), '-o', str(out / 'm64')], capture_output=True, text=True)
    if r.returncode == 0: break
    bad = set(re.findall(r"'(\w+)' undeclared", r.stderr)) & {n for n, _ in consts}
    if not bad:
        print(r.stderr[:4000]); sys.exit(1)
    skip |= bad
(out / 'missing.txt').write_text('\n'.join(sorted(skip)) + '\n')
(out / 'headers.txt').write_text('\n'.join(headers) + '\n')
for abi in ['m32', 'm64']:
    subprocess.run(['cc', '-' + abi, '-std=gnu11', '-w', *[f'-I{tree / p}' for p in INCLUDES],
                    str(out / 'probe.c'), '-o', str(out / abi)], check=True)
    (out / f'probe-{abi}.tsv').write_bytes(subprocess.check_output([str(out / abi)]))
print('ok', len(entries), 'layouts', len(consts), 'consts,', len(skip), 'consts not plain macros')
