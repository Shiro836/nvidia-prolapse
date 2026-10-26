#!/usr/bin/env python3
"""Generate video/src/av1/cdf_defaults.rs from the public AV1 default CDF tables.

Usage: gen_cdf_defaults.py <mesa>/src/amd/common/ac_vcn_av1_default.h  (Mesa 25.2.0, 0b97dcd; MIT)
Step 1 compiles a tiny C dumper against the header (tables.txt), step 2 maps them into the NVDEC layout."""
import re,sys,os,subprocess,tempfile,struct
HDR=sys.argv[1] if len(sys.argv)>1 else None
def dump_tables(H, outp):
    src=open(H).read()
    decls=re.findall(r'static const uint16_t (\w+)\s*((?:\[[^\]]+\]\s*)+)=',src)
    out=['#include <stdint.h>','#include <stdio.h>','#define CLAMP(a,b,c) a','#define MIN2(a,b) a','#define MAX2(a,b) a','#include "%s"'%H,'int main(){']
    for name,dims in decls:
        d=re.findall(r'\[([^\]]+)\]',dims)
        out.append('{const uint16_t*p=(const uint16_t*)%s; int n=sizeof(%s)/2; printf("%s %%d %%d",n,(int)(%s)); for(int i=0;i<n;i++)printf(" %%d",p[i]); printf("\\n");}'%(name,name,name,d[-1]))
    # nmv context: emit as struct of rows
    out.append('{const uint16_t*p=(const uint16_t*)&default_nmv_context; int n=sizeof(default_nmv_context)/2; printf("default_nmv_context %d 0",n); for(int i=0;i<n;i++)printf(" %d",p[i]); printf("\\n");}')
    out.append('return 0;}')
    open(os.path.join(td,'dump.c'),'w').write('\n'.join(out))
    subprocess.check_call(['gcc','-w','-o',os.path.join(td,'dump'),os.path.join(td,'dump.c')])
    open(outp,"w").write(subprocess.check_output([os.path.join(td,'dump')]).decode())
def tables(path='tables.txt'):
    T={}
    for l in open(path):
        p=l.split(); name=p[0]; n=int(p[1]); inner=int(p[2]); vals=list(map(int,p[3:]))
        rows=[]
        if inner==0:
            i=0
            while i<len(vals):
                r=[]
                while vals[i]!=0: r.append(vals[i]); i+=1
                i+=2; rows.append(r)
        else:
            for k in range(0,n,inner):
                c=vals[k:k+inner]; r=[]
                for x in c:
                    if x==0: break
                    r.append(x)
                rows.append(r)
        T[name]=rows
    return T
if __name__=="__main__":
    td=tempfile.mkdtemp(); dump_tables(HDR, os.path.join(td,"tables.txt")); T=tables(os.path.join(td,"tables.txt"))
else:
    T=tables()
def rows(name,a=None,b=None,q=None,ctx=None):
    r=T[name]
    if q is not None:
        k=len(r)//4; r=r[q*k:(q+1)*k]
    return r[a:b]
def cdf(r): return [32768-v for v in r]   # icdf -> spec cdf
# sub-tables: name -> (rows-func(q) ), vals
SUB={}
def sub(n,vals,f): SUB[n]=(vals,f)
sub('PARTITION_W8',3,lambda q:rows('default_partition_cdf',0,4))
sub('PARTITION_W16_W64',9,lambda q:rows('default_partition_cdf',4,16))
sub('PARTITION_W128',7,lambda q:rows('default_partition_cdf',16,20))
sub('RESTORATION_WIENER',1,lambda q:rows('default_wiener_restore_cdf'))
sub('RESTORATION_SGRPROJ',1,lambda q:rows('default_sgrproj_restore_cdf'))
sub('RESTORATION_SWITCHABLE',2,lambda q:rows('default_switchable_restore_cdf'))
sub('INTRA_FRAME_Y_MODE',12,lambda q:rows('default_kf_y_mode_cdf'))
sub('SEGMENT_ID_PREDICTED',1,lambda q:rows('default_segment_pred_cdf'))
sub('SEGMENT_ID',7,lambda q:rows('default_spatial_pred_seg_tree_cdf'))
sub('SKIP',1,lambda q:rows('default_skip_cdfs'))
sub('DELTA_Q',3,lambda q:rows('default_delta_q_cdf'))
sub('DELTA_LF',3,lambda q:rows('default_delta_lf_cdf'))
sub('DELTA_LF_MULTI',3,lambda q:rows('default_delta_lf_multi_cdf'))
sub('SKIP_MODE',1,lambda q:rows('default_skip_mode_cdfs'))
sub('TXFM_SPLIT',1,lambda q:rows('default_txfm_partition_cdf'))
sub('TX_8X8',1,lambda q:rows('default_tx_size_cdf',0,3))
sub('TX_16X16_64X64',2,lambda q:rows('default_tx_size_cdf',3,12))
sub('Y_MODE',12,lambda q:rows('default_if_y_mode_cdf'))
sub('UV_MODE_CFL_NOT_ALLOWED',12,lambda q:rows('default_uv_mode_cdf',0,13))
sub('UV_MODE_CFL_ALLOWED',13,lambda q:rows('default_uv_mode_cdf',13,26))
sub('IS_INTER',1,lambda q:rows('default_intra_inter_cdf'))
sub('COMP_MODE',1,lambda q:rows('default_comp_inter_cdf'))
sub('SINGLE_REF',1,lambda q:rows('default_single_ref_cdf'))
sub('COMP_REF_TYPE',1,lambda q:rows('default_comp_ref_type_cdf'))
sub('UNI_COMP_REF',1,lambda q:rows('default_uni_comp_ref_cdf'))
sub('COMP_REF',1,lambda q:rows('default_comp_ref_cdf'))
sub('COMP_BWDREF',1,lambda q:rows('default_comp_bwdref_cdf'))
sub('NEW_MV',1,lambda q:rows('default_newmv_cdf'))
sub('ZERO_MV',1,lambda q:rows('default_zeromv_cdf'))
sub('REF_MV',1,lambda q:rows('default_refmv_cdf'))
sub('DRL_MODE',1,lambda q:rows('default_drl_cdf'))
sub('INTERP_FILTER',2,lambda q:rows('default_switchable_interp_cdf'))
N=T['default_nmv_context']
sub('MV_JOINT',3,lambda q:[N[0]])
sub('MV_CLASS',10,lambda q:[N[1]])
sub('MV_CLASS0_FR',3,lambda q:N[2:4])
sub('MV_FR',3,lambda q:[N[4]])
sub('MV_SIGN',1,lambda q:[N[5]])
sub('MV_CLASS0_HP',1,lambda q:[N[6]])
sub('MV_HP',1,lambda q:[N[7]])
sub('MV_CLASS0_BIT',1,lambda q:[N[8]])
sub('MV_BIT',1,lambda q:N[9:19])
sub('USE_OBMC',1,lambda q:rows('default_obmc_cdf'))
sub('MOTION_MODE',2,lambda q:rows('default_motion_mode_cdf'))
sub('COMPOUND_MODE',7,lambda q:rows('default_inter_compound_mode_cdf'))
sub('COMPOUND_TYPE',1,lambda q:rows('default_compound_type_cdf'))
sub('INTERINTRA',1,lambda q:rows('default_interintra_cdf'))
sub('INTERINTRA_MODE',3,lambda q:rows('default_interintra_mode_cdf'))
sub('WEDGE_INTERINTRA',1,lambda q:rows('default_wedge_interintra_cdf'))
sub('WEDGE_INDEX',15,lambda q:rows('default_wedge_idx_cdf'))
sub('PALETTE_Y_MODE',1,lambda q:rows('default_palette_y_mode_cdf',0,21))
sub('PALETTE_UV_MODE',1,lambda q:rows('default_palette_uv_mode_cdf'))
sub('PALETTE_Y_SIZE',6,lambda q:rows('default_palette_y_size_cdf',0,7))
sub('PALETTE_UV_SIZE',6,lambda q:rows('default_palette_uv_size_cdf',0,7))
sub('CFL_SIGN',7,lambda q:rows('default_cfl_sign_cdf'))
sub('CFL_ALPHA',15,lambda q:rows('default_cfl_alpha_cdf'))
sub('INTRABC',1,lambda q:rows('default_intrabc_cdf'))
sub('ANGLE_DELTA',6,lambda q:rows('default_angle_delta_cdf'))
sub('FILTER_INTRA_MODE',4,lambda q:rows('default_filter_intra_mode_cdf'))
sub('FILTER_INTRA',1,lambda q:rows('default_filter_intra_cdfs'))
sub('COMP_GROUP_IDX',1,lambda q:rows('default_comp_group_idx_cdfs'))
sub('COMPOUND_IDX',1,lambda q:rows('default_compound_idx_cdfs'))
for s in range(2,9):
    sub(f'PALETTE_SIZE_{s}_Y_COLOR',s-1,lambda q,s=s:rows('default_palette_y_color_index_cdf',(s-2)*5,(s-1)*5))
    sub(f'PALETTE_SIZE_{s}_UV_COLOR',s-1,lambda q,s=s:rows('default_palette_uv_color_index_cdf',(s-2)*5,(s-1)*5))
sub('INTRA_TX_SET1',6,lambda q:rows('default_intra_ext_tx_cdf',52,104))
sub('INTRA_TX_SET2',4,lambda q:rows('default_intra_ext_tx_cdf',104,156))
sub('INTER_TX_SET1',15,lambda q:rows('default_inter_ext_tx_cdf',4,8))
sub('INTER_TX_SET2',11,lambda q:rows('default_inter_ext_tx_cdf',8,12))
sub('INTER_TX_SET3',1,lambda q:rows('default_inter_ext_tx_cdf',12,16))
# coefficient (q-context) tables
QSUB=set()
def qsub(n,vals,f): SUB[n]=(vals,f); QSUB.add(n)
qsub('TXB_SKIP',1,lambda q:rows('av1_default_txb_skip_cdfs',q=q))
def eobx(q):
    r=rows('av1_default_eob_extra_cdfs',q=q); return [r[i*22+c] for i in range(10) for c in range(3,12)]
qsub('EOB_EXTRA',1,eobx)
for i,n in enumerate([16,32,64,128,256,512,1024]):
    qsub(f'EOB_PT_{n}',4+i,lambda q,n=n:rows(f'av1_default_eob_multi{n}_cdfs',q=q))
qsub('COEFF_BASE_EOB',2,lambda q:rows('av1_default_coeff_base_eob_multi_cdfs',q=q))
qsub('COEFF_BASE',3,lambda q:rows('av1_default_coeff_base_multi_cdfs',q=q))
qsub('DC_SIGN',1,lambda q:rows('av1_default_dc_sign_cdfs',q=q))
qsub('COEFF_BR',3,lambda q:rows('av1_default_coeff_lps_multi_cdfs',q=q))
# placements: (byte offset, table, row stride in bytes)
P=[(0x000,'PARTITION_W8',6),(0x018,'RESTORATION_WIENER',2),(0x01a,'RESTORATION_SGRPROJ',2),(0x01c,'RESTORATION_SWITCHABLE',4),
   (0x020,'PARTITION_W16_W64',0x20),(0x132,'PARTITION_W128',0x20),(0x1a0,'INTRA_FRAME_Y_MODE',24),
   (0x3f8,'SEGMENT_ID_PREDICTED',2),(0x3fe,'SEGMENT_ID',14),(0x428,'SKIP',2),(0x42e,'DELTA_Q',6),(0x434,'DELTA_LF',6),
   (0x43a,'DELTA_LF_MULTI',6),(0x452,'SKIP_MODE',2),(0x458,'TXFM_SPLIT',2),(0x484,'TX_8X8',4),(0x490,'TX_16X16_64X64',4),
   (0x4b4,'Y_MODE',24),(0x514,'UV_MODE_CFL_NOT_ALLOWED',26),(0x51e+0x148,'UV_MODE_CFL_ALLOWED',26),
   (0x7b8,'IS_INTER',2),(0x7c0,'COMP_MODE',2),(0x7ca,'SINGLE_REF',2),(0x7ee,'COMP_REF_TYPE',2),(0x7f8,'UNI_COMP_REF',2),
   (0x80a,'COMP_REF',2),(0x81c,'COMP_BWDREF',2),(0x828,'NEW_MV',2),(0x834,'ZERO_MV',2),(0x838,'REF_MV',2),(0x84a,'DRL_MODE',2),
   (0x850,'INTERP_FILTER',4)]
for c in range(2):
    P+=[(0x890,'MV_JOINT',6)] if c==0 else []
    P+=[(0x896+2*c,'MV_SIGN',2),(0x89a+0x14*c,'MV_CLASS',20),(0x8c2+0xc*c,'MV_CLASS0_FR',6),(0x8da+6*c,'MV_FR',6),
        (0x8e6+2*c,'MV_CLASS0_HP',2),(0x8ea+2*c,'MV_HP',2),(0x8ee+2*c,'MV_CLASS0_BIT',2),(0x8f2+0x14*c,'MV_BIT',2)]
P+=[(0x91a,'USE_OBMC',2),(0x946,'MOTION_MODE',4),(0x99e,'COMPOUND_MODE',14),(0xa70,'COMPOUND_TYPE',2),(0xa9c,'INTERINTRA',2),
    (0xaa4,'INTERINTRA_MODE',6),(0xabc,'WEDGE_INTERINTRA',2),(0xae8,'WEDGE_INDEX',30),(0xd7c,'PALETTE_Y_MODE',2),
    (0xda6,'PALETTE_UV_MODE',2),(0xdaa,'PALETTE_Y_SIZE',12),(0xdfe,'PALETTE_UV_SIZE',12),(0xe52,'CFL_SIGN',14),
    (0xe60,'CFL_ALPHA',30),(0xf14,'INTRABC',2),(0xf16,'ANGLE_DELTA',12),(0xf76,'FILTER_INTRA_MODE',8),(0xf7e,'FILTER_INTRA',2),
    (0xfaa,'COMP_GROUP_IDX',2),(0xfb8,'COMPOUND_IDX',2)]
for base,pl in ((0xfe0,'Y'),(0x1100,'UV')):
    for s,o in ((2,0),(3,0x50),(4,0xa0)):
        P+=[(base+o,f'PALETTE_SIZE_{s}_{pl}_COLOR',16),(base+o+2*(s-1),f'PALETTE_SIZE_{10-s}_{pl}_COLOR',16)]
    P+=[(base+0xf0,f'PALETTE_SIZE_5_{pl}_COLOR',8)]
P+=[(0x1220,'INTRA_TX_SET1',16),(0x1560,'INTRA_TX_SET2',8),(0x1700,'INTER_TX_SET1',32),(0x1780,'INTER_TX_SET2',32),(0x1796,'INTER_TX_SET3',32),
    (0x1800,'TXB_SKIP',2),(0x1882,'EOB_EXTRA',2),(0x1940,'EOB_PT_16',8),(0x1960,'EOB_PT_32',16),(0x19a0,'EOB_PT_64',16),(0x19e0,'EOB_PT_128',16),
    (0x1a20,'EOB_PT_256',16),(0x1a60,'EOB_PT_512',32),(0x1ae0,'EOB_PT_1024',32),(0x1b60,'COEFF_BASE_EOB',4),(0x1c00,'COEFF_BASE',8),
    (0x2920,'DC_SIGN',2),(0x2930,'COEFF_BR',8)]
o=[]
o.append('''// SPDX-License-Identifier: MIT
// GENERATED by native/tools/av1/gen_cdf_defaults.py -- do not edit.
//
// AV1 default CDF tables (AV1 bitstream spec section 9.4 / libaom entropy defaults), stored in the
// spec's CDF form without the trailing 32768 terminator and adaptation counter. Values transcribed by the
// generator from Mesa 25.2.0 (commit 0b97dcd) src/amd/common/ac_vcn_av1_default.h (MIT, AMD), which carries
// the libaom tables in AOM_ICDF form; the generator inverts them back (cdf = 32768 - icdf).
//
// NVDEC layout (PLACEMENTS): byte offset of each table inside the 0x3000-byte CDF buffer, row stride in
// bytes; NVDEC stores each value as little-endian u16 in inverse form (32768 - cdf). Derived by matching
// these public tables against the default CDF buffers stock libnvcuvid 615.71.09 hands to NVDEC on an
// RTX 5090 (one buffer per q-context); the result reproduces all 4 x 12288 bytes. Interop layout only.
#![allow(clippy::unreadable_literal)]
''')
def arr(v): 
    s=[]; 
    for i in range(0,len(v),16): s.append('    '+', '.join(str(x) for x in v[i:i+16])+',')
    return '\n'.join(s)
for name,(vals,f) in SUB.items():
    if name in QSUB:
        qs=[[32768-x for r in f(q) for x in r] for q in range(4)]
        n=len(qs[0])
        o.append(f'/// {vals} value(s) per row, {n//vals} rows per q-context.')
        o.append(f'pub static {name}: [[u16; {n}]; 4] = [')
        for q in qs: o.append('  [\n'+arr(q)+'\n  ],')
        o.append('];')
    else:
        v=[32768-x for r in f(0) for x in r]
        o.append(f'/// {vals} value(s) per row, {len(v)//vals} rows.')
        o.append(f'pub static {name}: [u16; {len(v)}] = [\n'+arr(v)+'\n];')
o.append('''
/// Source of one placement: a q-invariant table or one of the four q-context variants.
#[derive(Clone, Copy)]
pub enum Src {
    Fixed(&'static [u16]),
    PerQ([&'static [u16]; 4]),
}

/// One table inside the NVDEC CDF buffer: `rows = table.len() / vals`, row `r` value `k` at
/// `off + r * stride + 2 * k`.
#[derive(Clone, Copy)]
pub struct Placement {
    pub off: usize,
    pub vals: usize,
    pub stride: usize,
    pub src: Src,
    pub name: &'static str,
}

pub static PLACEMENTS: &[Placement] = &[''')
for off,name,stride in P:
    vals=SUB[name][0]
    src=f'Src::PerQ([&{name}[0], &{name}[1], &{name}[2], &{name}[3]])' if name in QSUB else f'Src::Fixed(&{name})'
    o.append(f'    Placement {{ off: {off:#06x}, vals: {vals}, stride: {stride}, src: {src}, name: "{name}" }},')
o.append('];')
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../video/src/av1/cdf_defaults.rs'),'w').write('\n'.join(o)+'\n')
