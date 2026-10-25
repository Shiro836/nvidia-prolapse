#!/usr/bin/env python3
"""Offline regression: production bridge + pinned Mesa with fake DRM/RM calls."""
from pathlib import Path
import json, os, shlex, subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BUILD=ROOT/'build/nvk-host'
SOURCE=ROOT/'build/mesa-rm-source'
OUT=ROOT/'build/gpu-coverage/pdev-preflight'
OUT.mkdir(parents=True,exist_ok=True)
commands=json.loads((BUILD/'compile_commands.json').read_text())
def run(c):
 subprocess.run(list(map(str,c)),cwd=BUILD,check=True,env=env,timeout=30)
env=dict(os.environ)
env['LD_LIBRARY_PATH']=str(ROOT/'build/nvk-host-deps/root/usr/lib')
objects=[]
for suffix,source in [('nvkmd_rm.c',HERE/'pdev-preflight.c'),('vk_device.c',HERE/'sync-timeline-mode.c'),('vk_fence.c',SOURCE/'src/vulkan/runtime/vk_fence.c'),('vk_semaphore.c',SOURCE/'src/vulkan/runtime/vk_semaphore.c'),('u_sync_provider.c',SOURCE/'src/util/u_sync_provider.c'),('nvk_mem_stream.c',SOURCE/'src/nouveau/vulkan/nvk_mem_stream.c'),('vk_queue.c',SOURCE/'src/vulkan/runtime/vk_queue.c')]:
 row=next(r for r in commands if r['file'].endswith('/'+suffix))
 args=shlex.split(row['command']);new=[];skip=False
 for i,a in enumerate(args):
  if skip:skip=False;continue
  if a in ['-o','-MQ','-MF']:skip=True;continue
  if a in ['-MD','-c'] or a==row['file']:continue
  new.append(a)
 obj=OUT/(suffix+'.o');objects.append(obj)
 run(new+['-UNDEBUG','-Wno-missing-prototypes','-Wno-unused-parameter','-I'+str(SOURCE/'src/nouveau/vulkan/nvkmd/rm'),'-c',source,'-o',obj])
lines=subprocess.check_output(['ninja','-C',str(BUILD),'-t','commands','src/nouveau/vulkan/libvulkan_nouveau.so'],text=True).splitlines()
args=shlex.split(lines[-1]);new=[];skip=False
for a in args:
 if skip:skip=False;continue
 if a=='-o':skip=True;continue
 if a in ['-shared','-Wl,--whole-archive','-Wl,--no-whole-archive','-lnv_bionic_native','-Wl,-Bsymbolic'] or a.startswith('-Wl,-soname,'):continue
 new.append(a)
wraps=['nanosleep','fprintf','open64','close','fstat64','calloc','free','drmGetVersion','drmFreeVersion','drmGetCap','drmSyncobjCreate','drmSyncobjWait','drmSyncobjDestroy','drmSyncobjTimelineWait','os_time_get_nano','__vk_errorf','__vk_errorv','__vk_log_impl']
exe=OUT/'pdev-preflight'
run([new[0],*objects,*new[1:],*[f'-Wl,--wrap={x}' for x in wraps],'-o',exe])
run([exe])
