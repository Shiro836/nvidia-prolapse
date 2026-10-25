/* SPDX-License-Identifier: MIT. Independent RM bridge packet/memory gate. */
#include "nv_rm.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { int r=(x); if(r) {fprintf(stderr,"FAIL %s = %d\n",#x,r);return 1;} }while(0)
static void method(uint32_t *w,unsigned *n,uint32_t sub,uint32_t m,unsigned count,const uint32_t *v){w[(*n)++]=(2u<<28)|(count<<16)|(sub<<13)|(m>>2);for(unsigned j=0;j<count;j++)w[(*n)++]=v[j];}
int main(int argc,char**argv){
 unsigned domain,bus,slot,func;
 if((argc!=2 && argc!=3)||sscanf(argv[1],"%x:%x:%x.%x",&domain,&bus,&slot,&func)!=4){fprintf(stderr,"usage: bridge-check PCI_ADDRESS (select NVIDIA by vendor/driver)\n");return 2;}
 struct NvRmDevice *d=NULL;struct NvRmInfo i;CHECK(nv_rm_open(domain,bus,slot,func,&d,&i));
 printf("%s chipset=%x SM%u GPC%u TPC%u MP%u warps%u classes=%x/%x/%x/%x/%x VRAM=%lu BAR=%lu\n",i.name,i.chipset,i.sm,i.gpc_count,i.tpc_count,i.mp_per_tpc,i.max_warps_per_mp,i.cls_copy,i.cls_eng2d,i.cls_eng3d,i.cls_m2mf,i.cls_compute,i.vram_size,i.bar_size);
 struct NvRmAllocation a[3];uint64_t map[3],va[3];
 for(unsigned j=0;j<3;j++){CHECK(nv_rm_alloc(d,65536,65536,j==1,0,0,&a[j]));va[j]=(1ull<<34)+j*65536;CHECK(nv_rm_bind(d,a[j].id,0,a[j].size,va[j],0,&map[j]));}
 uint64_t ctx;CHECK(nv_rm_context_create(d,1|4|16,&ctx));
 CHECK(nv_rm_exec(d,ctx,NULL,0));
 volatile uint32_t *src=(void*)a[0].cpu_address,*dst=(void*)a[1].cpu_address;
 uint32_t *cmd=(void*)a[2].cpu_address;
 for(unsigned round=0;round<3;round++){
   for(unsigned j=0;j<16384;j++){src[j]=(j*17)^round^0xabc123;dst[j]=0;}
   unsigned n=0;uint32_t object=i.cls_copy,addresses[]={va[0]>>32,va[0],va[1]>>32,va[1]},length=65536,launch=2|(1<<7)|(1<<8);
   /* Only first submission binds the engine: subsequent batches retain it. */
   if(!round)method(cmd,&n,4,0,1,&object);
   method(cmd,&n,4,0x400,4,addresses);method(cmd,&n,4,0x418,1,&length);method(cmd,&n,4,0x300,1,&launch);
   struct NvRmExec push={.address=va[2],.size_B=n*4};CHECK(nv_rm_exec(d,ctx,&push,1));
   for(unsigned j=0;j<16384;j++)if(dst[j]!=src[j]){fprintf(stderr,"GPU copy mismatch round%u word%u got%x expected%x\n",round,j,dst[j],src[j]);return 1;}
 }
 if(argc==3){
   if(strcmp(argv[2],"--queue-wrap")){fprintf(stderr,"unknown option\n");return 2;}
   /* Empty FIFO packets contain only the backend's WFI/release tail. This
    * isolates producer-ring reuse without shaders, copies or allocation churn. */
   for(unsigned j=0;j<1030;j++)CHECK(nv_rm_exec(d,ctx,NULL,0));
   puts("PASS 1030 empty completion packets across producer-ring wrap");
 }
 CHECK(nv_rm_context_destroy(d,ctx));
 for(unsigned j=0;j<3;j++){CHECK(nv_rm_unbind(d,map[j]));CHECK(nv_rm_free(d,a[j].id));}
 nv_rm_close(d);puts("PASS fixed VA, cached GART, persistent graphics/compute/copy channel: 49152 GPU-copied words");return 0;
}
