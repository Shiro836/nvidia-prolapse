/* TSan does not intercept glibc's C11 mtx_/cnd_/thrd_/call_once (GCC 16 libtsan, clang 23 compiler-rt: no such
 * interceptors). glibc implements them on the same storage as pthread_* (mtx_t=pthread_mutex_t, cnd_t=pthread_cond_t,
 * once_flag={int}=pthread_once_t, thrd_t=pthread_t), so these wrappers forward to the pthread_* functions TSan does
 * intercept. Linked into the TSan test executable with -rdynamic: the executable is first in the global lookup scope,
 * so Mesa's mtx_lock@GLIBC_2.34 references bind here. Semantics as glibc nptl/mtx_*.c, cnd_*.c, thrd_*.c. */
#include <threads.h>
#include <pthread.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
static int map(int e){return e==0?thrd_success:e==EBUSY?thrd_busy:e==ETIMEDOUT?thrd_timedout:e==ENOMEM?thrd_nomem:thrd_error;}
int mtx_init(mtx_t *m,int type){pthread_mutexattr_t a;pthread_mutexattr_init(&a);
 if(type&mtx_recursive)pthread_mutexattr_settype(&a,PTHREAD_MUTEX_RECURSIVE);
 int r=pthread_mutex_init((pthread_mutex_t*)m,&a);pthread_mutexattr_destroy(&a);return map(r);}
void mtx_destroy(mtx_t *m){pthread_mutex_destroy((pthread_mutex_t*)m);}
int mtx_lock(mtx_t *m){return map(pthread_mutex_lock((pthread_mutex_t*)m));}
int mtx_trylock(mtx_t *m){return map(pthread_mutex_trylock((pthread_mutex_t*)m));}
int mtx_unlock(mtx_t *m){return map(pthread_mutex_unlock((pthread_mutex_t*)m));}
int cnd_init(cnd_t *c){return map(pthread_cond_init((pthread_cond_t*)c,NULL));}
void cnd_destroy(cnd_t *c){pthread_cond_destroy((pthread_cond_t*)c);}
int cnd_signal(cnd_t *c){return map(pthread_cond_signal((pthread_cond_t*)c));}
int cnd_broadcast(cnd_t *c){return map(pthread_cond_broadcast((pthread_cond_t*)c));}
int cnd_wait(cnd_t *c,mtx_t *m){return map(pthread_cond_wait((pthread_cond_t*)c,(pthread_mutex_t*)m));}
int cnd_timedwait(cnd_t *restrict c,mtx_t *restrict m,const struct timespec *restrict t){
 return map(pthread_cond_timedwait((pthread_cond_t*)c,(pthread_mutex_t*)m,t));}
void call_once(once_flag *f,void(*fn)(void)){pthread_once((pthread_once_t*)f,fn);}
struct tramp{thrd_start_t fn;void*arg;};
static void *tr(void*p){struct tramp t=*(struct tramp*)p;free(p);return (void*)(intptr_t)t.fn(t.arg);}
int thrd_create(thrd_t *thr,thrd_start_t fn,void*arg){struct tramp*t=malloc(sizeof*t);if(!t)return thrd_nomem;t->fn=fn;t->arg=arg;
 int r=pthread_create((pthread_t*)thr,NULL,tr,t);if(r){free(t);return r==EAGAIN?thrd_nomem:thrd_error;}return thrd_success;}
int thrd_join(thrd_t thr,int*res){void*v;int r=pthread_join((pthread_t)thr,&v);if(r)return thrd_error;if(res)*res=(int)(intptr_t)v;return thrd_success;}
thrd_t thrd_current(void){return (thrd_t)pthread_self();}
