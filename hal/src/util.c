/* Logging and the mutex-protected handle maps. */
#include "nvhal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __ANDROID__
#include <android/log.h>
#endif

int nvhal_trace;
pthread_mutex_t nvhal_lock = PTHREAD_MUTEX_INITIALIZER;

void nvhal_log_init(void) {
    const char *t = getenv("NVHAL_TRACE");
    nvhal_trace = t && *t && strcmp(t, "0") != 0;
}

void nvhal_log(enum nvhal_level lvl, const char *fmt, ...) {
    char buf[1536];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    static const char *const lvl_name[] = {"E", "W", "I"};
#ifdef __ANDROID__
    static const int prio[] = {ANDROID_LOG_ERROR, ANDROID_LOG_WARN, ANDROID_LOG_INFO};
    __android_log_write(prio[lvl], NVHAL_LOG_TAG, buf);
    if (nvhal_trace)
        fprintf(stderr, NVHAL_LOG_TAG " %s: %s\n", lvl_name[lvl], buf);
#else
    fprintf(stderr, NVHAL_LOG_TAG " %s: %s\n", lvl_name[lvl], buf);
#endif
}

struct nvhal_vkr_str nvhal_vkr_fmt(VkResult r) {
    struct nvhal_vkr_str o;
    const char *n = nvhal_result_name(r);
    if (n)
        snprintf(o.s, sizeof o.s, "%s (%d)", n, (int)r);
    else
        snprintf(o.s, sizeof o.s, "VkResult %d (unknown name)", (int)r);
    return o;
}

void *nvhal_map_get(const struct nvhal_map *m, const void *key) {
    for (size_t i = 0; i < m->n; i++)
        if (m->e[i].key == key) return m->e[i].val;
    return NULL;
}

bool nvhal_map_put(struct nvhal_map *m, const void *key, void *val) {
    for (size_t i = 0; i < m->n; i++)
        if (m->e[i].key == key) { m->e[i].val = val; return true; }
    if (m->n == m->cap) {
        size_t cap = m->cap ? m->cap * 2 : 8;
        struct nvhal_map_entry *e = realloc(m->e, cap * sizeof *e);
        if (!e) return false;
        m->e = e;
        m->cap = cap;
    }
    m->e[m->n].key = key;
    m->e[m->n].val = val;
    m->n++;
    return true;
}

void *nvhal_map_del(struct nvhal_map *m, const void *key) {
    for (size_t i = 0; i < m->n; i++)
        if (m->e[i].key == key) {
            void *v = m->e[i].val;
            m->e[i] = m->e[--m->n];
            return v;
        }
    return NULL;
}

void nvhal_map_del_val(struct nvhal_map *m, const void *val) {
    for (size_t i = 0; i < m->n;)
        if (m->e[i].val == val) m->e[i] = m->e[--m->n];
        else i++;
}

void nvhal_map_free(struct nvhal_map *m) {
    free(m->e);
    m->e = NULL;
    m->n = m->cap = 0;
}

/* ---- read-mostly hash tables ---- */

#define HASH_TOMB UINT64_MAX /* never a live handle: non-dispatchable handles are pointers or ids */

static size_t hash_slot(uint64_t k, size_t cap) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    return (size_t)k & (cap - 1);
}

static bool hash_find_locked(const struct nvhal_hash *h, uint64_t key, size_t *where) {
    if (!h->cap) return false;
    for (size_t i = hash_slot(key, h->cap), probes = 0; probes < h->cap; i = (i + 1) & (h->cap - 1), probes++) {
        if (h->k[i] == 0) return false;
        if (h->k[i] == key) {
            *where = i;
            return true;
        }
    }
    return false;
}

void *nvhal_hash_get(struct nvhal_hash *h, uint64_t key) {
    if (!key || key == HASH_TOMB) return NULL;
    pthread_rwlock_rdlock(&h->lock);
    size_t at;
    void *v = hash_find_locked(h, key, &at) ? h->v[at] : NULL;
    pthread_rwlock_unlock(&h->lock);
    return v;
}

static bool hash_grow_locked(struct nvhal_hash *h) {
    size_t cap = h->cap ? h->cap : 64;
    while ((h->n + 1) * 2 > cap) cap *= 2; /* load <= 1/2 after the rehash, tombstones dropped */
    uint64_t *k = calloc(cap, sizeof *k);
    void **v = calloc(cap, sizeof *v);
    if (!k || !v) { free(k); free(v); return false; }
    for (size_t i = 0; i < h->cap; i++)
        if (h->k[i] && h->k[i] != HASH_TOMB) {
            size_t j = hash_slot(h->k[i], cap);
            while (k[j]) j = (j + 1) & (cap - 1);
            k[j] = h->k[i];
            v[j] = h->v[i];
        }
    free(h->k);
    free(h->v);
    h->k = k;
    h->v = v;
    h->cap = cap;
    h->used = h->n;
    return true;
}

bool nvhal_hash_put(struct nvhal_hash *h, uint64_t key, void *val) {
    if (!key || key == HASH_TOMB) return false;
    bool ok = true;
    size_t at;
    pthread_rwlock_wrlock(&h->lock);
    if (hash_find_locked(h, key, &at)) {
        h->v[at] = val;
    } else if ((h->used + 1) * 4 > h->cap * 3 && !hash_grow_locked(h)) {
        ok = false;
    } else {
        size_t i = hash_slot(key, h->cap);
        while (h->k[i] && h->k[i] != HASH_TOMB) i = (i + 1) & (h->cap - 1);
        if (h->k[i] == 0) h->used++;
        h->k[i] = key;
        h->v[i] = val;
        h->n++;
    }
    pthread_rwlock_unlock(&h->lock);
    return ok;
}

void *nvhal_hash_del(struct nvhal_hash *h, uint64_t key) {
    if (!key || key == HASH_TOMB) return NULL;
    void *v = NULL;
    size_t at;
    pthread_rwlock_wrlock(&h->lock);
    if (hash_find_locked(h, key, &at)) {
        v = h->v[at];
        h->k[at] = HASH_TOMB;
        h->v[at] = NULL;
        h->n--;
    }
    pthread_rwlock_unlock(&h->lock);
    return v;
}

size_t nvhal_hash_sweep(struct nvhal_hash *h, bool (*pred)(void *val, void *arg), void *arg,
                        void (*on_del)(void *val)) {
    size_t n = 0;
    pthread_rwlock_wrlock(&h->lock);
    for (size_t i = 0; i < h->cap; i++)
        if (h->k[i] && h->k[i] != HASH_TOMB && pred(h->v[i], arg)) {
            if (on_del) on_del(h->v[i]);
            h->k[i] = HASH_TOMB;
            h->v[i] = NULL;
            h->n--;
            n++;
        }
    pthread_rwlock_unlock(&h->lock);
    return n;
}
