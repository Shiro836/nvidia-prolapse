/* SPDX-License-Identifier: MIT
 *
 * nvrm program cache: NAK binaries keyed by the NIR st/mesa hands to
 * nvrm_compile_shader, in a process-wide in-memory table (in-process
 * dedupe and re-links) backed by the screen's Mesa disk_cache (relaunches).
 *
 * The disk_cache object is the one st/mesa uses for its GLSL metadata and
 * NIR (get_disk_shader_cache -> ctx->Cache, st_context.c), so one store holds
 * both levels, as iris (iris_disk_cache.c), radeonsi (si_state_shaders.cpp
 * si_shader_cache_insert/load) and nvc0 (nvc0_program.c) do with theirs.
 *
 * Key: BLAKE3 over the cache's driver keys (build id of this library, which
 * covers nvrm + NAK + st + NIR since they are linked into it; driver_flags =
 * 3D class, SM, warps per SM, NAK_DEBUG flags) and nir_serialize(strip=true)
 * of the shader with info.source_blake3 cleared. nvrm's own lowering and
 * NAK are deterministic functions of that NIR and those inputs, so the key
 * is taken before them and a hit skips both. strip=true is the iris/ir3/nvc0
 * key form (iris_program.c nir_sha1, ir3_disk_cache.c, nvc0 via
 * nv50_ir_prog_info_serialize); source_blake3 is the GLSL source hash the
 * linker records (gl_nir_linker.c), read by no nvrm or NAK pass, and would
 * otherwise split identical NIR from different sources into separate keys.
 * Value: NVK's serialized shader layout (nvk_shader.c nvk_shader_serialize:
 * raw nak_shader_info, code size, code) plus nvrm's per-shader metadata;
 * nothing in it is an address (NAK code is PC-relative; addresses come from
 * the upload). A value that does not parse exactly is dropped and the shader
 * is compiled, as radeonsi does with a bad cache entry.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "nvrm_priv.h"

#include "nak.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_serialize.h"
#include "util/blob.h"
#include "util/detect_os.h"
#include "util/disk_cache.h"
#include "util/disk_cache_os.h"
#include "util/hash_table.h"
#include "util/hex.h"
#include "util/log.h"
#include "util/mesa-blake3.h"
#include "util/u_atomic.h"
#include "util/u_memory.h"

/* Bumped whenever the value layout below changes (the build id changes with
 * any rebuild anyway; this guards layout changes within one source tree). */
#define NVRM_SHADER_CACHE_VALUE_VERSION 1

/* driver_flags of the screen's disk_cache: every key (st/mesa's and ours)
 * covers them. The 3D class decides the caps (and so st/mesa's NIR); SM and
 * warps per SM reach the NAK binary (NAK encoders per SM; warps as the
 * immediate of load_warps_per_sm_nv, nak_nir.c); NAK_DEBUG changes code
 * generation. One 3D class spans several SMs, so the class alone would hand
 * a binary to another chip of the same class. NAK flags in bits 48..63 as
 * NVK's compiler flags (nvk_physical_device_compiler_flags,
 * nvk_shader.c:70-83). */
static uint64_t
nvrm_shader_cache_driver_flags(const struct nvrm_screen *screen)
{
   const uint64_t nak_flags = nak_debug_flags(screen->nak);
   assert(screen->info.cls_eng3d <= UINT16_MAX);
   assert(screen->info.sm <= UINT8_MAX);
   assert(screen->info.max_warps_per_mp <= UINT8_MAX);
   assert(nak_flags <= UINT16_MAX);
   return (uint64_t)screen->info.cls_eng3d |
          (uint64_t)screen->info.sm << 16 |
          (uint64_t)screen->info.max_warps_per_mp << 24 |
          nak_flags << 48;
}

#if DETECT_OS_ANDROID
/* The application's private code cache directory, or NULL.
 *
 * Android's framework sets TMPDIR = Context.getCacheDir() in every
 * non-isolated application process (ActivityThread.handleBindApplication,
 * before any application code, so before eglInitialize creates this
 * screen); getCodeCacheDir() is its sibling "code_cache", meant for
 * compiled code and cleared on application and platform upgrades. Isolated
 * processes, system_server and native processes (SurfaceFlinger, tools) have
 * no such directory: the caller keeps compiled shaders in memory only.
 * (AOSP android-13 frameworks/base: ActivityThread.java:6424-6434 sets TMPDIR
 * to Context.getCacheDir() with Os.setenv; Context.java:1563-1578 documents
 * code_cache; isolated processes skip it, ActivityThread.java:6716-6725.)
 * The directory must exist, belong to this uid and be writable. */
static char *
nvrm_app_code_cache_dir(void)
{
   const char *tmpdir = getenv("TMPDIR");
   if (!tmpdir || tmpdir[0] != '/')
      return NULL;
   size_t len = strlen(tmpdir);
   while (len > 1 && tmpdir[len - 1] == '/')
      len--;
   while (len > 0 && tmpdir[len - 1] != '/')
      len--;
   while (len > 1 && tmpdir[len - 1] == '/')
      len--;
   if (len <= 1)
      return NULL; /* TMPDIR directly under / has no application parent */
   char *dir = NULL;
   if (asprintf(&dir, "%.*s/code_cache", (int)len, tmpdir) < 0)
      return NULL;
   struct stat st;
   if (stat(dir, &st) != 0 || !S_ISDIR(st.st_mode) || st.st_uid != getuid() ||
       access(dir, W_OK | X_OK) != 0) {
      free(dir);
      return NULL;
   }
   return dir;
}
#endif

static uint32_t
nvrm_shader_key_hash(const void *key)
{
   uint32_t h;
   memcpy(&h, key, sizeof(h));
   return h;
}

static bool
nvrm_shader_key_equal(const void *a, const void *b)
{
   return memcmp(a, b, CACHE_KEY_SIZE) == 0;
}

/* Called once at screen creation. */
void
nvrm_shader_cache_init(struct nvrm_screen *screen)
{
   simple_mtx_init(&screen->shader_bins_mutex, mtx_plain);
   screen->shader_bins = _mesa_hash_table_create(NULL, nvrm_shader_key_hash,
                                                 nvrm_shader_key_equal);

   /* Cache identity as nouveau_disk_cache_create (Mesa 26.2
    * nouveau_screen.c:177-197): this library's build id. */
   blake3_hasher ctx;
   unsigned char blake3[BLAKE3_KEY_LEN];
   char cache_id[BLAKE3_HEX_LEN];
   _mesa_blake3_init(&ctx);
   if (!disk_cache_get_function_identifier(nvrm_shader_cache_init, &ctx)) {
      nvrm_error("no build id in the driver library: no shader disk cache, "
                 "no program binaries\n");
      return;
   }
   _mesa_blake3_final(&ctx, blake3);
   mesa_bytes_to_hex(cache_id, blake3, BLAKE3_KEY_LEN);
   const uint64_t flags = nvrm_shader_cache_driver_flags(screen);

#if DETECT_OS_ANDROID
   /* Mesa's default location does not exist for Android applications and
    * its Android default is off (disk_cache_enabled): the cache goes to the
    * application's code cache instead. Logged once per process (one screen
    * per process in practice). */
   static bool logged;
   char *code_cache = nvrm_app_code_cache_dir();
   if (code_cache) {
      char *path = NULL;
      if (asprintf(&path, "%s/nvrm_shader_cache", code_cache) >= 0) {
         screen->disk_cache = disk_cache_create_with_path("nvrm", cache_id, flags, path);
         if (!logged) {
            if (screen->disk_cache && !screen->disk_cache->path_init_failed)
               mesa_logi("nvrm: shader cache at %s\n", path);
            else
               mesa_logw("nvrm: shader cache at %s could not be opened (or "
                         "MESA_SHADER_CACHE_DISABLE is set): compiled shaders "
                         "are kept in memory for this process only\n", path);
            logged = true;
         }
         free(path);
      }
      free(code_cache);
   } else if (!logged) {
      const char *tmpdir = getenv("TMPDIR");
      mesa_logi("nvrm: no application code_cache directory (TMPDIR %s): "
                "compiled shaders are kept in memory for this process only\n",
                tmpdir ? tmpdir : "unset");
      logged = true;
   }
   if (screen->disk_cache)
      return;
#endif
   /* Mesa's own location and options (MESA_SHADER_CACHE_DIR, XDG cache;
    * disabled by default on Android, where it still serves the
    * EGL_ANDROID_blob_cache callbacks and program binaries). */
   screen->disk_cache = disk_cache_create("nvrm", cache_id, flags);
}

static void
nvrm_shader_bin_free(struct nvrm_shader_bin *bin)
{
   free(bin->code);
   FREE(bin);
}

void
nvrm_shader_bin_unref(struct nvrm_shader_bin *bin)
{
   if (p_atomic_dec_zero(&bin->refcnt))
      nvrm_shader_bin_free(bin);
}

static void
nvrm_shader_cache_drop_entry(struct hash_entry *entry)
{
   nvrm_shader_bin_unref(entry->data);
}

/* Called at screen destruction, after every shader was deleted. */
void
nvrm_shader_cache_fini(struct nvrm_screen *screen)
{
   if (!screen->shader_bins)
      return; /* screen creation failed before nvrm_shader_cache_init */
   _mesa_hash_table_destroy(screen->shader_bins, nvrm_shader_cache_drop_entry);
   screen->shader_bins = NULL;
   simple_mtx_destroy(&screen->shader_bins_mutex);
}

/* The NAK fragment key nvrm compiles every fragment shader with (no sample
 * shading: set_min_samples is not implemented). Part of the cache key the
 * day it is not constant (NVK hashes it, nvk_shader.c nvk_hash_state). */
const struct nak_fs_key nvrm_nak_fs_key = { 0 };

void
nvrm_shader_cache_key(struct nvrm_screen *screen, nir_shader *nir, cache_key key)
{
   struct blob b;
   blob_init(&b);
   blob_write_bytes(&b, "NAK1", 4);
   blob_write_uint32(&b, nir->info.stage);
   blob_write_uint32(&b, 0 /* robust2_modes, nvrm_nak_compile */);
   blob_write_bytes(&b, &nvrm_nak_fs_key, sizeof(nvrm_nak_fs_key));
   blake3_hash source;
   memcpy(source, nir->info.source_blake3, sizeof(source));
   memset(nir->info.source_blake3, 0, sizeof(source));
   nir_serialize(&b, nir, true);
   memcpy(nir->info.source_blake3, source, sizeof(source));
   if (b.out_of_memory) {
      nvrm_error("out of memory serializing a shader for its cache key\n");
      abort();
   }
   if (screen->disk_cache)
      disk_cache_compute_key(screen->disk_cache, b.data, b.size, key);
   else
      _mesa_blake3_compute(b.data, b.size, key); /* in-process table only */
   blob_finish(&b);
}

static void
nvrm_shader_bin_write(struct blob *b, const struct nvrm_shader_bin *bin)
{
   blob_write_uint32(b, NVRM_SHADER_CACHE_VALUE_VERSION);
   /* nak.h builds nak_shader_info with -Wpadded as an error: its raw bytes
    * are deterministic (NVK serializes it raw too). */
   blob_write_bytes(b, &bin->info, sizeof(bin->info));
   blob_write_uint32(b, bin->code_size);
   blob_write_bytes(b, bin->code, bin->code_size);
   blob_write_uint32(b, bin->tex_units);
   blob_write_uint32(b, bin->image_units);
   blob_write_uint32(b, bin->cbuf_mask);
   blob_write_uint8(b, bin->uses_desc);
}

static struct nvrm_shader_bin *
nvrm_shader_bin_read(const struct nvrm_screen *screen, const void *data,
                     size_t size, mesa_shader_stage stage)
{
   struct blob_reader r;
   blob_reader_init(&r, data, size);
   if (blob_read_uint32(&r) != NVRM_SHADER_CACHE_VALUE_VERSION)
      return NULL;
   struct nvrm_shader_bin *bin = CALLOC_STRUCT(nvrm_shader_bin);
   if (!bin)
      return NULL;
   blob_copy_bytes(&r, &bin->info, sizeof(bin->info));
   bin->code_size = blob_read_uint32(&r);
   if (!r.overrun && bin->code_size <= (size_t)(r.end - r.current)) {
      bin->code = malloc(MAX2(bin->code_size, 1));
      if (bin->code)
         blob_copy_bytes(&r, bin->code, bin->code_size);
   }
   bin->tex_units = blob_read_uint32(&r);
   bin->image_units = blob_read_uint32(&r);
   bin->cbuf_mask = blob_read_uint32(&r);
   bin->uses_desc = blob_read_uint8(&r);
   if (!bin->code || r.overrun || r.current != r.end || bin->info.stage != stage ||
       bin->info.sm != screen->info.sm || bin->code_size == 0) {
      nvrm_shader_bin_free(bin);
      return NULL;
   }
   bin->refcnt = 1;
   return bin;
}

/* Table insert under shader_bins_mutex: returns the entry for `key` with a
 * reference for the caller. If another thread inserted the same key first,
 * `bin` is dropped in favour of the existing one. */
static struct nvrm_shader_bin *
nvrm_shader_cache_insert_locked(struct nvrm_screen *screen, struct nvrm_shader_bin *bin)
{
   struct hash_entry *entry = _mesa_hash_table_search(screen->shader_bins, bin->key);
   if (entry) {
      struct nvrm_shader_bin *existing = entry->data;
      p_atomic_inc(&existing->refcnt);
      nvrm_shader_bin_unref(bin);
      return existing;
   }
   p_atomic_inc(&bin->refcnt); /* the table's reference */
   _mesa_hash_table_insert(screen->shader_bins, bin->key, bin);
   return bin;
}

struct nvrm_shader_bin *
nvrm_shader_cache_find(struct nvrm_screen *screen, const cache_key key,
                       mesa_shader_stage stage)
{
   simple_mtx_lock(&screen->shader_bins_mutex);
   struct hash_entry *entry = _mesa_hash_table_search(screen->shader_bins, key);
   if (entry) {
      struct nvrm_shader_bin *bin = entry->data;
      p_atomic_inc(&bin->refcnt);
      simple_mtx_unlock(&screen->shader_bins_mutex);
      return bin;
   }
   simple_mtx_unlock(&screen->shader_bins_mutex);

   if (!screen->disk_cache)
      return NULL;
   size_t size = 0;
   void *data = disk_cache_get(screen->disk_cache, key, &size);
   if (!data)
      return NULL;
   struct nvrm_shader_bin *bin = nvrm_shader_bin_read(screen, data, size, stage);
   free(data);
   if (!bin) {
      /* Not ours or damaged past the cache's own CRC: drop it so the
       * compiled shader replaces it. */
      disk_cache_remove(screen->disk_cache, key);
      return NULL;
   }
   memcpy(bin->key, key, CACHE_KEY_SIZE);
   simple_mtx_lock(&screen->shader_bins_mutex);
   bin = nvrm_shader_cache_insert_locked(screen, bin);
   simple_mtx_unlock(&screen->shader_bins_mutex);
   return bin;
}

struct nvrm_shader_bin *
nvrm_shader_cache_add(struct nvrm_screen *screen, const cache_key key,
                      struct nvrm_shader_bin *bin)
{
   memcpy(bin->key, key, CACHE_KEY_SIZE);
   simple_mtx_lock(&screen->shader_bins_mutex);
   struct nvrm_shader_bin *ret = nvrm_shader_cache_insert_locked(screen, bin);
   simple_mtx_unlock(&screen->shader_bins_mutex);
   if (ret == bin && screen->disk_cache) {
      struct blob b;
      blob_init(&b);
      nvrm_shader_bin_write(&b, bin);
      if (b.out_of_memory) {
         nvrm_error("out of memory serializing a shader for the disk cache\n");
         abort();
      }
      /* disk_cache_put_nocopy takes the buffer (compressed and written on
       * the cache's own queue). */
      size_t size;
      void *data;
      blob_finish_get_buffer(&b, &data, &size);
      disk_cache_put_nocopy(screen->disk_cache, key, data, size, NULL);
   }
   return ret;
}
