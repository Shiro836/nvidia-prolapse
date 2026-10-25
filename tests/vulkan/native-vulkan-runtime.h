/* Platform runtime accounting shared by the independent Vulkan probes.
 * No Vulkan, RM, or driver-private interfaces are used here. */
#ifndef NATIVE_VULKAN_RUNTIME_H
#define NATIVE_VULKAN_RUNTIME_H
#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline void nv_probe_android_logger_close(void) {
#ifdef __ANDROID__
  void *library = dlopen("liblog.so", RTLD_NOW | RTLD_NOLOAD);
  if (library) {
    void (*close_log)(void) = (void (*)(void))dlsym(library, "__android_log_close");
    if (!close_log) {
      fputs("FAIL: loaded Android liblog lacks __android_log_close\n", stderr);
      exit(1);
    }
    close_log();
    dlclose(library);
  }
#endif
}
static inline int nv_probe_runtime_fd_count(void) {
  DIR *directory = opendir("/proc/self/fd");
  if (!directory) {
    perror("runtime fd count");
    exit(1);
  }
  int count = 0;
  struct dirent *entry;
  while ((entry = readdir(directory)))
    if (entry->d_name[0] != '.') count++;
  closedir(directory);
  return count;
}
static inline void nv_probe_android_runtime_init(void) {
#ifdef __ANDROID__
  /* cutils/trace.h documents eager atrace_init(). trace-dev.cpp opens
   * trace_marker under pthread_once and has no shutdown API. Retain this
   * library reference for the process lifetime, just like the runtime fd. */
  void *library = dlopen("libcutils.so", RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "FAIL: Android tracing runtime libcutils: %s\n", dlerror());
    exit(1);
  }
  void (*initialize)(void) = (void (*)(void))dlsym(library, "atrace_init");
  if (!initialize) {
    fputs("FAIL: Android libcutils lacks public atrace_init\n", stderr);
    exit(1);
  }
  int before = nv_probe_runtime_fd_count();
  initialize();
  nv_probe_android_logger_close();
  int once = nv_probe_runtime_fd_count();
  initialize();
  nv_probe_android_logger_close();
  int twice = nv_probe_runtime_fd_count();
  printf("Android process tracing descriptors %d -> %d -> %d\n", before, once, twice);
  if (once != twice) {
    fputs("FAIL: repeated atrace_init retained additional descriptors\n", stderr);
    exit(1);
  }
#endif
}
static inline void nv_probe_dump_fds(void) {
  DIR *directory = opendir("/proc/self/fd");
  if (!directory) return;
  struct dirent *entry;
  while ((entry = readdir(directory))) {
    if (entry->d_name[0] == '.' || atoi(entry->d_name) == dirfd(directory)) continue;
    char path[512], target[4096];
    snprintf(path, sizeof(path), "/proc/self/fd/%s", entry->d_name);
    ssize_t size = readlink(path, target, sizeof(target) - 1);
    if (size < 0) continue;
    target[size] = 0;
    fprintf(stderr, "remaining fd %s -> %s\n", entry->d_name, target);
  }
  closedir(directory);
}
#endif
