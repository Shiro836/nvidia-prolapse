/* Independent C ABI check for the actual Rust external-memory module.
 */
#define _GNU_SOURCE
#include "nv_rm.h"
#include <dirent.h>
#include <dlfcn.h>
#include <drm/drm.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <nv_drm_common_ioctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define FUNCTIONS(X) X(open) X(close) X(alloc) X(free) X(export) X(import)
static int32_t (*popen_rm)(uint32_t, uint32_t, uint32_t, uint32_t,
                           struct NvRmDevice **, struct NvRmInfo *);
static void (*pclose_rm)(struct NvRmDevice *);
static int32_t (*palloc_rm)(struct NvRmDevice *, uint64_t, uint64_t, uint32_t,
                            uint32_t, uint32_t, struct NvRmAllocation *);
static int32_t (*pfree_rm)(struct NvRmDevice *, uint64_t);
static int32_t (*pexport_rm)(struct NvRmDevice *, uint64_t, int, int *);
static int32_t (*pimport_rm)(struct NvRmDevice *, int, int,
                             struct NvRmAllocation *);
static void require(int value, const char *message) {
  if (!value) {
    fprintf(stderr, "FAIL: %s: errno=%d %s\n", message, errno, strerror(errno));
    exit(1);
  }
}
static void check(int32_t result, const char *message) {
  if (result) {
    fprintf(stderr, "FAIL: %s returned %d\n", message, result);
    exit(1);
  }
}
static int fd_count(void) {
  DIR *d = opendir("/proc/self/fd");
  require(d != NULL, "count descriptors");
  int n = 0;
  struct dirent *e;
  while ((e = readdir(d)))
    if (e->d_name[0] != '.')
      n++;
  closedir(d);
  return n;
}
static int select_gpu(uint32_t pci[4]) {
  DIR *directory = opendir("/sys/class/drm");
  require(directory != NULL, "DRM sysfs");
  struct dirent *entry;
  int fd = -1;
  while ((entry = readdir(directory))) {
    if (strncmp(entry->d_name, "renderD", 7))
      continue;
    char path[512], line[128];
    snprintf(path, sizeof(path), "/sys/class/drm/%s/device/vendor",
             entry->d_name);
    FILE *f = fopen(path, "r");
    if (!f)
      continue;
    unsigned vendor = 0;
    int parsed = fscanf(f, "%x", &vendor);
    fclose(f);
    if (parsed != 1 || vendor != 0x10de)
      continue;
    snprintf(path, sizeof(path), "/sys/class/drm/%s/device", entry->d_name);
    ssize_t n = readlink(path, line, sizeof(line) - 1);
    if (n < 0)
      continue;
    line[n] = 0;
    const char *last = strrchr(line, '/');
    if (!last || sscanf(last + 1, "%x:%x:%x.%x", &pci[0], &pci[1], &pci[2],
                        &pci[3]) != 4)
      continue;
    snprintf(path, sizeof(path), "/dev/dri/%s", entry->d_name);
    fd = open(path, O_RDWR | O_CLOEXEC);
    require(fd >= 0, "open NVIDIA render node");
    char name[64] = {0};
    struct drm_version v = {.name_len = sizeof(name) - 1, .name = name};
    require(ioctl(fd, DRM_IOCTL_VERSION, &v) == 0 &&
                !strcmp(name, "nvidia-drm"),
            "NVIDIA DRM driver");
    printf("device %s PCI %04x:%02x:%02x.%x\n", path, pci[0], pci[1], pci[2],
           pci[3]);
    break;
  }
  closedir(directory);
  require(fd >= 0, "NVIDIA render device");
  return fd;
}
static uint32_t pattern(uint32_t i, uint32_t placement) {
  return (i * 0x12345u + 0x98765432u) ^ (placement * 0xfedcba98u);
}
int main(int argc, char **argv) {
  require(argc == 2 && argv[1][0] == '/',
          "usage: CHECK /absolute/libnv_bionic_native.so");
  setvbuf(stdout, NULL, _IONBF, 0);
  int before = fd_count();
  void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!library) {
    fprintf(stderr, "%s\n", dlerror());
    return 1;
  }
#define LOAD(name)                                                             \
  *(void **)(&p##name##_rm) = dlsym(library, "nv_rm_" #name);                  \
  require(p##name##_rm != NULL, "nv_rm_" #name);
  FUNCTIONS(LOAD)
#undef LOAD
  uint32_t pci[4];
  int drm = select_gpu(pci);
  for (uint32_t placement = 0; placement < 2; placement++) {
    struct NvRmDevice *producer = NULL, *consumer = NULL;
    struct NvRmInfo info = {0};
    check(popen_rm(pci[0], pci[1], pci[2], pci[3], &producer, &info),
          "open producer");
    check(popen_rm(pci[0], pci[1], pci[2], pci[3], &consumer, &info),
          "open consumer");
    struct NvRmAllocation source = {0}, imported = {0}, alias = {0};
    check(palloc_rm(producer, 65536, 4096, placement, 0, 0, &source),
          "allocate source");
    require(source.size >= 65536 && source.cpu_address, "source allocation");
    volatile uint32_t *src = (volatile uint32_t *)source.cpu_address;
    for (uint32_t i = 0; i < 16384; i++)
      src[i] = pattern(i, placement);
    __sync_synchronize();
    int fd = -1;
    check(pexport_rm(producer, source.id, drm, &fd), "export source");
    require(fd >= 0 && (fcntl(fd, F_GETFD) & FD_CLOEXEC),
            "owned CLOEXEC export fd");
    struct drm_prime_handle sentinel = {.fd = fd};
    require(ioctl(drm, DRM_IOCTL_PRIME_FD_TO_HANDLE, &sentinel) == 0,
            "caller-owned GEM sentinel");
    check(pimport_rm(consumer, drm, fd, &imported), "import across RM clients");
    require(fcntl(fd, F_GETFD) >= 0, "import borrows original fd");
    require(imported.size == source.size && imported.cpu_address,
            "imported allocation metadata");
    check(pfree_rm(producer, source.id), "destroy original allocation");
    pclose_rm(producer);
    producer = NULL;
    close(fd);
    fd = -1;
    struct drm_nvidia_gem_identify_object_params identify = {
        .handle = sentinel.handle};
    require(ioctl(drm, DRM_IOCTL_NVIDIA_GEM_IDENTIFY_OBJECT, &identify) == 0,
            "Rust import must not close caller GEM handle");
    struct drm_gem_close close_gem = {.handle = sentinel.handle};
    require(ioctl(drm, DRM_IOCTL_GEM_CLOSE, &close_gem) == 0, "close sentinel");
    volatile uint32_t *dst = (volatile uint32_t *)imported.cpu_address;
    for (uint32_t i = 0; i < 16384; i++)
      require(dst[i] == pattern(i, placement),
              "import survives producer/client/fd destruction");
    for (uint32_t i = 0; i < 16384; i++)
      dst[i] = ~pattern(i, placement);
    __sync_synchronize();
    check(pexport_rm(consumer, imported.id, drm, &fd),
          "re-export imported allocation");
    check(pimport_rm(consumer, drm, fd, &alias), "re-import alias");
    close(fd);
    check(pfree_rm(consumer, imported.id), "destroy first imported allocation");
    volatile uint32_t *again = (volatile uint32_t *)alias.cpu_address;
    for (uint32_t i = 0; i < 16384; i++)
      require(again[i] == ~pattern(i, placement),
              "re-export retains alias bytes");
    check(pfree_rm(consumer, alias.id), "destroy alias");
    pclose_rm(consumer);
    printf("placement %s: 16384 words correct, cross-client ownership, "
           "re-export, caller GEM retained\n",
           placement ? "GART cached sysmem" : "VRAM");
  }
  close(drm);
  require(dlclose(library) == 0, "dlclose");
  int after = fd_count();
  printf("file descriptors %d -> %d\n", before, after);
  require(before == after, "descriptor leak");
  puts("PASS: Rust external-memory bridge VRAM + cached sysmem; no GPU "
       "execution or presentation tested");
  return 0;
}
