/* SPDX-License-Identifier: MIT */
#ifndef NVRM_PUBLIC_H
#define NVRM_PUBLIC_H

struct pipe_screen;
struct pipe_screen_config;

/* Gallium screen over the native RM backend (nv_rm.h) for an fd whose kernel
 * driver is NVIDIA's nvidia-drm. Returns NULL (after a named error on stderr)
 * for any other fd or an unsupported 3D class. */
struct pipe_screen *nvrm_screen_create(int fd, const struct pipe_screen_config *config);

#endif
