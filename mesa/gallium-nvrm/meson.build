# SPDX-License-Identifier: MIT
# nvrm: gallium driver over the native RM backend (nv_rm.h), NAK and nil.
files_libnvrm = files(
  'nvrm_compute.c',
  'nvrm_context.c',
  'nvrm_copy.c',
  'nvrm_draw.c',
  'nvrm_handle.c',
  'nvrm_mme.c',
  'nvrm_image.c',
  'nvrm_priv.h',
  'nvrm_public.h',
  'nvrm_query.c',
  'nvrm_resource.c',
  'nvrm_screen.c',
  'nvrm_shader_cache.c',
  'nvrm_texture.c',
)

libnvrm = static_library(
  'nvrm',
  files_libnvrm,
  include_directories : [
    inc_src, inc_include, inc_gallium, inc_gallium_aux,
    include_directories('../../../nouveau/vulkan/nvkmd/rm'),
    include_directories('../../../nouveau/winsys'), # drf.h (DRF_LO/HI)
    include_directories('../../../nouveau/vulkan'), # nvk_pascal_a.h (shared with NVK)
  ],
  gnu_symbol_visibility : 'hidden',
  dependencies : [
    dep_libdrm, idep_mesautil, idep_nir_headers, idep_nvidia_headers,
    idep_nak, idep_nil, idep_nouveau_mme,
  ],
)

driver_nvrm = declare_dependency(
  compile_args : '-DGALLIUM_NVRM',
  link_with : [libnvrm],
  dependencies : [idep_nak, idep_nil],
  link_args : ['-lnv_bionic_native'],
)
