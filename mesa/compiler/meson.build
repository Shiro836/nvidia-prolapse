native_nak = shared_library('nv_native_nak', 'nv_nak.c',
  include_directories : [inc_include, inc_src],
  dependencies : [idep_nak, idep_vtn, idep_nir, idep_mesautil, dep_thread, dep_m],
  gnu_symbol_visibility : 'hidden',
  install : false)

native_nak_check = executable('nv-nak-check', 'check.c', link_with : native_nak, install : false)
