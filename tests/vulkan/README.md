# tests/vulkan — driver-level Vulkan tests

Small public-API Vulkan programs that load the built driver directly and check results on a real NVIDIA GPU:

| Program | What it checks |
|---|---|
| `native-vulkan-probe.c` | runtime-compiled compute shaders, buffer copies, fences, rejected unknown extensions/features (`--negative`) |
| `native-vulkan-offscreen.c` | a triangle rendered into a tiled image and read back, every pixel compared |
| `native-vulkan-interop.c` | dma-buf memory and sync_fd export/import between two devices |
| `../../hal/test/*` | the HAL's ANB, AHB, ASTC and ETC2 paths over NVK |

    tests/vulkan/build.py                                  # host + x86_64 + i686 programs, SPIR-V
    tests/vulkan/run.py --target host --rust-profile release           # print the commands
    tests/vulkan/run.py --target host --rust-profile release --run     # run them (needs the GPU)

`run.py` checks the build records `mesa/build.py` wrote before it runs anything. The x86_64 programs run under the
image's unmodified bionic `linker64` (`build/bionic-runtime64`, `mesa/prepare-runtime64.py`). The i686 programs run
under the image's unmodified bionic linker (`build/nvk-android32/runtime`, `mesa/prepare-runtime32.py`) in a new user
and PID namespace, so a host PID above 65535 cannot overflow bionic's 32-bit thread id.
