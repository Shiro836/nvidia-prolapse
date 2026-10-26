AV1 regression streams (CPU-only tests in `../tests.rs`; hardware check with `nv-video-test`).

Source: Big Buck Bunny (c) 2008 Blender Foundation, CC-BY 3.0 (https://peach.blender.org/), first 40
frames of the 1080p 30 fps cut used by the video gallery, scaled to 640x360 yuv420p (`s360.y4m`).

    ffmpeg -i s360.y4m -c:v libaom-av1 -cpu-used 6 -row-mt 1 -tiles 2x2 -crf 32 -b:v 0 \
        -lag-in-frames 35 -auto-alt-ref 1 -g 120 -f ivf aom_hidden_640.ivf
    ffmpeg -i s360.y4m -c:v libsvtav1 -preset 8 -crf 35 -g 64 -svtav1-params film-grain=12 \
        -f ivf svt_grain_640.ivf

`aom_hidden_640`: temporal units with up to several hidden (alt-ref) frames, show_existing_frame.
`svt_grain_640`: film grain on shown and hidden frames, show_existing_frame of grain frames.
Both decode bit-exact against libdav1d (grain applied) on an RTX 5090 (2026-10-14).
