/* SPDX-License-Identifier: MIT
 * nvc2-xmlgen — media_codecs_c2.xml from the live GPU (nv_video_query_caps).
 *
 *   nvc2-xmlgen [-o FILE]       write the xml (stdout without -o)
 *   nvc2-xmlgen --check FILE    exit 0 iff FILE is exactly what this GPU produces now,
 *                               else name the difference and exit 1
 *
 * A GPU on which this decoder drives no codec exits 3 (kExitNoNvdec) instead of 0: no NVDEC class
 * this decoder knows (nvdec_class 0), or a known class on which no codec is supported (Volta, classes
 * not verified yet). The xml then lists no codecs and says which case (no_decoder_xml), and --check
 * compares against it. A query error or unusable caps stay exit 1.
 *
 * Runs on the host (GNU build of the backend) when the guest tree is staged, and inside
 * the guest by the service at start (--check), so a GPU/driver change can never leave a
 * stale codec list behind silently.
 */
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "nvc2_caps.h"

int nvc2_xmlgen_main(int argc, char **argv)
{
    const char *outpath = nullptr, *check = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc)
            outpath = argv[++i];
        else if (!strcmp(argv[i], "--check") && i + 1 < argc)
            check = argv[++i];
        else {
            fprintf(stderr, "usage: nvc2-xmlgen [-o FILE] | --check FILE\n");
            return 2;
        }
    }
    NvVideoDevice *dev = nullptr;
    NvVideoDeviceInfo info{};
    int rc = nv_video_device_open(&dev, &info);
    if (rc != 0) {
        fprintf(stderr, "nvc2-xmlgen: nv_video_device_open failed (%d): %s\n", rc, nv_video_last_error());
        return 1;
    }
    if (info.abi_version != NV_VIDEO_ABI_VERSION) {
        fprintf(stderr, "nvc2-xmlgen: backend ABI %u, built for %u\n", info.abi_version, NV_VIDEO_ABI_VERSION);
        nv_video_device_close(dev);
        return 1;
    }
    std::vector<nvc2::CodecEntry> entries;
    if (info.nvdec_class != 0) {
        std::string err;
        entries = nvc2::supported_codecs(dev, &err);
        if (!err.empty()) {
            nv_video_device_close(dev);
            fprintf(stderr, "nvc2-xmlgen: %s\n", err.c_str());
            return 1;
        }
    }
    nv_video_device_close(dev);
    std::string xml;
    int ok = 0;
    if (entries.empty()) { /* no NVDEC class, or none of VP9/AV1/HEVC/H.264 on this one */
        fprintf(stderr, "nvc2-xmlgen: %s\n", nvc2::no_decoder_reason(info.name, info.nvdec_class).c_str());
        xml = nvc2::no_decoder_xml(info.name, info.nvdec_class);
        ok = nvc2::kExitNoNvdec;
    } else {
        xml = nvc2::media_codecs_xml(entries);
    }
    if (check) {
        std::ifstream f(check);
        if (!f) {
            fprintf(stderr, "nvc2-xmlgen: cannot read %s\n", check);
            return 1;
        }
        std::stringstream ss;
        ss << f.rdbuf();
        if (ss.str() != xml) {
            fprintf(stderr, "nvc2-xmlgen: %s does not match the live caps of %s; regenerate it. Expected:\n%s", check,
                    info.name, xml.c_str());
            return 1;
        }
        return ok;
    }
    if (outpath) {
        std::ofstream f(outpath);
        f << xml;
        if (!f) {
            fprintf(stderr, "nvc2-xmlgen: cannot write %s\n", outpath);
            return 1;
        }
    } else {
        fputs(xml.c_str(), stdout);
    }
    return ok;
}

#ifndef NVC2_XMLGEN_NO_MAIN
int main(int argc, char **argv) { return nvc2_xmlgen_main(argc, argv); }
#endif
