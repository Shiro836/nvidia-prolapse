#!/usr/bin/env python3
"""vk.xml-driven generator + coverage gate for vulkan.nvidia.so (run with `python3 -I`).

AHB images are HAL-owned wrapper handles (src/ahb.c), so every command that can carry a VkImage
must translate it. This tool derives that set from the registry instead of trusting a hand list:

  * an "image command" is a command with a VkImage parameter, or a parameter whose struct type
    reaches a VkImage member through members, pointers, arrays, unions or pNext extension structs
    (structextends), in core Vulkan (every version) or in any vulkan device/instance extension;
  * an "object-handle command" passes a VkImage as an untyped uint64 (objectHandle + objectType);
  * a "create-info command" reaches VkImageCreateInfo, VkBufferCreateInfo,
    VkSamplerYcbcrConversionCreateInfo, VkPhysicalDeviceImageFormatInfo2 or
    VkPhysicalDeviceExternalBufferInfo, where the AHB handle type / VkExternalFormatANDROID appear.

Each of them must be intercepted (named in the intercept table of src/hal.c) or belong only to
extensions listed in tools/image-hidden-exts.txt. Core commands cannot be hidden. Any violation
fails the build (exit 1) and names the command and the path to the VkImage.

It also generates gen/nvhal_gen.c:
  nvhal_stype_size()        sType -> sizeof, guarded by the header macros of the defining
                            features/extensions (used for pNext surgery on const chains)
  nvhal_image_ext_hidden()  the hidden list, plus every intercepted image extension whose header
                            macro is absent from the header set being compiled (its translators
                            are compiled out there, so the extension must not be offered)
  nvhal_device_ext_known()  every device extension in this registry: an ICD extension unknown to
                            it cannot be checked for VkImage parameters and is hidden at runtime
  nvhal_result_name()       VkResult -> its name (+ aliases), for every VkResult value in vk.xml and in
                            every --header-dir (literal case values, so no header guards); NULL if unknown
  nvhal_registry_version()

usage: vkimage-coverage.py --xml vk.xml --hal-c src/hal.c [--hal-c src/imgxlate.c] --hidden tools/image-hidden-exts.txt
                           --out gen/nvhal_gen.c [--report]
"""
import argparse
import re
import sys
import xml.etree.ElementTree as ET

# untyped object handles that only ever come back to the app itself (debug-report messages it
# injects); a wrapper handle there is harmless and needs no translation
ECHO_ONLY = {
    "vkDebugReportMessageEXT": "the app reports a message about its own object to its own callbacks",
    "vkSubmitDebugUtilsMessageEXT": "the app injects a message naming its own objects into its own messengers",
}
CREATE_INFO_ROOTS = {"VkImageCreateInfo", "VkBufferCreateInfo", "VkSamplerYcbcrConversionCreateInfo",
                     "VkPhysicalDeviceImageFormatInfo2", "VkPhysicalDeviceExternalBufferInfo"}


def api_ok(e):
    a = e.get("api")
    return a is None or "vulkan" in a.split(",")


def text(e):
    return "".join(e.itertext())


class Registry:
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.structs, self.type_alias, self.extends, self.stype = {}, {}, {}, {}
        for t in self.root.find("types"):
            if t.get("category") not in ("struct", "union") or not api_ok(t):
                continue
            n = t.get("name")
            if t.get("alias"):
                self.type_alias[n] = t.get("alias")
                continue
            mems = []
            for m in t.findall("member"):
                if not api_ok(m):
                    continue
                ty, nm = m.find("type").text, m.find("name").text
                pre = text(m).split(nm)[0]
                mems.append({"name": nm, "type": ty, "ptr": pre.count("*"), "len": m.get("len")})
                if nm == "sType" and m.get("values"):
                    self.stype[n] = m.get("values")
            self.structs[n] = {"cat": t.get("category"), "mems": mems}
            for b in (t.get("structextends") or "").split(","):
                if b:
                    self.extends.setdefault(b, []).append(n)
        self.cmds, self.cmd_alias = {}, {}
        for c in self.root.find("commands"):
            if not api_ok(c):
                continue
            if c.get("alias"):
                self.cmd_alias[c.get("name")] = c.get("alias")
                continue
            params = []
            for p in c.findall("param"):
                if not api_ok(p):
                    continue
                ty, nm = p.find("type").text, p.find("name").text
                pre = text(p).split(nm)[0]
                params.append({"name": nm, "type": ty, "ptr": pre.count("*"), "const": "const" in pre,
                               "len": p.get("len")})
            self.cmds[c.find("proto").find("name").text] = params
        # who provides what: features (api vulkan) and vulkan extensions
        self.cmd_from, self.type_from = {}, {}
        self.dev_exts, self.all_exts = [], set()
        self.header_version = None
        for f in self.root.findall("feature"):
            if "vulkan" not in f.get("api").split(","):
                continue
            macro = "VK_VERSION_" + f.get("number").replace(".", "_")
            self._req(f, ("core", f.get("name"), macro))
        for e in self.root.find("extensions"):
            sup = e.get("supported") or ""
            if "vulkan" not in sup.split(","):
                continue
            self.all_exts.add(e.get("name"))
            if e.get("type") == "device":
                self.dev_exts.append(e.get("name"))
            self._req(e, ("ext", e.get("name"), e.get("name")))
        for t in self.root.find("types"):
            if t.get("category") == "define" and api_ok(t) and t.find("name") is not None and \
                    t.find("name").text == "VK_HEADER_VERSION":
                m = re.search(r"VK_HEADER_VERSION\s+(\d+)", text(t))
                if m:
                    self.header_version = int(m.group(1))

    def _req(self, elem, who):
        for r in elem.findall("require"):
            if not api_ok(r):
                continue
            for c in r.findall("command"):
                self.cmd_from.setdefault(c.get("name"), []).append(who)
            for t in r.findall("type"):
                self.type_from.setdefault(t.get("name"), []).append(who)

    def rt(self, t):
        while t in self.type_alias:
            t = self.type_alias[t]
        return t

    def paths(self, t, want, seen=frozenset()):
        """paths from struct t to members of type in `want` (members, pointers, unions, pNext)"""
        t = self.rt(t)
        if t not in self.structs or t in seen:
            return []
        seen = seen | {t}
        out = []
        for m in self.structs[t]["mems"]:
            mt = self.rt(m["type"])
            if m["name"] == "pNext":
                for x in self.extends.get(t, []):
                    out += [f"pNext<{x}>.{p}" for p in self.paths(x, want, seen)]
                continue
            arr = f"[{m['len']}]" if m["len"] else ""
            if mt in want:
                out.append(f"{m['name']}{arr}")
            elif mt in self.structs:
                sep = "->" if m["ptr"] else "."
                out += [f"{m['name']}{arr}{sep}{p}" for p in self.paths(mt, want, seen)]
        return out

    def reaches_struct(self, t, roots, seen=frozenset()):
        t = self.rt(t)
        if t in roots:
            return [t]
        if t not in self.structs or t in seen:
            return []
        seen = seen | {t}
        for m in self.structs[t]["mems"]:
            if m["name"] == "pNext":
                for x in self.extends.get(t, []):
                    r = self.reaches_struct(x, roots, seen)
                    if r:
                        return [f"pNext<{x}>"] + r
                continue
            r = self.reaches_struct(m["type"], roots, seen)
            if r:
                return [m["name"]] + r
        return []

    def object_handle(self, t, seen=frozenset()):
        t = self.rt(t)
        if t not in self.structs or t in seen:
            return []
        seen = seen | {t}
        names = {m["name"] for m in self.structs[t]["mems"]}
        for h in ("objectHandle", "object"):
            if h in names and "objectType" in names:
                return [f"{t}.{h}"]
        out = []
        for m in self.structs[t]["mems"]:
            if m["name"] != "pNext":
                out += self.object_handle(m["type"], seen)
        return out


def vkresult_values(R, header_dirs):
    """value -> (canonical name, sorted aliases) over vk.xml (api vulkan) and every header's
    `typedef enum VkResult`; a name bound to two different values anywhere is a hard error"""
    val, canon_xml, alias = {}, {}, {}

    def bind(name, v, where):
        if name in val and val[name] != v:
            print(f"vkimage-coverage: VkResult {name} is {val[name]} in one source and {v} in {where}", file=sys.stderr)
            sys.exit(1)
        val[name] = v

    root = R.root
    for es in root.findall("enums"):
        if es.get("name") == "VkResult":
            for e in es.findall("enum"):
                if api_ok(e) and e.get("value") is not None:
                    bind(e.get("name"), int(e.get("value"), 0), "vk.xml")
                    canon_xml[e.get("name")] = True

    def walk(elem, extnumber):
        for r in elem.findall("require"):
            if not api_ok(r):
                continue
            for e in r.findall("enum"):
                if e.get("extends") != "VkResult" or not api_ok(e):
                    continue
                if e.get("alias"):
                    alias[e.get("name")] = e.get("alias")
                    continue
                if e.get("value") is not None:
                    v = int(e.get("value"), 0)
                else:
                    num = int(e.get("extnumber") or extnumber)
                    v = 1000000000 + (num - 1) * 1000 + int(e.get("offset"))
                    if e.get("dir") == "-":
                        v = -v
                bind(e.get("name"), v, "vk.xml")
                canon_xml[e.get("name")] = True

    for f in root.findall("feature"):
        if "vulkan" in f.get("api").split(","):
            walk(f, None)
    for e in root.find("extensions"):
        if "vulkan" in (e.get("supported") or "").split(","):
            walk(e, e.get("number"))

    for d in header_dirs:
        txt = open(f"{d}/vulkan_core.h").read()
        m = re.search(r"typedef enum VkResult \{(.*?)\} VkResult;", txt, re.S)
        if not m:
            print(f"vkimage-coverage: {d}/vulkan_core.h: no `typedef enum VkResult`", file=sys.stderr)
            sys.exit(1)
        for name, rhs in re.findall(r"^\s*(VK_\w+)\s*=\s*([-\w]+)\s*,?", m.group(1), re.M):
            if name == "VK_RESULT_MAX_ENUM":
                continue
            if re.fullmatch(r"-?(0x[0-9A-Fa-f]+|\d+)", rhs):
                bind(name, int(rhs, 0), f"{d}/vulkan_core.h")
            else:
                alias[name] = rhs

    for a, t in alias.items():
        seen = {a}
        while t in alias and t not in seen:
            seen.add(t)
            t = alias[t]
        if t not in val:
            print(f"vkimage-coverage: VkResult alias {a} -> {t}, which has no value", file=sys.stderr)
            sys.exit(1)
        bind(a, val[t], "alias resolution")
    by_val = {}
    for n, v in val.items():
        by_val.setdefault(v, []).append(n)
    out = {}
    for v, names in by_val.items():
        canon = sorted(n for n in names if n in canon_xml) or sorted(n for n in names if n not in alias) or sorted(names)
        out[v] = (canon[0], sorted(n for n in names if n != canon[0]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--hal-c", required=True, action="append", help="source with an intercept table (repeatable)")
    ap.add_argument("--hidden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--header-dir", action="append", default=[],
                    help="a Vulkan include dir the generated file is compiled against (repeatable); a struct is "
                         "emitted only where its header has it, version-guarded where the header sets differ")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    R = Registry(a.xml)

    intercepted = set()
    for src in a.hal_c:
        intercepted |= set(re.findall(r'\{\s*"(vk\w+)"\s*,\s*X?F\(', open(src).read()))
    hidden = {}
    for line in open(a.hidden):
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, why = line.partition(" ")
            if name not in R.all_exts:
                print(f"vkimage-coverage: {a.hidden}: {name} is not an extension in {a.xml}", file=sys.stderr)
                sys.exit(1)
            hidden[name] = why.strip()

    errors, report = [], []
    image_exts_covered = set()

    def gated(path):
        """a path through pNext<X> exists only when X's extension is enabled: if every provider of
        X is a hidden extension, the path cannot occur"""
        for x in re.findall(r"pNext<(\w+)>", path):
            prov = R.type_from.get(x, [])
            if prov and all(k == "ext" and n in hidden for k, n, _ in prov):
                return True
        return False

    def classify(cmd, paths):
        live = [p for p in paths if not gated(p)]
        for p in paths:
            if p not in live:
                report.append(f"  gated       {cmd}: {p} (only through a hidden extension's pNext struct)")
        if not live:
            return
        why = "; ".join(live)
        providers = R.cmd_from.get(cmd, [])
        if not providers:
            return  # not provided by any vulkan feature/extension (e.g. vulkansc-only)
        if cmd in intercepted:
            for kind, name, _ in providers:
                if kind == "ext":
                    image_exts_covered.add(name)
            report.append(f"  translated  {cmd}: {why}")
            return
        core = [n for k, n, _ in providers if k == "core"]
        if core:
            errors.append(f"{cmd} ({', '.join(core)}) carries a VkImage ({why}) and is core: it must be "
                          f"intercepted (intercept tables: {', '.join(a.hal_c)})")
            return
        unhidden = [n for k, n, _ in providers if n not in hidden]
        if unhidden:
            errors.append(f"{cmd} ({', '.join(unhidden)}) carries a VkImage ({why}): intercept it "
                          f"({', '.join(a.hal_c)}) or list the extension in {a.hidden}")
        else:
            report.append(f"  hidden      {cmd}: {why} [{', '.join(n for _, n, _ in providers)}]")

    names = sorted(set(R.cmds) | set(R.cmd_alias))
    for c in names:
        real = R.cmd_alias.get(c, c)
        params = R.cmds.get(real)
        if params is None:
            continue
        img = []
        for p in params:
            if R.rt(p["type"]) == "VkImage":
                img.append(p["name"] + (" (output)" if p["ptr"] and not p["const"] else ""))
            img += [f"{p['name']}->{x}" if p["ptr"] else f"{p['name']}.{x}" for x in R.paths(p["type"], {"VkImage"})]
        if img:
            classify(c, img)
        obj = []
        pn = {p["name"] for p in params}
        if ("objectHandle" in pn or "object" in pn) and "objectType" in pn:
            obj.append("objectHandle parameter")
        for p in params:
            obj += R.object_handle(p["type"])
        if obj:
            if c in ECHO_ONLY:
                report.append(f"  echo-only   {c}: {ECHO_ONLY[c]}")
            else:
                classify(c, ["untyped object handle: " + o for o in obj])
        ci = []
        for p in params:
            r = R.reaches_struct(p["type"], CREATE_INFO_ROOTS)
            if r:
                ci.append(f"{p['name']}->{'->'.join(r)}")
        if ci:
            classify(c, ["create info that can carry the AHB handle type: " + x for x in ci])

    for name in sorted(intercepted):
        if name not in R.cmds and name not in R.cmd_alias:
            errors.append(f"an intercept table names {name}, which {a.xml} does not define")

    # ---- generate ----
    out = ["/* GENERATED by tools/vkimage-coverage.py from " + a.xml + " (VK_HEADER_VERSION " +
           str(R.header_version) + "). Do not edit. */",
           "#include \"nvhal.h\"", "#include <string.h>", ""]

    # which header sets define which structs (structs get added to existing extensions over time,
    # so the extension macro alone does not prove the struct exists in an older header)
    headers = []
    for d in a.header_dir:
        txt = ""
        for f in ("vulkan_core.h", "vulkan_android.h"):
            try:
                txt += open(f"{d}/{f}").read()
            except OSError:
                pass
        m = re.search(r"#define VK_HEADER_VERSION (\d+)", txt)
        if not m:
            print(f"vkimage-coverage: {d}: no VK_HEADER_VERSION in vulkan_core.h", file=sys.stderr)
            sys.exit(1)
        # sType enumerants usable without VK_ENABLE_BETA_EXTENSIONS (some headers define a struct
        # in vulkan_core.h while its enumerant is beta-guarded, e.g. NDK r30 VK_NV_present_metering)
        enums, beta = set(), False
        for line in txt.splitlines():
            ls = line.strip()
            if ls.startswith("#ifdef VK_ENABLE_BETA_EXTENSIONS"):
                beta = True
            elif ls.startswith("#endif"):
                beta = False
            else:
                em = re.match(r"(VK_STRUCTURE_TYPE_\w+)\s*=", ls)
                if em and not beta:
                    enums.add(em.group(1))
        names = {n for n in re.findall(r"\}\s*(Vk\w+)\s*;", txt) if R.stype.get(n) in enums}
        headers.append((int(m.group(1)), names))

    def guard(tname):
        macros = sorted({m for _, _, m in R.type_from.get(tname, [])})
        g = " || ".join(f"defined({m})" for m in macros)
        if g and headers:
            have = [v for v, names in headers if tname in names]
            if not have:
                return ""
            if len(have) != len(headers):
                g = f"({g}) && VK_HEADER_VERSION >= {min(have)}"
        return g

    out.append("size_t nvhal_stype_size(VkStructureType s) {")
    out.append("    switch ((int)s) {")
    seen_st = set()
    for n in sorted(R.structs):
        st = R.stype.get(n)
        if not st or st in seen_st:
            continue
        g = guard(n)
        if not g:
            continue
        seen_st.add(st)
        out.append(f"#if {g}")
        out.append(f"    case {st}: return sizeof({n});")
        out.append("#endif")
    out += ["    default: return 0;", "    }", "}", ""]

    out.append("bool nvhal_image_ext_hidden(const char *name) {")
    out.append("    static const char *const always[] = {")
    out += [f"        \"{n}\", /* {why[:90]} */" for n, why in sorted(hidden.items())]
    out.append("    };")
    out.append("    for (size_t i = 0; i < sizeof always / sizeof *always; i++)")
    out.append("        if (!strcmp(name, always[i])) return true;")
    for e in sorted(image_exts_covered):
        out.append(f"#ifndef {e}")
        out.append(f"    if (!strcmp(name, \"{e}\")) return true; /* translators compiled out: header lacks it */")
        out.append("#endif")
    out += ["    return false;", "}", ""]

    out.append("bool nvhal_device_ext_known(const char *name) {")
    out.append("    static const char *const known[] = {")
    out += [f"        \"{n}\"," for n in sorted(R.dev_exts)]
    out.append("    };")
    out.append("    for (size_t i = 0; i < sizeof known / sizeof *known; i++)")
    out.append("        if (!strcmp(name, known[i])) return true;")
    out += ["    return false;", "}", ""]
    out.append(f"const char *nvhal_registry_version(void) {{ return \"vk.xml VK_HEADER_VERSION {R.header_version}\"; }}")
    out.append("")
    results = vkresult_values(R, a.header_dir)
    out.append("const char *nvhal_result_name(VkResult r) {")
    out.append("    switch ((int)r) {")
    for v in sorted(results):
        canon, al = results[v]
        nm = canon + (" aka " + "/".join(al) if al else "")
        out.append(f"    case {v}: return \"{nm}\";")
    out += ["    default: return NULL;", "    }", "}"]
    with open(a.out, "w") as f:
        f.write("\n".join(out) + "\n")

    if a.report:
        print("\n".join(report))
    if errors:
        for e in errors:
            print(f"vkimage-coverage: FAIL: {e}", file=sys.stderr)
        sys.exit(1)
    nt = len({r.split()[1].rstrip(":") for r in report if r.startswith("  translated")})
    nh = len({r.split()[1].rstrip(":") for r in report if r.startswith("  hidden")})
    print(f"vkimage-coverage: OK: {nt} commands carrying a VkImage / untyped image handle / AHB create info are "
          f"intercepted, {nh} belong only to hidden "
          f"extensions ({', '.join(sorted(hidden))}), registry VK_HEADER_VERSION {R.header_version}; "
          f"intercepted image extensions: {', '.join(sorted(image_exts_covered))}")


if __name__ == "__main__":
    main()
