"""OF-0013 revision 1, section 3 (criteria N1, N2): the full Makani controller library linked WITHOUT GSL, for Linux.

  build_nogsl_linux.py BASE BASE_RECEIPT SNAPSHOT GENERATED OUT RECEIPT.json      (with /usr/bin/python3)

As ``OF-0008 makani_lib/build_linux_portable.py`` (node 3's library), with one change: in place of GSL's static
archives, OpenFAWE's no-GSL subset (``src/nogsl.c``, compiled here with the same options) is linked. Every Makani object
file is BASE's own (BASE/obj, the link of BASE's receipt; their SHA-256 recorded) - including ``common/c_math/linalg``
and ``gsl_linalg_extra``, compiled by BASE against GSL's headers; the subset keeps GSL's interface (types, enumerations,
status codes), so these objects link against it unchanged.

N1: ``--no-undefined``; NEEDED only libc.so.6 and libm.so.6; every GLIBC version <= 2.35; the library's GSL-named symbols
are exactly the subset's (``nm`` of the subset object), and no other object defines any. N2: the object hashes. The
binding check (node 3 addendum 3's L3) as the portable build, so that the delivered binding loads this library.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
MAKANI_LIB = os.path.join(REPO, "docs", "tasks", "OF-0008-g5control20261006", "makani_lib")
sys.path.insert(0, MAKANI_LIB)
import build_linux_portable as P  # noqa: E402  (its helpers, unchanged)

NOGSL = os.path.join(HERE, "src", "nogsl.c")
NOGSL_INCLUDE = os.path.join(HERE, "include")


def gsl_symbols(path, defined=True):
    flag = "--defined-only" if defined else "--undefined-only"
    out = P.run(["nm", flag, path]).stdout
    return sorted({line.split()[-1] for line in out.splitlines() if line.split() and line.split()[-1].startswith(("gsl_", "cblas_"))})


def main(argv):
    ap = argparse.ArgumentParser(prog="build_nogsl_linux.py")
    for name in ("base", "base_receipt", "snapshot", "generated", "out", "receipt"):
        ap.add_argument(name)
    a = ap.parse_args(argv)
    base, out = os.path.abspath(a.base), os.path.abspath(a.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    os.makedirs(os.path.join(out, "lib"))
    os.makedirs(os.path.join(out, "obj"))
    base_receipt = json.load(open(a.base_receipt, encoding="utf-8"))
    words = base_receipt["library"]["link"].split()
    objects = [os.path.join(base, "obj", w) for w in words if w.endswith(".o")]
    compat_obj = os.path.join(out, "obj", "openfawe_glibc_compat.o")
    nogsl_obj = os.path.join(out, "obj", "openfawe_nogsl.o")
    c1 = P.run(["gcc", *P.C_OPTS, "-c", P.COMPAT, "-o", compat_obj])
    c2 = P.run(["gcc", *P.C_OPTS, "-I" + NOGSL_INCLUDE, "-c", NOGSL, "-o", nogsl_obj])
    library = os.path.join(out, "lib", P.LIBRARY)
    link = ["gcc", "-shared", "-o", library, *objects, compat_obj, nogsl_obj, "-Wl,--wrap=fmod", "-lm", "-Wl,--no-undefined"]
    linked = P.run(link) if c1.returncode == 0 and c2.returncode == 0 else None
    ok = linked is not None and linked.returncode == 0
    result = dict(
        what="OF-0013 revision 1 section 3: the full controller library without GSL (OpenFAWE's no-GSL subset), Linux",
        base=base, base_library_sha256=base_receipt["library"]["sha256"],
        objects={os.path.basename(p): P.sha(p) for p in objects},
        nogsl=dict(source_sha256=P.sha(NOGSL), headers={f: P.sha(os.path.join(NOGSL_INCLUDE, "gsl", f))
                                                        for f in sorted(os.listdir(os.path.join(NOGSL_INCLUDE, "gsl")))},
                   compile_returncode=c2.returncode, compile_stderr=c2.stderr.strip()[-800:] or None),
        compat=dict(source_sha256=P.sha(P.COMPAT), compile_returncode=c1.returncode),
        gcc=P.run(["gcc", "--version"]).stdout.splitlines()[0],
        link=" ".join(os.path.basename(w) if w.endswith(".o") else w.replace(out, "OUT") for w in link),
        link_returncode=None if linked is None else linked.returncode,
        link_stderr=None if linked is None else (linked.stderr.strip()[-2000:] or None))
    if ok:
        versions = P.glibc_versions(library)
        top = max(versions.values()) if versions else None
        need = P.needed(library)
        subset = gsl_symbols(nogsl_obj)
        defined_elsewhere = {os.path.basename(o): gsl_symbols(o) for o in objects if gsl_symbols(o)}
        in_library = gsl_symbols(library)
        result.update(
            library_sha256=P.sha(library), needed=need, glibc_max=".".join(map(str, top)) if top else None,
            gsl_named=dict(subset=subset, in_library=in_library, defined_by_other_objects=defined_elsewhere,
                           undefined_in_library=gsl_symbols(library, defined=False)),
            N1=dict(linked=True, needed_ok=set(need) <= P.ALLOWED_NEEDED,
                    glibc_ok=top is not None and top[:2] <= P.MAX_GLIBC,
                    gsl_names_only_the_subset=set(in_library) <= set(subset) and not defined_elsewhere),
            N2=dict(objects=len(objects), note="every Makani object is BASE's own file (hashes above)", passed=True))
        result["N1"]["passed"] = all(result["N1"].values())
        with tempfile.TemporaryDirectory() as tmp:
            regenerated = os.path.join(tmp, P.BINDING)
            command = ["/usr/bin/python3", os.path.join(MAKANI_LIB, "gen_bindings.py"), P.HEADER, regenerated, "--include",
                       os.path.abspath(a.snapshot), "--include", os.path.abspath(a.generated), "--include",
                       os.path.join(MAKANI_LIB, "shim"), "--root", os.path.abspath(a.snapshot), "--root",
                       os.path.abspath(a.generated), "--library", library, "--check"]
            done = P.run(command, cwd=tmp)
            base_lines = open(os.path.join(base, "lib", P.BINDING), encoding="utf-8").read().split("\n")
            new_lines = open(regenerated, encoding="utf-8").read().split("\n") if os.path.exists(regenerated) else []
            differing = [i for i, (x, y) in enumerate(zip(base_lines, new_lines)) if x != y]
            only_path = (len(base_lines) == len(new_lines) and len(differing) == 1
                         and base_lines[differing[0]].startswith("_lib = ctypes.CDLL("))
            result["binding_check"] = dict(check_returncode=done.returncode, differing_lines=[i + 1 for i in differing],
                                           passed=done.returncode == 0 and only_path)
        if result["binding_check"]["passed"]:
            lines = list(base_lines)
            lines[differing[0]] = ("_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "
                                   "'libmakani_full_controller.so'))")
            lines.insert(lines.index("import ctypes") + 1, "import os")
            with open(os.path.join(out, "lib", P.BINDING), "w", encoding="utf-8", newline="\n") as handle:
                handle.write("\n".join(lines))
            result["binding_sha256"] = P.sha(os.path.join(out, "lib", P.BINDING))
    with open(a.receipt, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")
    print(json.dumps({k: result.get(k) for k in ("link_returncode", "library_sha256", "needed", "glibc_max", "N1",
                                                 "binding_check")}, indent=1))
    if not ok:
        print(result["link_stderr"] or result["nogsl"]["compile_stderr"])
    return 0 if ok and result["N1"]["passed"] and result["binding_check"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
