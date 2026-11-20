"""OF-0013 revision 1, section 3: the full Makani controller library for Windows x86-64 WITHOUT GSL.

  build_nogsl_windows.py BASE OUT RECEIPT.json        (with /usr/bin/python3)

BASE is a finished ``OF-0008 makani_lib/build_windows_full.py`` build (BASE/obj, BASE/lib and its receipt next to it as
BASE-receipt.json, or the path given by the environment variable NOGSL_BASE_RECEIPT). Its link line is used as it is,
with ``-L GSL/lib -lgsl -lgslcblas`` replaced by OpenFAWE's no-GSL subset (``src/nogsl.c``) compiled here by the same
cross compiler with the same options; every Makani object is BASE's own file (SHA-256 recorded). The DLL is linked
twice and must come out byte-identical (as BASE's build). The binding and the layout file are BASE's (the subset changes
no exported name of the API); the receipt records the imported DLLs and names (msvcrt's ``_assert``/``_wassert`` must
not be imported, as BASE), the GSL-named symbols of every object (only the subset may define any) and of the DLL.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CROSS, OBJDUMP, NM = "x86_64-w64-mingw32-gcc", "x86_64-w64-mingw32-objdump", "x86_64-w64-mingw32-nm"
C_OPTS = ("-std=c99", "-Wmissing-prototypes", "-Wnested-externs", "-Wstrict-prototypes", "-O2")
LIBRARY = "libmakani_full_controller.so"
SIDE_FILES = ("makani_full_controller_api.py", "makani_controller_api_layout.json")


def run(command, **kw):
    return subprocess.run(command, capture_output=True, text=True, **kw)


def sha(path):
    import hashlib
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def gsl_defined(path):
    out = run([NM, "--defined-only", path]).stdout
    return sorted({l.split()[-1] for l in out.splitlines() if l.split() and l.split()[-1].startswith(("gsl_", "cblas_"))})


def imports(dll):
    """The imported DLLs and the names imported from msvcrt.dll (BASE's own parser, build_windows_full.msvcrt_imports)."""
    import re
    sys.path.insert(0, os.path.join(HERE, "..", "..", "OF-0008-g5control20261006", "makani_lib"))
    import build_windows_full as BWF                          # noqa: E402
    dump = run([OBJDUMP, "-p", dll]).stdout
    return sorted(set(re.findall(r"DLL Name: (\S+)", dump))), BWF.msvcrt_imports(dump)


def link(base, out, words, nogsl_obj, target):
    command, skip = [], False
    for w in words:
        if skip:
            skip = False
            continue
        if w == "-L":
            skip = True                      # "-L GSL/lib"
            continue
        if w in ("-lgsl", "-lgslcblas"):
            continue
        if w == "OUT/lib/" + LIBRARY:
            command.append(target)
        elif w.endswith(".o"):
            command.append(os.path.join(base, "obj", w))
        else:
            command.append(w)
    insert = command.index("-lm")
    command[insert:insert] = [nogsl_obj]
    return command, run(command)


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    base, out, receipt_path = (os.path.abspath(p) for p in argv)
    base_receipt_path = os.environ.get("NOGSL_BASE_RECEIPT", base + "-receipt.json")
    base_receipt = json.load(open(base_receipt_path, encoding="utf-8"))
    if os.path.exists(out):
        raise SystemExit(f"{out} exists")
    os.makedirs(os.path.join(out, "lib"))
    os.makedirs(os.path.join(out, "obj"))
    words = base_receipt["link"].split()
    nogsl_obj = os.path.join(out, "obj", "openfawe_nogsl.o")
    compiled = run([CROSS, *C_OPTS, "-I" + os.path.join(HERE, "include"), "-c", os.path.join(HERE, "src", "nogsl.c"),
                    "-o", nogsl_obj])
    result = dict(what="OF-0013 revision 1 section 3: the full controller library for Windows without GSL",
                  base=base, base_receipt=base_receipt_path, base_dll_sha256=base_receipt.get("dll_sha256"),
                  compiler=run([CROSS, "--version"]).stdout.splitlines()[0], options=list(C_OPTS),
                  nogsl=dict(source_sha256=sha(os.path.join(HERE, "src", "nogsl.c")), compile_returncode=compiled.returncode,
                             compile_stderr=compiled.stderr.strip()[-800:] or None))
    ok = compiled.returncode == 0
    if ok:
        target = os.path.join(out, "lib", LIBRARY)
        command, first = link(base, out, words, nogsl_obj, target)
        second_target = os.path.join(out, "second", LIBRARY)       # the same file name: a DLL records its own name
        os.makedirs(os.path.dirname(second_target))
        _, second = link(base, out, words, nogsl_obj, second_target)
        ok = first.returncode == 0 and second.returncode == 0
        result.update(link=" ".join(os.path.basename(w) if w.endswith(".o") else w.replace(out, "OUT") for w in command),
                      link_returncode=first.returncode, link_stderr=first.stderr.strip()[-2000:] or None)
        if ok:
            objects = [w for w in command if w.endswith(".o")]
            dlls, msvcrt = imports(target)
            result.update(
                dll_sha256=sha(target), dll_bytes=os.path.getsize(target), second_build_identical=sha(second_target) == sha(target),
                object_sha256={os.path.basename(o): sha(o) for o in objects},
                imported_libraries=dlls, msvcrt_imported=msvcrt,
                gsl_named=dict(subset=gsl_defined(nogsl_obj),
                               other_objects={os.path.basename(o): gsl_defined(o) for o in objects
                                              if o != nogsl_obj and gsl_defined(o)}))
            for name in SIDE_FILES:
                shutil.copy2(os.path.join(base, "lib", name), os.path.join(out, "lib", name))
            result["side_files"] = {name: sha(os.path.join(out, "lib", name)) for name in SIDE_FILES}
            result["N1"] = dict(linked=True, second_build_identical=result["second_build_identical"],
                                only_system_dlls=set(d.lower() for d in dlls) <= {"kernel32.dll", "msvcrt.dll"},
                                no_msvcrt_assert=not ({"_assert", "_wassert"} & set(msvcrt)),
                                gsl_names_only_the_subset=not result["gsl_named"]["other_objects"])
            result["N1"]["passed"] = all(result["N1"].values())
            shutil.rmtree(os.path.dirname(second_target))
    with open(receipt_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=1)
        handle.write("\n")
    print(json.dumps({k: result.get(k) for k in ("dll_sha256", "dll_bytes", "imported_libraries", "N1")}, indent=1))
    return 0 if ok and result.get("N1", {}).get("passed") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
