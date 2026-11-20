"""OF-0013 addendum 4: the data root of the packaged studio, laid out as the repository (PyInstaller adds it as data).

  python apps/studio/release/stage_bundle.py STAGE [--controller third_party/makani-controller-win-nogsl]
         [--reference selftest/turnkey-20.json] [--examples FOLDER] [--version 1.0.0] [--commit SHA]

- ``src/openfawe``: only the modules the studio can reach - the import closure (``ast``, imports inside functions
  included) of the studio's package and of ``resources.TASK_FILES``'s scripts - each compiled to a ``.pyc`` beside where
  its ``.py`` is in the repository (no ``.py``; the file name inside the bytecode is the repository-relative path).
  Package ``__init__`` files of every reached module come with it. Modules nothing reaches (for example the offshore
  work for the next version) are not in the bundle.
- ``resources.TASK_FILES``: the scripts as ``.pyc`` at their paths, the data files as they are.
- ``cases/`` as in the repository but ``cases/floating`` (addendum 22); the fuselage meshes the studio draws.
- The controller folder (``lib/``: the library, its ctypes binding - generated from Makani's headers, read as text by
  ``mode_names_from_binding``, so kept as ``.py`` - and the layout file) at the same relative path, and the release file
  ``studio_release.json`` that names it (``resources.controller_build``).
- ``third_party``'s licence and notice files; the binary licence ``LICENSE-BINARIES.md`` (the public repository's text) and ``THIRD_PARTY_NOTICES.md``.
- ``selftest/turnkey-20.json`` (RS-2's reference, written by the packaging job) and ``examples/`` when given.

Writes ``STAGE/bundle-manifest.json``: every staged file with its size and SHA-256, the modules and why each is there.
"""
import argparse
import ast
import hashlib
import json
import py_compile
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STUDIO = REPO/"apps"/"studio"
sys.path.insert(0, str(STUDIO))
from openfawe_studio import resources  # noqa: E402

LIBRARY = REPO/"src"
NOTICES = ("LICENSE-Makani.txt", "MAKANI-NOTICE.md", "LICENSE-OpenFAST.txt", "OPENFAST-NOTICE.md", "LICENSE-VSM.txt",
           "VSM-NOTICE.md", "OPENFAST-WING-WAKE-NOTICE.md", "CAMBER-SOURCES.md",
           "LICENSE-TUDELFT-V3.txt")        # addendum 22 (REL-13): the V3 kite data in cases/ is MIT
EXTRA_DATA = (resources.FUSELAGE, "docs/tasks/OF-0006-body20260930/mesh/fuselage-s030-r16.npz")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_file(name):
    """The repository file of module ``name`` (``openfawe...``): the package's __init__.py or the .py; else None."""
    base = LIBRARY.joinpath(*name.split("."))
    if (base/"__init__.py").is_file():
        return base/"__init__.py"
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    return None


def module_name(file):
    rel = Path(file).relative_to(LIBRARY).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def imports_of(file, package=None):
    """Every dotted name imported anywhere in ``file`` (for ``from a import b``: 'a' and 'a.b')."""
    tree = ast.parse(Path(file).read_text(encoding="utf-8"), filename=str(file))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if package is None:
                    continue
                parts = package.split(".")
                base = ".".join(parts[:len(parts)-node.level+1])
                origin = f"{base}.{node.module}" if node.module else base
            else:
                origin = node.module
            names.add(origin)
            names.update(f"{origin}.{alias.name}" for alias in node.names if alias.name != "*")
    return names


def closure(starts):
    """openfawe modules reached from the files ``starts``: dict(name -> first file that imports it)."""
    reached, queue = {}, []

    def add(name, why):
        if not (name == "openfawe" or name.startswith("openfawe.")):
            return
        parts = name.split(".")
        for k in range(1, len(parts)+1):              # the packages above it come with it
            sub = ".".join(parts[:k])
            if sub not in reached and module_file(sub) is not None:
                reached[sub] = why
                queue.append(sub)
    for file in starts:
        for name in imports_of(file):
            add(name, str(Path(file).relative_to(REPO)))
    while queue:
        name = queue.pop()
        file = module_file(name)
        package = name if file.name == "__init__.py" else name.rpartition(".")[0]
        for imported in imports_of(file, package):
            add(imported, str(file.relative_to(REPO)))
    return reached


def third_party_imports(files):
    """Top-level dotted names (not openfawe, not the task scripts' own) imported by ``files``: PyInstaller's hidden
    imports for the modules it cannot see (openfawe and the task scripts are data in the bundle)."""
    local = {Path(f).stem for f in resources.TASK_FILES if f.endswith(".py")}
    names = set()
    for file in files:
        package = module_name(file) if Path(file).is_relative_to(LIBRARY) else None
        if package is not None and Path(file).name != "__init__.py":
            package = package.rpartition(".")[0]
        for name in imports_of(file, package):
            top = name.split(".")[0]
            if top in ("openfawe", "openfawe_studio") or top in local or top.startswith("_"):
                continue
            names.add(name)
    return sorted(names)


def compile_to(source, target, rel):
    target.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(source), cfile=str(target), dfile=rel, doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)


LEFT_OUT_CASES = ("floating",)       # addendum 22 (REL-13): the offshore models of the next version are not in 1.0


def _cases_ignore(folder, names):
    """``cases/`` as in the repository without ``__pycache__`` and the top-level folders of ``LEFT_OUT_CASES``."""
    left = {"__pycache__"}
    if Path(folder).resolve() == (REPO/"cases").resolve():
        left |= set(LEFT_OUT_CASES)
    return [n for n in names if n in left]


def stage(out, controller, reference=None, examples=None, version=None, commit=None):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists")
    out.mkdir(parents=True)
    studio_files = sorted((STUDIO/"openfawe_studio").glob("*.py"))
    task_scripts = [REPO/f for f in resources.TASK_FILES if f.endswith(".py")]
    modules = closure(studio_files + task_scripts)
    files = {}
    for name in sorted(modules):
        source = module_file(name)
        rel = source.relative_to(REPO).as_posix()
        target = out/Path(rel).with_suffix(".pyc")
        compile_to(source, target, rel)
        files[target] = dict(source=rel, source_sha256=sha(source))
    for rel in resources.TASK_FILES:
        source = REPO/rel
        if rel.endswith(".py"):
            target = out/Path(rel).with_suffix(".pyc")
            compile_to(source, target, rel)
        else:
            target = out/rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        files[target] = dict(source=rel, source_sha256=sha(source))
    shutil.copytree(REPO/"cases", out/"cases", ignore=_cases_ignore, dirs_exist_ok=True)
    for rel in EXTRA_DATA:
        target = out/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/rel, target)
    controller = Path(controller)
    lib = (REPO/controller/"lib") if not controller.is_absolute() else controller/"lib"
    rel_controller = controller.as_posix() if not controller.is_absolute() else f"third_party/{controller.name}"
    shutil.copytree(lib, out/rel_controller/"lib", ignore=shutil.ignore_patterns("__pycache__"))
    for name in NOTICES:
        shutil.copy2(REPO/"third_party"/name, out/"third_party"/name)
    for name in ("LICENSE-BINARIES.md", "THIRD_PARTY_NOTICES.md"):   # the binary licence (public v0.2.0 text), notices
        shutil.copy2(STUDIO/"release"/name, out/name)
    release = dict(app="OpenFAWE Studio", version=version, commit=commit, controller_build=rel_controller,
                   note="OF-0013: the packaged studio's settings (resources.release_settings)")
    (out/resources.RELEASE_FILE).write_text(json.dumps(release, indent=1) + "\n", encoding="utf-8")
    if reference:
        (out/"selftest").mkdir()
        shutil.copy2(reference, out/"selftest"/"turnkey-20.json")
    if examples:
        shutil.copytree(examples, out/"examples")
    manifest = dict(
        what="OF-0013 addendum 4: the packaged studio's data root", version=version, commit=commit,
        modules={name: why for name, why in sorted(modules.items())},
        hidden_imports=third_party_imports([module_file(n) for n in modules] + task_scripts),
        files={p.relative_to(out).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p),
                                                   **files.get(p, {})) for p in sorted(out.rglob("*")) if p.is_file()})
    (out/"bundle-manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stage_bundle.py")
    ap.add_argument("out")
    ap.add_argument("--controller", default="third_party/makani-controller-win-nogsl")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--examples", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--commit", default=None)
    a = ap.parse_args(argv)
    manifest = stage(a.out, a.controller, a.reference, a.examples, a.version, a.commit)
    print(json.dumps(dict(modules=len(manifest["modules"]), files=len(manifest["files"]),
                          bytes=sum(f["bytes"] for f in manifest["files"].values()),
                          hidden_imports=len(manifest["hidden_imports"])), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
