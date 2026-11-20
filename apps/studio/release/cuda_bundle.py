"""OF-0013 addendum 20: numba-cuda's CUDA target placed in the bundle as ``numba.cuda`` (read by openfawe_studio.spec).

In the source tree numba-cuda's ``.pth`` file installs a redirector that makes ``import numba.cuda`` load numba-cuda's
``numba_cuda/numba/cuda`` package. A frozen program does not run ``.pth`` files, so it would get numba's own older
``numba.cuda`` (which fails on the bundled CUDA 12.9 NVVM, addendum 18 result 3). The spec therefore leaves numba's
``numba.cuda`` out and copies numba-cuda's tree to ``numba/cuda`` in the bundle (source and extension files, no tests),
where the frozen program's path finder imports it as ``numba.cuda``.
"""
from pathlib import Path


def is_numba_cuda_name(name):
    return name == "numba.cuda" or name.startswith("numba.cuda.")


def is_cuda_name(name):
    """The 'cuda' namespace package (cuda-bindings, cuda-core, cuda-pathfinder) - addendum 20 supplement 5."""
    return name == "cuda" or name.startswith("cuda.")


def numba_cuda_root(spec):
    """numba-cuda's ``numba/cuda`` folder from ``importlib.util.find_spec('numba_cuda')`` (a package or a namespace)."""
    places = list(spec.submodule_search_locations or []) or ([str(Path(spec.origin).parent)] if spec.origin else [])
    for place in places:
        root = Path(place) / "numba" / "cuda"
        if (root / "__init__.py").is_file():
            return root
    raise FileNotFoundError(f"numba-cuda's numba/cuda not found under {places}")


def tree_datas(root, dest=("numba", "cuda"), tests_anywhere=False):
    """PyInstaller ``datas`` entries (source file, destination folder) for every file of ``root`` but the ``tests``
    folder (only the top one, or with ``tests_anywhere`` any folder named so) and ``__pycache__``, destinations under
    ``dest`` (default ``numba/cuda``; addendum 20 supplement 5: also the whole 'cuda' namespace folder as ``cuda``)."""
    root = Path(root)
    out = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        tests = ("tests" in rel.parts[:-1]) if tests_anywhere else (len(rel.parts) > 1 and rel.parts[0] == "tests")
        if not path.is_file() or "__pycache__" in rel.parts or tests:
            continue
        out.append((str(path), str(Path(*dest) / rel.parent)))
    return out


def libs_binaries(package_dirs):
    """OF-0013 addendum 20 supplement 1: the DLLs that delvewheel put next to a repaired wheel's package
    (``<site-packages>/<package>.libs/*.dll``, e.g. numba-cuda's and cuda-bindings' own MSVC runtime under a mangled
    name, which their extension modules load) as PyInstaller ``binaries`` into ``<package>.libs`` in the bundle. A frozen
    program never runs the wheel's DLL-folder patch for numba-cuda's tree placed as numba.cuda, so the entry adds these
    folders (``openfawe_studio.frozen_cuda``)."""
    out = []
    for package_dir in package_dirs:
        package_dir = Path(package_dir)
        libs = package_dir.parent / (package_dir.name + ".libs")
        if libs.is_dir():
            out += [(str(dll), libs.name) for dll in sorted(libs.glob("*.dll"))]
    return out


METADATA_PREFIXES = ("cuda-", "cuda_", "numba-cuda", "numba_cuda", "nvidia-", "nvidia_")


def metadata_datas(distributions=None, prefixes=METADATA_PREFIXES):
    """OF-0013 addendum 20 supplement 6: the ``METADATA`` file of every installed distribution whose name starts with one
    of ``prefixes`` (cuda-bindings, cuda-core, cuda-pathfinder, numba-cuda, the nvidia-* runtime wheels), as PyInstaller
    ``datas`` into ``<name>.dist-info`` at the bundle's root - cuda-core asks ``importlib.metadata`` for cuda-bindings'
    version while it is imported, and the 'cuda' namespace is copied, not analysed, so no hook brings the metadata.
    Only ``METADATA`` (what the version lookup reads): ``RECORD`` and ``direct_url.json`` can hold paths of the build PC."""
    import importlib.metadata as md
    out = set()
    for dist in (md.distributions() if distributions is None else distributions):
        name = (dist.metadata["Name"] or "").lower()
        folder = getattr(dist, "_path", None)
        if folder is None or not name.startswith(prefixes):
            continue
        folder = Path(folder)
        if folder.name.endswith(".dist-info") and (folder / "METADATA").is_file():
            out.add((str(folder / "METADATA"), folder.name))
    return sorted(out)


NOT_RUNTIME_SUFFIXES = (".exe", ".lib", ".cmake")
NOT_RUNTIME_FOLDERS = ("cuda_cccl",)


def runtime_only(entries):
    """OF-0013 addendum 22 (review REL-05): of the collected NVIDIA files, leave out what a program never runs - the
    developer tools (``ptxas.exe``), import and static libraries (``.lib``), CMake files and the CUDA C++ core libraries'
    headers (``nvidia/cuda_cccl``). Entries are PyInstaller (source, destination folder) pairs."""
    out = []
    for source, destination in entries:
        parts = Path(source).parts
        if str(source).lower().endswith(NOT_RUNTIME_SUFFIXES) or any(f in parts for f in NOT_RUNTIME_FOLDERS):
            continue
        out.append((source, destination))
    return out
