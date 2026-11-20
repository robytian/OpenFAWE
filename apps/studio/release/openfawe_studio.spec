# -*- mode: python ; coding: utf-8 -*-
# OF-0013 addendum 4: PyInstaller build of OpenFAWE Studio (one folder). Run by release_build.py:
#   OPENFAWE_STAGE=<staged data root> python -m PyInstaller apps/studio/release/openfawe_studio.spec --noconfirm --clean
# The staged data root (stage_bundle.py: openfawe and the task scripts as .pyc at their repository paths, cases, the
# controller library, notices, the self-test reference) is copied into the bundle's data root as it is. openfawe itself
# is excluded from the archive (the studio puts data_root/src on sys.path); the third-party modules it and the task
# scripts import are listed by stage_bundle.py (bundle-manifest.json "hidden_imports"), as PyInstaller cannot see them.
import importlib.util
import json
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

RELEASE = Path(SPECPATH).resolve()
STUDIO = RELEASE.parent
STAGE = Path(os.environ["OPENFAWE_STAGE"]).resolve()
NAME = "OpenFAWE Studio"
manifest = json.loads((STAGE / "bundle-manifest.json").read_text(encoding="utf-8"))


def importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


hidden = [n for n in manifest["hidden_imports"] if importable(n)]
hiddenimports = (hidden + collect_submodules("openfawe_studio") + collect_submodules("pyqtgraph", filter=lambda n: not n.startswith(("pyqtgraph.examples", "pyqtgraph.jupyter")))
                 + collect_submodules("pyvista") + collect_submodules("pyvistaqt") + collect_submodules("vtkmodules")
                 + collect_submodules("numba") + collect_submodules("scipy.interpolate") + collect_submodules("scipy.linalg")
                 + ["PySide6.QtOpenGL", "PySide6.QtOpenGLWidgets", "PySide6.QtSvg", "PySide6.QtPrintSupport", "qtpy"])
datas = [(str(STAGE), ".")] + collect_data_files("pyqtgraph") + collect_data_files("pyvista")
binaries = []
excludes = ["openfawe", "tkinter", "IPython", "pytest", "PyQt5", "PyQt6"]
if importable("numba_cuda"):                # the GPU (CUDA 12 wheels), when the build environment has them
    # OF-0013 addendum 20: numba-cuda's CUDA target as numba.cuda (the source tree gets it through numba-cuda's .pth
    # redirector, which a frozen program never runs) - numba's own numba.cuda out, numba-cuda's tree at numba/cuda.
    _spec = importlib.util.spec_from_file_location("cuda_bundle", RELEASE / "cuda_bundle.py")
    cuda_bundle = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(cuda_bundle)
    # supplement 5: the 'cuda' namespace (cuda-bindings, cuda-core, cuda-pathfinder) is not analysed but copied as it is
    # installed - PyInstaller's submodule collection left out cuda-core's cu12/cu13 modules, and the empty namespace made
    # cuda.core's path recurse
    hiddenimports = [n for n in hiddenimports if not cuda_bundle.is_numba_cuda_name(n) and not cuda_bundle.is_cuda_name(n)]
    excludes += ["numba.cuda", "cuda"]
    hiddenimports += collect_submodules("numba_cuda")
    datas += collect_data_files("numba_cuda")
    datas += cuda_bundle.tree_datas(cuda_bundle.numba_cuda_root(importlib.util.find_spec("numba_cuda")))
    for _place in importlib.util.find_spec("cuda").submodule_search_locations or []:
        datas += cuda_bundle.tree_datas(_place, dest=("cuda",), tests_anywhere=True)
    # supplement 6: cuda-core asks importlib.metadata for cuda-bindings' version at import - the copied namespace brings
    # no metadata, so the CUDA distributions' METADATA files go in as <name>.dist-info
    datas += cuda_bundle.metadata_datas()
    _places = [Path(p) for name in ("numba_cuda", "cuda.bindings", "cuda.core") if importlib.util.find_spec(name)
               for p in (importlib.util.find_spec(name).submodule_search_locations or [])]
    binaries += cuda_bundle.libs_binaries([p if p.parent.name != "cuda" else p.parent.parent / f"cuda_{p.name}"
                                           for p in _places])
    for package in ("nvidia",):
        try:                                # namespace packages; a failure leaves the GPU out, not the build
            # addendum 22 (REL-05): only what runs - no ptxas.exe, .lib, CMake files or CCCL headers
            binaries += cuda_bundle.runtime_only(collect_dynamic_libs(package))
            datas += cuda_bundle.runtime_only(collect_data_files(package, include_py_files=False))
        except Exception as error:          # noqa: BLE001
            print(f"WARNING: GPU files of {package!r} not collected: {type(error).__name__}: {error}")

a = Analysis(
    [str(RELEASE / "studio_entry.py")],
    pathex=[str(STUDIO)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    # addendum 20 supplement 4: every module a .pyc file in _internal, imported by Python's standard machinery as in an
    # installed environment (with the archive, PyInstaller's own importer made the 'cuda' namespace path recurse)
    noarchive=True,
    # addendum 22 (REL-04): Qt for Python's own Python modules as plain files beside its binaries, so that the LGPL
    # library can be replaced as a whole (not inside the program's archive)
    module_collection_mode={"PySide6": "py", "shiboken6": "py",
                            # addendum 20 supplement 3: the 'cuda' namespace package (cuda-bindings, cuda-core,
                            # cuda-pathfinder) on disk as plain .py, so that Python's own path finder handles it alone (in
                            # the archive, with its extensions and numba/cuda on disk, its path recalculated without end)
                            "cuda": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=NAME, console=os.name != "nt", upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
