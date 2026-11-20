"""OF-0013 addendum 20: in the packaged program (PyInstaller, frozen) numba-cuda finds the CUDA pieces the program
carries - NVVM and libdevice under ``nvidia/cuda_nvcc`` (``CUDA_HOME``'s layout), and the runtime, NVRTC, nvJitLink and
NVVM DLL folders. Called by the entry point only when frozen; the source tree never calls it. Only this process's
environment is changed; a ``CUDA_PATH`` the user set is left as it is.
"""
import os
import sys
from pathlib import Path

BIN_FOLDERS = (("cuda_runtime", "bin"), ("cuda_nvrtc", "bin"), ("nvjitlink", "bin"), ("cuda_nvcc", "nvvm", "bin"))
WHEEL_LIBS = ("numba_cuda.libs", "cuda_bindings.libs", "cuda_core.libs")   # addendum 20 supplements 1 and 5


def setup(base=None, environ=None, add_dll_directory=None):
    """Set CUDA_HOME to the bundled ``nvidia/cuda_nvcc`` and put the bundled DLL folders first on PATH (and in the DLL
    search, on Windows). Returns what was done ({} when the program carries no CUDA)."""
    base = Path(base if base is not None else getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    environ = os.environ if environ is None else environ
    nvidia = base / "nvidia"
    nvcc = nvidia / "cuda_nvcc"
    if not (nvcc / "nvvm").is_dir():
        return {}
    environ["CUDA_HOME"] = str(nvcc)
    folders = [str(nvidia.joinpath(*parts)) for parts in BIN_FOLDERS if nvidia.joinpath(*parts).is_dir()]
    folders += [str(base / libs) for libs in WHEEL_LIBS if (base / libs).is_dir()]     # supplement 1 (delvewheel DLLs)
    environ["PATH"] = os.pathsep.join(folders + ([environ["PATH"]] if environ.get("PATH") else []))
    adder = add_dll_directory if add_dll_directory is not None else getattr(os, "add_dll_directory", None)
    handles = []
    if adder is not None:
        for folder in folders:
            try:
                handles.append(adder(folder))
            except OSError:
                pass
    setup.handles = handles                 # kept alive: a handle that is collected removes its folder again
    return dict(CUDA_HOME=environ["CUDA_HOME"], dll_folders=folders)
