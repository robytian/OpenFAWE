"""The packaged OpenFAWE Studio's entry point (OF-0013): the bundled CUDA pieces registered, then multiprocessing's frozen
bootstrap, then the studio. Addendum 20 supplement 2: ``frozen_cuda.setup()`` comes before ``freeze_support()``, so a
child process (a free flight, the self-test's GPU probe) registers the DLL folders too - ``os.add_dll_directory`` is per
process and a spawned child finishes inside ``freeze_support()``."""
import multiprocessing as mp
import sys

if __name__ == "__main__":
    if getattr(sys, "frozen", False):       # OF-0013 addendum 20: the bundled CUDA pieces for numba-cuda
        from openfawe_studio import frozen_cuda
        frozen_cuda.setup()
    mp.freeze_support()                     # a free flight runs in a child process of the executable
    from openfawe_studio.__main__ import main
    sys.exit(main())
