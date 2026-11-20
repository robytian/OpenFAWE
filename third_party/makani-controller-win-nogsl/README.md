# Makani controller library for Windows x86-64, without GSL (OpenFAWE v1.0.0)

OF-0013 revision 1, section 3 (decision 0057, owner 2026-11-12 19:54: "选 a（把 GSL 换掉重编）").

- `lib/libmakani_full_controller.so`: a Windows DLL (PE32+, x86-64) under the name the binding opens from its own folder.
  It holds Makani's flight-controller C sources (Apache License 2.0, `../MAKANI-NOTICE.md`), compiled unchanged with
  MinGW-w64, and OpenFAWE's own implementation of the few linear-algebra routines Makani's `common/c_math` calls through
  the GNU Scientific Library's interface (`docs/tasks/OF-0013-release-v1-20261112/nogsl/`). **No GSL code is in it.**
- `lib/makani_full_controller_api.py`, `lib/makani_controller_api_layout.json`: the ctypes binding and the structure
  layout, unchanged from the GSL build (the API is the same).
- Build: `nogsl/build_nogsl_windows.py` from a rebuild of `OF-0008 makani_lib/build_windows_full.py` (receipts
  `docs/tasks/OF-0013-release-v1-20261112/results/nogsl-windows-build.json`, `windows-full-base-rebuild.json`). The
  DLL links twice to the same bytes; it imports only KERNEL32.dll and msvcrt.dll.
- Checks: OF-0013 criteria N1-N5 (`docs/tasks/OF-0013-release-v1-20261112/contract.zh.md`). Its results agree with the
  GSL build at the rounding level, not bit for bit. The research computations of paper 2 used the GSL build.
- The cross compiler of this build is MinGW-w64 GCC 13 with the POSIX thread model (the GSL build of 2026-10 used the
  win32 thread model, no longer installed in the cloud workspace); recorded in the receipts.
