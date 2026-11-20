"""OpenFAWE Studio: a desktop application that runs OpenFAWE's aerodynamic models live and shows them in 3D.

The studio drives the OpenFAWE library through its public functions only and changes no computation (decision 0052;
``tests/test_engine.py`` compares a studio run with the same run made directly through the library, bit for bit).
"""
__version__ = "0.1.0.dev0"
APP_NAME = "OpenFAWE Studio"
COPYRIGHT = "© 2026 Yinong Tian"

import sys as _sys

if getattr(_sys, "frozen", False):          # the packaged studio only (OF-0013 addendum 4): .pyc in place of a missing .py
    from . import frozen as _frozen
    from . import resources as _resources
    _frozen.install()
    _resources.ensure_library()             # openfawe is not in the bundle's archive: data_root()/src goes on sys.path
    __version__ = str(_resources.release_settings().get("version") or __version__)   # OF-0013 add 11: the release's
