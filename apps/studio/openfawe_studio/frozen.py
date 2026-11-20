"""OF-0013 addendum 4: the packaged studio carries no ``.py`` of OpenFAWE's own, only ``.pyc`` at the same places.

Two research scripts the free flight imports load a sibling module by its file name
(``importlib.util.spec_from_file_location(name, ".../a3_pitch.py")``: ``closed_loop_checks.py`` line 69, ``a5_loop.py``
line 91). In the packaged studio only, ``install`` wraps ``spec_from_file_location`` so that a ``.py`` that is not there
is loaded from the ``.pyc`` beside it; everything else goes to the original function unchanged. The scripts themselves
are not changed, and a studio run from source never installs the wrapper (``openfawe_studio/__init__.py`` calls
``install`` only when ``sys.frozen`` is set).
"""
import importlib.util
from pathlib import Path

_ORIGINAL = importlib.util.spec_from_file_location


def sourceless(location):
    """The ``.pyc`` to load in place of ``location``: when ``location`` is a ``.py`` that does not exist and the ``.pyc``
    of the same name beside it does; else None."""
    if location is None:
        return None
    try:
        path = Path(location)
    except TypeError:
        return None
    if path.suffix != ".py" or path.exists():
        return None
    compiled = path.with_suffix(".pyc")
    return compiled if compiled.is_file() else None


def spec_from_file_location(name, location=None, *args, **kwargs):
    compiled = sourceless(location)
    if compiled is not None and not args and not kwargs.get("loader"):
        return _ORIGINAL(name, str(compiled), **kwargs)
    return _ORIGINAL(name, location, *args, **kwargs)


def install():
    """Wrap ``importlib.util.spec_from_file_location`` (once)."""
    importlib.util.spec_from_file_location = spec_from_file_location


def installed():
    return importlib.util.spec_from_file_location is spec_from_file_location
