"""Where the studio finds the OpenFAWE package and its data: the repository in development, the bundle when frozen."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def data_root():
    """The folder that holds ``cases/`` (and ``docs/tasks/.../mesh`` for the fuselage)."""
    if os.environ.get("OPENFAWE_STUDIO_DATA"):
        return Path(os.environ["OPENFAWE_STUDIO_DATA"]).resolve()
    if getattr(sys, "frozen", False):                       # PyInstaller: data next to the executable's internals
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    return HERE.parents[2]                                  # apps/studio/openfawe_studio -> the repository


def ensure_library():
    """Make ``import openfawe`` work from a source checkout (the repository's src/)."""
    try:
        import openfawe  # noqa: F401
        return
    except ImportError:
        pass
    source = data_root()/"src"
    if source.is_dir():
        sys.path.insert(0, str(source))


RELEASE_FILE = "studio_release.json"          # OF-0013 addendum 4: only in a packaged studio, at the data root
DEFAULT_CONTROLLER_BUILD = "third_party/makani-controller-win-full"


def release_settings():
    """The packaged studio's settings (``studio_release.json`` at the data root), or {} in a source checkout."""
    import json
    file = data_root()/RELEASE_FILE
    if not file.is_file():
        return {}
    return json.loads(file.read_text(encoding="utf-8"))


def controller_build():
    """The bundled Makani controller folder (with lib/): the release file's ``controller_build`` when there is one (the
    packaged studio: the library without GSL), else DEFAULT_CONTROLLER_BUILD, as before OF-0013."""
    return data_root()/release_settings().get("controller_build", DEFAULT_CONTROLLER_BUILD)


def path(relative):
    full = data_root()/relative
    if not full.exists():
        raise FileNotFoundError(f"studio data file missing: {relative}")
    return full


# OF-0010 revision 19 addendum 26 (ARC-01): the whole-flight vortex hook, the power-system settings and the energy account
# live in task folders. The folders the studio puts on sys.path, and every repository file outside src/ that importing
# them and building the hook reads (turnkey_vortex.case_files() loads six check scripts of four task folders at run time).
# A packaged studio ships TASK_FILES as data at these relative paths, so the scripts' TASK.parents[2] is data_root().
# tests/test_studio_task_files.py keeps the list complete.
TASK_FOLDERS = ("docs/tasks/OF-0008-g5control20261006", "docs/tasks/OF-0010-platform20261022")
TASK_FILES = (
    "docs/tasks/OF-0007-g4motion20261002/a3_pitch.py",
    "docs/tasks/OF-0007-g4motion20261002/a5_loop.py",
    "docs/tasks/OF-0007-wingfull20261001/alpha_sweep.py",
    "docs/tasks/OF-0008-g5control20261006/closed_loop_checks.py",
    "docs/tasks/OF-0008-g5control20261006/drag_checks.py",
    "docs/tasks/OF-0008-g5control20261006/pylon_loop_checks.py",
    "docs/tasks/OF-0008-g5control20261006/turnkey_checks.py",
    "docs/tasks/OF-0008-g5control20261006/turnkey_vortex.py",
    "docs/tasks/OF-0008-g5control20261006/turnkey_whole_vortex.py",
    "docs/tasks/OF-0010-platform20261022/platform_flight.py",
    "cases/qualification/m600_makani_reconstructed_pylons_profile_drag.json",
    "cases/qualification/m600_makani_xfoil_polars_pylons.json",
    "docs/tasks/OF-0006-body20260930/mesh/fuselage-s030-r16.npz",
)


def task_modules():
    """Put TASK_FOLDERS on sys.path (each at the front, in this order, unless it is there already) so that
    ``turnkey_whole_vortex``, ``turnkey_checks`` and ``platform_flight`` import by name."""
    for folder in TASK_FOLDERS:
        folder = str(path(folder))
        if folder not in sys.path:
            sys.path.insert(0, folder)


ROTOR_SET = "cases/rotors/m600_rotors_makani.json"
ROTOR_TABLE = "cases/rotors/makani/rotor_rev4.json"
FUSELAGE = "docs/tasks/OF-0006-body20260930/mesh/fuselage-s015-r24.npz"
RIGID_BODY = "cases/aircraft/m600_makani_rigid_body.json"
PLAYBOOK = "cases/flight/m600_crosswind_playbook_land.json"
