# SPDX-License-Identifier: Apache-2.0
"""Resume a long TurnKey run from a saved state (OF-0010 revision 12 with its addendum 1; opt-in).

``ground.turnkey.run_turnkey(checkpoint=...)`` calls ``checkpoint.save(k, make_blob)`` at the start of every period
``k`` that is a multiple of ``checkpoint.every`` (not the first period of the run); ``make_blob()`` returns the blob
(``turnkey_blob``; called by the checkpoint, so that an error in it is the checkpoint's to handle), which holds what the
loop carries from one period to the next:

* the loop's own quantities (wing state, tether nodes and speeds, servo model, the flap and speed commands of the period
  before, rotor speeds, thrust coefficient, the ground-station command of the period before, the mode counters, the kept
  records), the flying system (with ground station and tether), the power system and its command pipeline, and the
  segment hook's attributes (its vortex systems, wake and rotor rings), pickled;
* Makani's controller: ``FullController``'s ctypes structures as bytes (control state, synchronisation data, output,
  ground-side estimator state, fault masks) and its three Python-side values;
* the library's global structures reachable through the binding (``LIBRARY_GLOBALS``: the parameters and the
  telemetry structures, among them ``control_telemetry.c``'s static ``ControlTelemetry``, which keeps the values of the
  last mode that wrote them and which the records read), as bytes.

The objects the caller passes in and shares with the run (parameters, aerodynamic database, rotor discs, the library's
handle and parameter structures, the power system's configuration) are not pickled: a ``persistent_id`` names them and
``turnkey_restore`` links the names to the same objects of the new process.

``run_turnkey(resume=blob)`` initialises as without it, then puts these back and continues the loop at the saved period.
Everything else - where the blobs go, when a run may resume, the caller's own state - is the caller's (OF-0010's
``platform_flight.py``).
"""

import ctypes
import io
import pickle

FORMAT = "openfawe turnkey resume 1"
LIBRARY_GLOBALS = ("GetSystemParams", "GetControlParams", "GetControlTelemetry", "GetControlSlowTelemetry",
                   "GetSmallControlTelemetry", "GetControlDebugMessage")
CONTROLLER_STRUCTS = ("state", "sync_in", "output", "est", "no_fault", "ground_estimator_faults")
CONTROLLER_VALUES = ("est_time", "tether_release_latched", "gs_unpause_transform_count")
LOOP_NAMES = ("state", "nodes", "speeds", "system", "servo", "flap_command", "speed_command", "omegas", "thrust_coeff",
              "pending", "mode_entry", "last_mode", "gain_done_period", "descend_entry", "records", "pss", "pipeline")


def _has_pointers(kind, seen=None):
    """Whether a ctypes type holds a pointer anywhere (its bytes would not mean the same in another process)."""
    seen = set() if seen is None else seen
    if isinstance(kind, type) and issubclass(kind, (ctypes._Pointer, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_wchar_p,
                                                    ctypes._CFuncPtr)):
        return True
    if isinstance(kind, type) and issubclass(kind, ctypes.Array):
        return _has_pointers(kind._type_, seen)
    if isinstance(kind, type) and issubclass(kind, (ctypes.Structure, ctypes.Union)):
        if kind in seen:
            return False
        seen.add(kind)
        return any(_has_pointers(field[1], seen) for field in kind._fields_)
    return False


def _raw(obj):
    if _has_pointers(type(obj)):
        raise ValueError(f"{type(obj).__name__} holds pointers; its bytes cannot be carried to another process")
    return ctypes.string_at(ctypes.addressof(obj), ctypes.sizeof(obj))


def _put(obj, data):
    if len(data) != ctypes.sizeof(obj):
        raise ValueError(f"{type(obj).__name__}: {len(data)} bytes saved, {ctypes.sizeof(obj)} here")
    ctypes.memmove(ctypes.addressof(obj), data, len(data))


def _global(api, name):
    getter = getattr(api, name, None)
    if getter is None:
        return None
    if getattr(getter, "argtypes", None):
        raise ValueError(f"{name} takes arguments; not a global structure")
    pointer = getter()
    return None if not pointer else pointer.contents


def library_bytes(api):
    """The library's global structures (``LIBRARY_GLOBALS`` that the binding has), as bytes."""
    out = {}
    for name in LIBRARY_GLOBALS:
        obj = _global(api, name)
        if obj is not None:
            out[name] = _raw(obj)
    return out


def restore_library(api, data):
    for name, raw in data.items():
        obj = _global(api, name)
        if obj is None:
            raise ValueError(f"the library has no {name}")
        _put(obj, raw)


def controller_bytes(fc):
    return dict(structs={name: _raw(getattr(fc, name)) for name in CONTROLLER_STRUCTS},
                values={name: getattr(fc, name) for name in CONTROLLER_VALUES}, label=int(fc.label),
                follow_init_state=bool(fc.follow_init_state))


def restore_controller(fc, data):
    if (int(fc.label), bool(fc.follow_init_state)) != (data["label"], data["follow_init_state"]):
        raise ValueError("the controller's label or initialisation following differs from the saved one")
    for name in CONTROLLER_STRUCTS:
        _put(getattr(fc, name), data["structs"][name])
    for name in CONTROLLER_VALUES:
        setattr(fc, name, data["values"][name])


def shared_names(**objects):
    """id -> name for the objects shared with the caller (a tuple or list also names its elements)."""
    names = {}
    for name, obj in objects.items():
        if obj is None:
            continue
        names[id(obj)] = name
        if isinstance(obj, (tuple, list)):
            for i, item in enumerate(obj):
                names.setdefault(id(item), f"{name}[{i}]")
    return names


def shared_objects(**objects):
    """name -> object, the inverse of ``shared_names`` for the objects of this process."""
    out = {}
    for name, obj in objects.items():
        if obj is None:
            continue
        out[name] = obj
        if isinstance(obj, (tuple, list)):
            for i, item in enumerate(obj):
                out.setdefault(f"{name}[{i}]", item)
    return out


class _Pickler(pickle.Pickler):
    def __init__(self, stream, names):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        self.names = names

    def persistent_id(self, obj):
        return self.names.get(id(obj))


class _Unpickler(pickle.Unpickler):
    def __init__(self, stream, objects):
        super().__init__(stream)
        self.objects = objects

    def persistent_load(self, name):
        if name not in self.objects:
            raise pickle.UnpicklingError(f"the shared object {name!r} is not given in this process")
        return self.objects[name]


def turnkey_blob(k, loop, segment, fc, api, shared):
    """The state at the start of period ``k``: ``loop`` the loop's quantities (``LOOP_NAMES``), ``segment`` the hook or
    None, ``shared`` the caller's shared objects (``shared_objects`` keywords)."""
    missing = set(LOOP_NAMES) - set(loop)
    if missing:
        raise ValueError(f"loop quantities missing: {sorted(missing)}")
    stream = io.BytesIO()
    _Pickler(stream, shared_names(**shared)).dump(dict(
        format=FORMAT, k=int(k), loop={name: loop[name] for name in LOOP_NAMES},
        segment=None if segment is None else dict(vars(segment)), segment_class=None if segment is None else
        f"{type(segment).__module__}.{type(segment).__qualname__}", controller=controller_bytes(fc),
        library=library_bytes(api)))
    return stream.getvalue()


def turnkey_restore(blob, segment, fc, api, shared):
    """Put a ``turnkey_blob`` back: the controller and the library's globals in place, the hook's attributes into
    ``segment`` (the caller's object, so that the caller keeps its reference); returns (k, loop quantities)."""
    data = _Unpickler(io.BytesIO(blob), shared_objects(**shared)).load()
    if data.get("format") != FORMAT:
        raise ValueError("not a TurnKey resume blob of this format")
    if (segment is None) != (data["segment"] is None):
        raise ValueError("the saved run and this run differ in having a segment hook")
    if segment is not None:
        name = f"{type(segment).__module__}.{type(segment).__qualname__}"
        if name != data["segment_class"]:
            raise ValueError(f"the segment hook is a {name}, the saved one a {data['segment_class']}")
        vars(segment).clear()
        vars(segment).update(data["segment"])
    restore_controller(fc, data["controller"])
    restore_library(api, data["library"])
    return int(data["k"]), data["loop"]
