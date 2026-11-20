"""A second group of lifting surfaces on the same body, solved as a vortex system of its own (G5 task, pylon contract).

``CompanionSurfaces`` carries one group of lifting surfaces with its own free-wake state beside the group that
``flight.free_flight.run_free`` solves. It hangs on the ``extra_loads`` hook of ``run_free``: at every aerodynamic
step, after the main lifting surfaces are solved, the group is solved at the same pose (position, attitude, velocity
and angular velocity of that step), in the same wind and with the same calls as the main group
(``free_flight._aero``: ``start`` / ``advance``, or ``handover_step`` under a section hand-over with the weight of the
body origin). Its force and its moment about the body origin are returned to be added to the held loads of the step,
with record fields prefixed by the group's name.

**There is no mutual induction between the two groups**: each sees the ambient wind and its own wake only. This is
model (B) of decision 0028 for the pylons of the M600 (the four pylons as an assembly of their own, loads added);
solving them in one system with the wing and the tails is a later step of the plan.

The hook is called once per step, in step order, also while the body is pinned to a reference motion; the group's
wake therefore grows over a lead-in exactly as the main wake does. The hook does not work with the
predictor-corrector scheme (``run_free`` refuses hooks there): a second call within a step would advance the wake
twice.

A failed solve of the group raises the solver's own exception with the attribute ``failed_module`` set to
``"aerodynamics (<name>)"``; ``run_free`` records a failure under that name when the attribute is present. Under zero
seeding a zero-seeded solve that fails is retried once from the previous circulation by the hand-over rule itself and
recorded (``<name>_seeding``, ``<name>_zero_seed_failure``); that is not a failure of the group.

``chain_extra_loads`` joins several hooks into one: forces and moments are added in the order given, record fields
are merged. Mistakes of the set-up (a group called twice for one step, a field written by two hooks or already in the
record) raise ``RuntimeError``, which ``run_free`` does not catch: they end the run as errors of the caller, not as
failed steps.

Record fields: force and moment (inertial and body components), iterations, and the hand-over fields of the group
(weight, closure, seeding, panel counts by weight). The two roughness indicators the main record defines for the
main wing (``stall_switches``, ``main_panels_not_attached``) are not written for a group: they would refer to its
first surface only. The main record's own fields (``force_inertial_N``, ``aero_power_W``, the hand-over fields) stay
those of the main surfaces alone; the group's loads are in its own fields and in the held loads.

Nothing in the package imports this module.
"""

import numpy as np

from ..aero.handover import SectionHandover
from ..aero.methods import AeroMethod
from .free_flight import _aero
from .prescribed import FAILURES, handover_record


class CompanionSurfaces:
    """One more group of lifting surfaces with its own wake; ``extra_loads`` is the hook for ``run_free``."""

    MAIN_WING_ONLY = ("stall_switches", "main_panels_not_attached")          # hand-over fields that describe the main wing

    def __init__(self, name, backend, wing, wind, *, polar=None, handover=None):
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("the group needs a name of letters, digits and underscores (it prefixes its record fields)")
        if (polar is None) == (handover is None):
            raise ValueError("give either the section data or a section hand-over")
        if handover is not None:
            if not isinstance(handover, SectionHandover) or handover.weight_basis != "origin":
                raise ValueError("the hand-over must be a SectionHandover with the weight of the body origin")
            if backend.method != AeroMethod.VSM_FREE_WAKE or backend.wake.solve != handover.low_solve:
                raise ValueError("the hand-over needs the VSM free-wake method with the crosswind solve settings")
        self.name, self.backend, self.wing, self.wind = name, backend, wing, wind
        self.polar, self.handover = polar, handover
        self.state = None                      # aerodynamic state of the last solved step
        self.solves = 0
        self.last_time_s = None

    def extra_loads(self, pose, record):
        """Solve the group at the pose of this step; force and moment about the body origin (inertial), and fields."""
        if self.last_time_s is not None and not pose.time_s > self.last_time_s:
            raise RuntimeError(f"the companion surfaces '{self.name}' were called twice for one step, or out of order")
        try:
            solved, placed, choice, seeding, failure = _aero(self.state, self.backend, self.wing, self.polar, self.handover,
                                                             self.wind, pose)
        except FAILURES as error:
            error.failed_module = f"aerodynamics ({self.name})"
            raise
        self.state, self.solves, self.last_time_s = solved, self.solves+1, float(pose.time_s)
        force = np.array(solved.solution.force, dtype=float)
        moment = np.array(solved.solution.moment(placed, reference=pose.position_m), dtype=float)
        prefix = self.name
        fields = {f"{prefix}_force_inertial_N": force.tolist(), f"{prefix}_moment_origin_inertial_Nm": moment.tolist(),
                  f"{prefix}_force_body_N": (pose.rotation.T@force).tolist(),
                  f"{prefix}_moment_origin_body_Nm": (pose.rotation.T@moment).tolist(),
                  f"{prefix}_iterations": int(solved.solution.iterations)}
        if choice is not None:
            fields.update({f"{prefix}_{key}": value for key, value in handover_record(solved, placed, choice, seeding, failure).items()
                           if key not in self.MAIN_WING_ONLY})
        repeated = set(fields) & set(record)
        if repeated:
            raise RuntimeError(f"the companion surfaces '{self.name}' would overwrite record fields: {sorted(repeated)}")
        return force, moment, fields


def chain_extra_loads(*hooks):
    """One ``extra_loads`` hook from several: loads added in the order given, fields merged, a repeated field refused."""
    hooks = tuple(hook for hook in hooks if hook is not None)
    if not hooks:
        raise ValueError("no hook to chain")

    def chained(pose, record):
        force, moment, fields = np.zeros(3), np.zeros(3), {}
        for hook in hooks:
            added_force, added_moment, added_fields = hook(pose, record)
            force = force+np.asarray(added_force, dtype=float)
            moment = moment+np.asarray(added_moment, dtype=float)
            repeated = set(fields) & set(added_fields)
            if repeated:
                raise RuntimeError(f"two load hooks write the same record fields: {sorted(repeated)}")
            fields.update(added_fields)
        return force, moment, fields
    return chained
