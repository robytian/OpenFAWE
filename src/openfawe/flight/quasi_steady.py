"""OF-0011 revision 7 addendum 12 (conference paper 1's C1-R6): a quasi-steady group of lifting surfaces - the
steady VSM with a straight semi-infinite wake (``aero.vsm.solve_vsm``) solved afresh at every step, with no wake history.

It stands where ``platform_surfaces.PlatformGroup`` (VSM + free wake) stands in the whole-flight hook, takes the same
turned geometry (body axes) and section tables of the step, and returns the same kind of force, moment (about the body
origin, inertial axes) and fields. The ambient velocity at the three-quarter-chord closure points and at the quarter-
chord load points is the wind there, less the body's velocity there, plus ``induction(points)`` - the rotor rings,
given by the hook (surface core). The wake direction is ``solve_vsm``'s: the normalised mean of the closure-point
ambient. The previous step's circulation seeds the solve (iterations only). ``induced_at`` gives the solved system's
velocity (bound vortices and the straight trailing legs) at other points - the hubs of the induced rotor inflow.
"""
import numpy as np

from ..aero.handover import TAKEOFF_LANDING_SOLVE
from ..aero.vsm import geometry, horseshoe_influence, solve_vsm


class QuasiSteadyGroup:
    WAKE_MODEL = "quasi_steady"

    def __init__(self, name, wind, *, density_kg_m3, solve=TAKEOFF_LANDING_SOLVE, induction=None):
        if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
            raise ValueError("the group needs a name of letters, digits and underscores (it prefixes its record fields)")
        self.name, self.wind, self.solve = name, wind, solve
        self.density = float(density_kg_m3)
        self.induction = induction
        self.state = None                       # no wake state: the hook's rings stay standalone
        self.last_time_s = None
        self.solves = 0
        self.last = None                        # (placed geometry, solution) of the last step

    def _wind_at(self, points, time_s):
        if callable(self.wind):
            return np.asarray(self.wind(points, time_s), dtype=float)
        return np.broadcast_to(np.asarray(self.wind, dtype=float), points.shape).copy()

    def ambient(self, pose, points):
        """Wind less body velocity plus the given induction, at ``points`` (inertial)."""
        value = self._wind_at(points, pose.time_s) - np.asarray(pose.point_velocity(points), dtype=float)
        if self.induction is not None:
            value = value + np.asarray(self.induction(points), dtype=float)
        return value

    def extra_loads(self, pose, record, wing, polar):
        if self.last_time_s is not None and not pose.time_s > self.last_time_s:
            raise RuntimeError(f"the quasi-steady group '{self.name}' was called twice for one step, or out of order")
        placed = wing.transformed(pose.rotation, pose.position_m)
        closure_points = geometry(placed)[0]
        load_points = np.asarray(placed.cp, dtype=float)
        initial = None if self.last is None else np.asarray(self.last[1].circulation, dtype=float)
        if initial is not None and initial.shape != (placed.panels,):
            initial = None
        solution = solve_vsm(placed, polar, self.ambient(pose, closure_points),
                             load_relative_ambient=self.ambient(pose, load_points), density=self.density,
                             settings=self.solve, initial=initial)
        self.last, self.last_time_s, self.solves = (placed, solution), float(pose.time_s), self.solves + 1
        force = np.array(solution.force, dtype=float)
        moment = np.array(solution.moment(placed, reference=pose.position_m), dtype=float)
        p = self.name
        fields = {f"{p}_force_inertial_N": force.tolist(), f"{p}_moment_origin_inertial_Nm": moment.tolist(),
                  f"{p}_force_body_N": (pose.rotation.T@force).tolist(),
                  f"{p}_moment_origin_body_Nm": (pose.rotation.T@moment).tolist(),
                  f"{p}_iterations": int(solution.iterations), f"{p}_wake_model": self.WAKE_MODEL,
                  f"{p}_wake_direction": np.asarray(solution.wake_direction, dtype=float).tolist()}
        return force, moment, fields

    def induced_at(self, points):
        """Velocity of the last step's bound vortices and straight trailing legs at ``points`` (inertial); zeros before
        the first step."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.last is None:
            return np.zeros_like(points)
        placed, solution = self.last
        matrix = horseshoe_influence(placed, np.asarray(solution.wake_direction, dtype=float), points)
        return np.einsum("ijk,j->ik", matrix, np.asarray(solution.circulation, dtype=float))
