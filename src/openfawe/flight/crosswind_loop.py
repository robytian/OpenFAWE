# SPDX-License-Identifier: Apache-2.0
# The playbook construction and the nominal-loop law are transcribed from the
# Makani flight controller, google/makani commit 1d66923 (Apache-2.0):
# config/m600/control/crosswind_playbook_utils.py, control/crosswind/
# {crosswind_util,crosswind_frame,crosswind_path,crosswind_curvature,
# crosswind_power,crosswind_playbook}.c, control/simple_aero.c,
# common/c_math/{geometry,util}.c. The source anchors are given at each
# function. Modifications (2026): NumPy/SciPy implementation; the loop angle
# is advanced in time from the commanded kinematics (the controller tracks a
# measured position); the circle may be placed on the tether sphere instead of
# at the tether length (``centre_distance_m``); the tables may be read through a
# periodic cubic spline instead of the controller's linear interpolation
# (``interpolation``); the attitude rate and the acceleration are taken by
# fourth-order differences of the sampled law with a small time step instead
# of the controller's feed-forward differences.
"""Nominal crosswind loop of the Makani M600 controller as a prescribed rigid motion (G4 case A5).

What the motion is: the body origin runs on a circle of the playbook radius in the plane normal to the
line from the ground point to the path centre, with the kite speed that gives the commanded airspeed in
the uniform wind (``CalcCrosswindKinematics``); the attitude is the controller's commanded attitude: the
fly frame (x into the apparent wind, y along x cross the sphere normal) turned by the tether roll about
x, then by minus the sideslip about z and by the incidence about y (``CalcCrosswindAttitude``,
``CalcDcmFToB`` with the XZY order). The tether roll is the controller's command for the nominal path:
the geometric curvature -1/radius with the gravity and apparent-wind corrections removed, the force
balance ``CurvatureToTetherRoll`` with the simple aerodynamic model, the feed-forward term, and the
saturation to the nominal roll plus or minus the tension-dependent excursion. Nothing here is a flight
record: it is the controller's idealised target, which the real kite only tracks.

Frames. Makani's ground frame g is north-east-down with the ground-station origin; the OpenFAWE inertial
frame W is ``W = diag(1, -1, -1) g`` (x the same horizontal axis, z up). Makani's body frame (x forward,
y starboard, z down) maps to the case-file body axes (x chordwise towards the trailing edge, y spanwise,
z up) by ``diag(-1, 1, -1)``. The incidence is the same in both; the sideslip changes sign
(``openfawe.flight.kinematics.relative_wind_body`` gives ``-beta`` of Makani).

Sources, commit 1d66923: see the file header; the G4 contract of task OF-0007-g4motion20261002 (case A5)
says which settings are used in which run.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np
from scipy import interpolate

from .kinematics import RigidMotion, rotation_vector

TABLE_LENGTH = 50                                       # CROSSWIND_SCHEDULE_TABLE_LENGTH
MAKANI_G_TO_WORLD = np.diag([1.0, -1.0, -1.0])          # north-east-down -> x, -y, z up
MAKANI_BODY_TO_BODY = np.diag([-1.0, 1.0, -1.0])        # forward/starboard/down -> chord-aft/span/up
LOOP_DIRECTION_SIGN = -1.0                              # kLoopDirectionCw (crosswind_path.c, CalcGravityCurvature)
DBL_TOL = 1e-9                                          # common/c_math/util.h line 36 (was 1e-12 here until 2026-10-07)


# ----------------------------------------------------------------------------------------------------
# common/c_math/util.c
# ----------------------------------------------------------------------------------------------------
def wrap(x, left, right):
    """``Wrap``: x into [left, right)."""
    span = right-left
    value = math.fmod(x-left, span)
    return value+left if value >= 0.0 else value+right


def saturate(x, low, high):
    """``Saturate``."""
    if low > high:
        raise ValueError("saturation bounds are reversed")
    return min(max(x, low), high)


def crossfade(y0, y1, x, x_low, x_high):
    """``Crossfade``: linear blend of y0 (at x_low and below) and y1 (at x_high and above)."""
    if x_low > x_high:
        raise ValueError("crossfade bounds are reversed")
    span = x_high-x_low
    if span > 0.0:
        c = (x-x_low)/span
    else:                                   # the source divides by DBL_MIN: 0 at x_low, +-inf either side
        c = 0.0 if x == x_low else math.copysign(math.inf, x-x_low)
    c = min(max(c, 0.0), 1.0)
    return (1.0-c)*y0+c*y1


def circular_interp1(x, y, x_i):
    """``CircularInterp1``: wrap x_i into [x[0], x[-1]] and interpolate linearly (y[0] must equal y[-1])."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if y[0] != y[-1]:
        raise ValueError("a circular table must have equal end values")
    return float(np.interp(wrap(float(x_i), x[0], x[-1]), x, y))


def angle_to_dcm_xzy(r1, r2, r3):
    """``AngleToDcm(r1, r2, r3, kRotationOrderXzy)``: rotation about x by r1, then z by r2, then y by r3.

    The result maps parent-frame components to rotated-frame components (a direction-cosine matrix).
    """
    s1, c1, s2, c2, s3, c3 = math.sin(r1), math.cos(r1), math.sin(r2), math.cos(r2), math.sin(r3), math.cos(r3)
    return np.array([[c2*c3, c1*s2*c3+s1*s3, s1*s2*c3-c1*s3],
                     [-s2, c1*c2, s1*c2],
                     [c2*s3, c1*s2*s3-s1*c3, s1*s2*s3+c1*c3]])


# ----------------------------------------------------------------------------------------------------
# config/m600/control/crosswind_playbook_utils.py
# ----------------------------------------------------------------------------------------------------
def periodic_spline_table(control_points, table_length=TABLE_LENGTH):
    """``GeneratePeriodicSplineLookupTable``: 50 samples of the periodic spline through the control points.

    The control points sit at the centres of equal segments of the loop in the FBL convention (clockwise,
    zero at 12 o'clock, half-segment offset); the controller loop angle x maps to the FBL fraction
    ``(-x + 1.5 pi) / (2 pi)``. The last sample is set equal to the first.
    """
    points = [float(p) for p in control_points]
    n = len(points)
    spline_x = np.linspace(0.0, 1.0, n+1, endpoint=True)-0.5/n
    spline_y = np.array([points[-1]]+points)
    spline = interpolate.CubicSpline(spline_x, spline_y, bc_type="periodic", extrapolate="periodic")
    lookup_x = np.linspace(0.0, 2.0*np.pi, table_length)
    lookup_y = spline((-lookup_x+1.5*np.pi)/(2.0*np.pi))
    if abs(lookup_y[-1]-lookup_y[0]) >= 1e-6:
        raise ValueError("the periodic spline table does not close")
    lookup_y[-1] = lookup_y[0]
    return lookup_y.tolist()


def convolve_smooth(array, weights, edges=None):
    """``ConvolveSmooth`` with ``allow_scaling=False``: normalised weights, odd kernel, edge rules per axis.

    Edges: ``extend`` repeats the end values, ``continuous`` wraps assuming the end values overlap
    (``a[0] == a[-1]``), ``cutoff`` drops points outside and rescales the weights that remain.
    """
    a = np.array(array, dtype=float)
    w = np.array(weights, dtype=float)
    if w.ndim != a.ndim:
        raise ValueError("weights and array must have the same number of dimensions")
    if any(dim % 2 == 0 for dim in w.shape):
        raise ValueError("each dimension of weights must be odd")
    w = w/np.sum(w)
    scaling = float(np.sum(w))
    middle = [(dim-1)//2 for dim in w.shape]
    edges = ["extend"]*a.ndim if edges is None else list(edges)
    if len(edges) != a.ndim:
        raise ValueError("one edge rule per dimension is required")
    if any(edge not in ("extend", "continuous", "cutoff") for edge in edges):
        raise ValueError("edge rules are extend, continuous or cutoff")
    out = np.zeros(a.shape)
    for a_index in np.ndindex(*a.shape):
        w_sum = 0.0
        total = 0.0
        for w_index in np.ndindex(*w.shape):
            adjusted = [a_index[e]+w_index[e]-middle[e] for e in range(a.ndim)]
            cut_off = False
            for e, edge in enumerate(edges):
                if edge == "cutoff":
                    if adjusted[e] < 0 or adjusted[e] > a.shape[e]-1:
                        cut_off = True
                elif edge == "continuous":
                    if adjusted[e] >= a.shape[e]-1:
                        adjusted[e] -= a.shape[e]-1
                    elif adjusted[e] < 0:
                        adjusted[e] -= 1
                else:
                    adjusted[e] = int(np.clip(adjusted[e], 0, a.shape[e]-1))
            if not cut_off:
                w_sum += w[w_index]
                total += a[tuple(adjusted)]*w[w_index]
        out[a_index] = total*scaling/w_sum
    return out.tolist()


def adjust_wind_speed(wind_speed, min_offset, crossover, scaling_factor):
    """``_AdjustWindSpeed`` of ``MakePlaybook``."""
    return max(wind_speed+min_offset, crossover+(wind_speed-crossover)*scaling_factor)


LOOKUP_KEYS = ("alpha_lookup", "beta_lookup", "airspeed_lookup", "transout_airspeed_lookup")
SCALAR_KEYS = ("path_radius_target", "elevation", "azi_offset")


def build_playbook(rows, smoothing_params, wind_adjust_params, transout_airspeeds=None, table_length=TABLE_LENGTH):
    """``MakePlaybook`` followed by ``SmoothPlaybook``: the entries the controller interpolates between.

    ``rows`` are the parameterised solutions in ascending wind speed (keys ``wind_speed``, ``alpha``,
    ``beta``, ``airspeed``, ``radius``, ``elevation``, ``azi_offset``); ``transout_airspeeds`` one list of
    control points per row (the trans-out schedule; it only enters its own table). Returns a list of
    entries with the adjusted ``wind_speed``, the four 50-point tables, the three scalars and the loop
    angle grid; ``raw_entries`` of the result's second element are the tables before smoothing.
    """
    if transout_airspeeds is not None and len(transout_airspeeds) != len(rows):
        raise ValueError("one trans-out schedule per playbook row is required")
    entries = []
    last = 0.0
    for k, row in enumerate(rows):
        if not row["wind_speed"] > last:
            raise ValueError("playbook rows must be in strictly ascending wind speed")
        transout = transout_airspeeds[k] if transout_airspeeds is not None else row.get("transout_airspeed")
        entries.append(dict(wind_speed=adjust_wind_speed(row["wind_speed"], **wind_adjust_params),
                            alpha_lookup=periodic_spline_table(row["alpha"], table_length),
                            beta_lookup=periodic_spline_table(row["beta"], table_length),
                            airspeed_lookup=periodic_spline_table(row["airspeed"], table_length),
                            transout_airspeed_lookup=(periodic_spline_table(transout, table_length) if transout is not None
                                                      else [0.0]*table_length),
                            path_radius_target=float(row["radius"]), elevation=float(row["elevation"]),
                            azi_offset=float(row["azi_offset"]),
                            lookup_loop_angle=[2.0*np.pi*i/(table_length-1) for i in range(table_length)]))
        last = row["wind_speed"]
    raw = [dict(entry) for entry in entries]
    smoothed = {}
    for key, rule in smoothing_params.items():
        if key not in LOOKUP_KEYS+SCALAR_KEYS:
            raise ValueError(f"unknown playbook key {key!r} in the smoothing parameters")
        column = [entry[key] for entry in entries]
        smoothed[key] = convolve_smooth(column, rule["weights"], edges=rule.get("edges"))
    for key, values in smoothed.items():
        for k, value in enumerate(values):
            entries[k][key] = value
    return entries, raw


def playbook_entry(entries, wind_speed):
    """``GetNominalPlaybookEntry``: the entry at ``wind_speed`` by linear blend of the bracketing rows.

    Below the first row the first row is used; at or above the last row the last row.
    """
    lower = upper = 0
    for k, entry in enumerate(entries):
        if wind_speed >= entry["wind_speed"]:
            lower, upper = k, k+1
    if upper >= len(entries):
        upper = lower
    low, high = entries[lower], entries[upper]
    x_low, x_high = low["wind_speed"], high["wind_speed"]
    result = dict(lookup_loop_angle=list(low["lookup_loop_angle"]), wind_speed=float(wind_speed),
                  rows_blended=(lower, upper))
    for key in LOOKUP_KEYS:
        result[key] = [crossfade(a, b, wind_speed, x_low, x_high) for a, b in zip(low[key], high[key])]
    for key in SCALAR_KEYS:
        result[key] = crossfade(low[key], high[key], wind_speed, x_low, x_high)
    return result


def load_playbook(path):
    """The land playbook case file: returns (entries, raw entries, the file's dictionary)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "openfawe.crosswind_playbook.v1":
        raise ValueError("not an openfawe.crosswind_playbook.v1 file")
    entries, raw = build_playbook(data["playbook_parameterized"], data["smoothing_params"], data["wind_adjust_params"],
                                  data.get("transout_airspeeds"), data.get("table_length", TABLE_LENGTH))
    return entries, raw, data


# ----------------------------------------------------------------------------------------------------
# The nominal-loop law
# ----------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimpleAero:
    """``SimpleAeroModelParams`` of control/simple_aero.c (lift and side force only are used here)."""
    dCL_dalpha: float = 7.39413425
    CL_0: float = 2.02175
    dCY_dbeta: float = -1.4205533
    CY_0: float = -0.08687

    def CL(self, alpha):
        return self.dCL_dalpha*alpha+self.CL_0

    def CY(self, beta):
        return self.dCY_dbeta*beta+self.CY_0


@dataclass(frozen=True)
class LoopConstants:
    """System and controller constants of the law (defaults: the M600 land case file, see its provenance)."""
    wing_mass_kg: float = 1730.6
    tether_linear_density_kg_m: float = 0.917
    tether_length_m: float = 425.8
    reference_area_m2: float = 32.9
    air_density_kg_m3: float = 1.026
    gravity_m_s2: float = 9.81
    simple_aero: SimpleAero = field(default_factory=SimpleAero)
    min_airspeed_m_s: float = 25.0
    max_airspeed_m_s: float = 80.0
    alpha_min_rad: float = -math.radians(2.0)
    dalpha_dairspeed: float = -math.radians(8.0/15.0)
    alpha_min_airspeed_m_s: float = 75.0
    alpha_cmd_min_rad: float = -0.2
    alpha_cmd_max_rad: float = math.radians(12.0)
    beta_cmd_max_rad: float = math.radians(4.0)
    tether_roll_nom_rad: float = math.atan2(0.5, 4.7860+0.13035)
    tether_roll_max_excursion_low_rad: float = 0.2
    tether_roll_max_excursion_high_rad: float = 0.48
    tether_roll_tension_low_N: float = 5000.0
    tether_roll_tension_high_N: float = 20000.0
    tether_roll_ff_amplitude_rad: float = math.radians(5.0)
    tether_roll_ff_phase_rad: float = math.radians(45.0)
    bridle_radius_m: float = 4.7860

    @classmethod
    def from_case(cls, data):
        """Constants from the ``controller_constants`` and ``system_constants`` of a playbook case file."""
        c, s = data["controller_constants"], data["system_constants"]
        aero = c["simple_aero_model"]
        bridle = s["bridle_pos_m"]
        return cls(wing_mass_kg=s["wing_mass_kg"], tether_linear_density_kg_m=s["tether_linear_density_kg_m"],
                   tether_length_m=s["tether_length_m"], reference_area_m2=s["reference_area_m2"],
                   air_density_kg_m3=s["air_density_kg_m3"], gravity_m_s2=s["gravity_m_s2"],
                   simple_aero=SimpleAero(aero["dCL_dalpha"], aero["CL_0"], aero["dCY_dbeta"], aero["CY_0"]),
                   min_airspeed_m_s=c["min_airspeed_m_s"], max_airspeed_m_s=c["max_airspeed_m_s"],
                   alpha_min_rad=c["alpha_min_rad"], dalpha_dairspeed=c["dalpha_dairspeed_rad_per_m_s"],
                   alpha_min_airspeed_m_s=c["alpha_min_airspeed_m_s"], alpha_cmd_min_rad=c["alpha_cmd_min_rad"],
                   alpha_cmd_max_rad=c["alpha_cmd_max_rad"], beta_cmd_max_rad=c["beta_cmd_max_rad"],
                   tether_roll_nom_rad=math.atan2(-s["bridle_y_offset_m"], s["bridle_rad_m"]+bridle[1][2]),
                   tether_roll_max_excursion_low_rad=c["tether_roll_max_excursion_low_rad"],
                   tether_roll_max_excursion_high_rad=c["tether_roll_max_excursion_high_rad"],
                   tether_roll_tension_low_N=c["tether_roll_tension_low_N"], tether_roll_tension_high_N=c["tether_roll_tension_high_N"],
                   tether_roll_ff_amplitude_rad=c["tether_roll_ff_amplitude_rad"], tether_roll_ff_phase_rad=c["tether_roll_ff_phase_rad"],
                   bridle_radius_m=s["bridle_rad_m"])

    @property
    def effective_mass_kg(self):
        """Wing plus a third of the tether (``CurvatureToTetherRoll``, ``CalcGravityCurvature``)."""
        return self.wing_mass_kg+self.tether_linear_density_kg_m*self.tether_length_m/3.0


def sph_to_vec_g(azimuth, elevation, radius):
    """``SphToVecG`` with ``azi_ref_offset`` already removed: azimuth from +x_g, positive elevation towards -z_g."""
    r_cos = radius*math.cos(-elevation)
    return np.array([r_cos*math.cos(azimuth), r_cos*math.sin(azimuth), radius*math.sin(-elevation)])


def dcm_g_to_cw(path_centre_g):
    """``CalcDcmGToCw``: x towards the ground station from the path centre, y = z_g x x normalised, z = x x y."""
    ex = -np.asarray(path_centre_g, dtype=float)
    ex = ex/max(np.linalg.norm(ex), DBL_TOL)
    ey = np.cross([0.0, 0.0, 1.0], ex)
    ey = ey/max(np.linalg.norm(ey), DBL_TOL)
    ez = np.cross(ex, ey)
    return np.array([ex, ey, ez])


def dcm_f_to_g(apparent_wind_g, anchor_to_kite_g):
    """``CalcDcmFToG``: fly frame x into the apparent wind, y along x cross the sphere normal, as columns."""
    i_f = -np.asarray(apparent_wind_g, dtype=float)
    i_f = i_f/np.linalg.norm(i_f)
    normal = np.asarray(anchor_to_kite_g, dtype=float)
    normal = normal/np.linalg.norm(normal)
    j_f = np.cross(i_f, normal)
    j_f = j_f/np.linalg.norm(j_f)
    k_f = np.cross(i_f, j_f)
    k_f = k_f/np.linalg.norm(k_f)
    return np.column_stack([i_f, j_f, k_f])


def curvature_to_tether_roll(aero_curvature, lift_coeff, side_force_coeff, constants):
    """``CurvatureToTetherRoll``: the force balance ``k_aero = (CY - CL roll) / (m_eff / (rho A / 2))``."""
    if not lift_coeff > 0.0:
        raise ValueError("the lift coefficient must be positive")
    acc_coeff = aero_curvature*constants.effective_mass_kg/(0.5*constants.air_density_kg_m3*constants.reference_area_m2)
    return saturate((side_force_coeff-acc_coeff)/max(lift_coeff, np.finfo(float).eps), -np.pi/2.0, np.pi/2.0)


class NominalLoop:
    """The controller's nominal loop at one wind speed: schedules, kinematics, attitude and a sampled motion.

    ``entry`` is a playbook entry (``playbook_entry``), ``wind_g`` the uniform wind vector in Makani's
    ground frame (the direction the air moves; the playbook wind speed is its norm unless
    ``wind_speed_for_playbook`` is given). ``centre_distance_m`` places the path centre: ``None`` is
    Makani's ``SphToVecG(azimuth, elevation, tether length)``, i.e. the centre at the tether length and the
    kite at ``hypot(length, radius)`` from the ground point, further than the tether and bridle reach; a
    number puts the centre at that distance (``sphere_centre_distance`` gives the one that keeps the kite on
    a sphere of a given radius). ``interpolation`` reads the 50-point tables as the controller does
    (``linear``, ``CircularInterp1``) or through a periodic cubic spline through the same 50 points
    (``spline``, a C2 schedule so that the finite-difference checks of the motion samples apply; the
    difference between the two readings is reported by ``interpolation_difference``).
    ``roll_limit_tension_N`` is the tether tension the saturation band of the roll command is evaluated
    at (the controller uses the measured tension); ``None`` uses the wide band of a load-cell fault.
    """

    def __init__(self, entry, wind_g, constants=LoopConstants(), *, centre_distance_m=None, interpolation="spline",
                 roll_limit_tension_N=20000.0, wind_speed_for_playbook=None):
        if interpolation not in ("linear", "spline"):
            raise ValueError("interpolation is 'linear' or 'spline'")
        self.entry = entry
        self.constants = constants
        self.interpolation = interpolation
        self.wind_g = np.array(wind_g, dtype=float).reshape(3)
        self.wind_speed = float(np.linalg.norm(self.wind_g)) if wind_speed_for_playbook is None else float(wind_speed_for_playbook)
        if self.wind_speed <= 0.0:
            raise ValueError("the loop needs a positive wind speed")
        self.roll_limit_tension_N = roll_limit_tension_N
        self.radius_m = float(entry["path_radius_target"])
        self.elevation_rad = float(entry["elevation"])
        self.azi_offset_rad = float(entry["azi_offset"])
        self.wind_direction_rad = math.atan2(self.wind_g[1], self.wind_g[0])       # direction the wind points to
        self.centre_distance_m = constants.tether_length_m if centre_distance_m is None else float(centre_distance_m)
        if not self.centre_distance_m > 0.0:
            raise ValueError("the path centre distance must be positive")
        self.path_centre_g = sph_to_vec_g(wrap(self.wind_direction_rad+self.azi_offset_rad, -np.pi, np.pi),
                                          self.elevation_rad, self.centre_distance_m)
        self.dcm_g2cw = dcm_g_to_cw(self.path_centre_g)
        self.loop_angles = np.asarray(entry["lookup_loop_angle"], dtype=float)
        self.d_loop_angle = float(self.loop_angles[1]-self.loop_angles[0])
        if not np.allclose(np.diff(self.loop_angles), self.d_loop_angle, rtol=0.0, atol=1e-12):
            raise ValueError("the loop angle grid must be equispaced")
        self._tables = {}
        for key in ("alpha_lookup", "beta_lookup", "airspeed_lookup"):
            values = np.asarray(entry[key], dtype=float)
            if values[0] != values[-1]:
                raise ValueError(f"{key} is not periodic")
            self._tables[key] = (values, interpolate.CubicSpline(self.loop_angles, values, bc_type="periodic")
                                 if interpolation == "spline" else None)

    # ----- geometry ----------------------------------------------------------------------------------
    @property
    def kite_on_sphere_m(self):
        """Distance of the body origin from the ground point, the same at every loop angle."""
        return math.hypot(self.centre_distance_m, self.radius_m)

    @staticmethod
    def sphere_centre_distance(sphere_radius_m, loop_radius_m):
        """Centre distance that keeps a circle of ``loop_radius_m`` on the sphere of ``sphere_radius_m``."""
        if not sphere_radius_m > loop_radius_m:
            raise ValueError("the sphere must be larger than the loop")
        return math.sqrt(sphere_radius_m**2-loop_radius_m**2)

    # ----- schedules ---------------------------------------------------------------------------------
    def _lookup(self, key, loop_angle):
        values, spline = self._tables[key]
        angle = wrap(float(loop_angle), 0.0, 2.0*np.pi)
        if spline is None:
            return float(np.interp(angle, self.loop_angles, values))
        return float(spline(angle))

    def airspeed(self, loop_angle):
        """``SetAirspeed`` on the normal path: the table, saturated to the airspeed limits."""
        return saturate(self._lookup("airspeed_lookup", loop_angle), self.constants.min_airspeed_m_s, self.constants.max_airspeed_m_s)

    def d_airspeed_d_loop_angle(self, loop_angle):
        """``GetAirspeedDerivative``: central difference with the table spacing as the step (the controller's feed-forward)."""
        h = self.d_loop_angle
        return (self.airspeed(wrap(loop_angle+h, 0.0, 2.0*np.pi))-self.airspeed(wrap(loop_angle-h, 0.0, 2.0*np.pi)))/(2.0*h)

    def alpha_nom(self, loop_angle):
        return self._lookup("alpha_lookup", loop_angle)

    def beta_nom(self, loop_angle):
        return self._lookup("beta_lookup", loop_angle)

    def alpha_before_saturation(self, loop_angle):
        """``CrosswindCurvatureStep``: the fade of the nominal incidence towards ``alpha_min`` with the airspeed.

        The controller fades with the measured airspeed; here the commanded one stands in for it (the nominal path).
        Returns (incidence, whether the fade is active).
        """
        c = self.constants
        alpha_nom = self.alpha_nom(loop_angle)
        airspeed = self.airspeed(loop_angle)
        alpha_nom_airspeed = c.alpha_min_airspeed_m_s-(c.alpha_min_rad-alpha_nom)/c.dalpha_dairspeed
        alpha = crossfade(alpha_nom, c.alpha_min_rad, airspeed, alpha_nom_airspeed, c.alpha_min_airspeed_m_s)
        return alpha, alpha != alpha_nom

    def alpha_command(self, loop_angle):
        """The incidence command after the saturation to ``[alpha_cmd_min, alpha_cmd_max]``; returns (incidence, fade active)."""
        c = self.constants
        alpha, fading = self.alpha_before_saturation(loop_angle)
        return saturate(alpha, c.alpha_cmd_min_rad, c.alpha_cmd_max_rad), fading

    def beta_command(self, loop_angle):
        """The sideslip command after the saturation to ``+-beta_cmd_max`` (the nominal value before it is ``beta_nom``)."""
        c = self.constants
        return saturate(self.beta_nom(loop_angle), -c.beta_cmd_max_rad, c.beta_cmd_max_rad)

    # ----- kinematics (crosswind_util.c) -------------------------------------------------------------
    def loop_geometry(self, loop_angle):
        """``CalcCircularLoopGeometry``: position, unit velocity and unit centripetal direction in g."""
        pos_cw = np.array([0.0, math.cos(loop_angle), math.sin(loop_angle)])*self.radius_m
        vel_cw = np.array([0.0, math.sin(loop_angle), -math.cos(loop_angle)])
        acc_cw = np.array([0.0, -math.cos(loop_angle), -math.sin(loop_angle)])
        t = self.dcm_g2cw.T
        return t@pos_cw+self.path_centre_g, t@vel_cw, t@acc_cw

    def kite_speed(self, loop_angle):
        """``CalcCrosswindKinematics`` with ``airspeed_dot = 0``: the inertial speed that gives the commanded airspeed."""
        _, vel_hat, _ = self.loop_geometry(loop_angle)
        wind = float(np.linalg.norm(self.wind_g))
        wind_par = float(vel_hat@self.wind_g)
        wind_perp = math.sqrt(max(wind*wind-wind_par*wind_par, 0.0))
        airspeed = self.airspeed(loop_angle)
        if airspeed <= wind_perp:
            raise ValueError("the commanded airspeed is below the crosswind component of the wind: no kite speed exists")
        return wind_par+math.sqrt(airspeed*airspeed-wind_perp*wind_perp)

    def makani_kinematics(self, loop_angle):
        """``CalcCircularLoopKinematics``: velocity and the controller's feed-forward acceleration in g."""
        _, vel_hat, acc_hat = self.loop_geometry(loop_angle)
        kitespeed = self.kite_speed(loop_angle)
        airspeed = self.airspeed(loop_angle)
        airspeed_dot = self.d_airspeed_d_loop_angle(loop_angle)*(-kitespeed/self.radius_m)
        wind_par = float(vel_hat@self.wind_g)
        wind_par_dot = float(acc_hat@self.wind_g)*kitespeed/self.radius_m
        kiteaccel = (airspeed*airspeed_dot+kitespeed*wind_par_dot)/(kitespeed-wind_par)
        return kitespeed*vel_hat, kiteaccel*vel_hat+kitespeed*kitespeed/self.radius_m*acc_hat

    # ----- curvature and tether roll (crosswind_path.c, crosswind_curvature.c) ------------------------
    def gravity_curvature(self, loop_angle, speed):
        """``CalcGravityCurvature`` for the velocity direction at this loop angle."""
        c = self.constants
        vel_cw = np.array([0.0, math.sin(loop_angle), -math.cos(loop_angle)])
        g_cw = self.dcm_g2cw@np.array([0.0, 0.0, c.gravity_m_s2])
        g_cw[0] = 0.0
        g_parallel = float(np.linalg.norm(g_cw))
        gravity_angle = -math.atan2(vel_cw[1], vel_cw[2])
        return LOOP_DIRECTION_SIGN*c.wing_mass_kg/c.effective_mass_kg*g_parallel*math.sin(gravity_angle)/max(speed*speed, 1e-5)

    def apparent_wind_curvature_factor(self, loop_angle, speed):
        """``CalcApparentWindCurvature(1.0, ...)``: the factor that multiplies the aerodynamic curvature."""
        _, vel_hat, _ = self.loop_geometry(loop_angle)
        wind = float(np.linalg.norm(self.wind_g))
        unit_wind = self.wind_g/max(wind, 1e-5)
        wind_angle = math.acos(min(max(float(vel_hat@unit_wind), -1.0), 1.0))
        ratio = wind/max(speed, 1e-5)
        return -2.0*ratio*math.cos(wind_angle)+ratio*ratio

    def aero_curvature(self, loop_angle):
        """``RemoveCurvatureCorrections`` of the geometric curvature ``-1/radius`` (clockwise loop)."""
        speed = self.kite_speed(loop_angle)
        k_geom = -1.0/self.radius_m
        gravity = self.gravity_curvature(loop_angle, speed)
        denominator = 1.0+self.apparent_wind_curvature_factor(loop_angle, speed)
        if abs(denominator) < 1e-5:
            denominator = 1e-5 if denominator > 0.0 else -1e-5
        return (k_geom-gravity)/denominator

    def roll_excursion(self):
        c = self.constants
        if self.roll_limit_tension_N is None:
            return c.tether_roll_max_excursion_high_rad
        return crossfade(c.tether_roll_max_excursion_low_rad, c.tether_roll_max_excursion_high_rad, self.roll_limit_tension_N,
                         c.tether_roll_tension_low_N, c.tether_roll_tension_high_N)

    def tether_roll_command(self, loop_angle):
        """The roll command of ``CrosswindCurvatureStep`` on the normal path; returns (roll, unsaturated roll).

        As in the source, the force balance uses the incidence and sideslip commands before their own saturation
        (the rate limits, which come before, are not transcribed); ``k_aero`` is the geometric curvature of the
        nominal circle with the corrections removed, which stands in for the path controller's command.
        """
        c = self.constants
        alpha, _ = self.alpha_before_saturation(loop_angle)
        beta = self.beta_nom(loop_angle)
        roll = curvature_to_tether_roll(self.aero_curvature(loop_angle), c.simple_aero.CL(alpha), c.simple_aero.CY(beta), c)
        roll += c.tether_roll_ff_amplitude_rad*math.cos(loop_angle+c.tether_roll_ff_phase_rad)
        excursion = self.roll_excursion()
        return saturate(roll, c.tether_roll_nom_rad-excursion, c.tether_roll_nom_rad+excursion), roll

    # ----- attitude ----------------------------------------------------------------------------------
    def attitude_g2b(self, loop_angle):
        """``CalcCrosswindAttitude``: Makani's DCM from g to the body frame."""
        alpha, _ = self.alpha_command(loop_angle)
        beta = self.beta_command(loop_angle)
        roll, _ = self.tether_roll_command(loop_angle)
        dcm_f2b = angle_to_dcm_xzy(-roll, -beta, alpha)
        position, vel_hat, _ = self.loop_geometry(loop_angle)
        velocity = self.kite_speed(loop_angle)*vel_hat
        dcm_f2g = dcm_f_to_g(self.wind_g-velocity, position)
        return dcm_f2b@dcm_f2g.T

    def makani_pqr(self, loop_angle, kitespeed=None):
        """Body rates of the law by the stencil of ``CalcCrosswindPqr``: the attitude at loop angle +- 1e-3 rad.

        This is the angular velocity of the nominal law by a two-point difference, not what the controller computes:
        ``CalcCrosswindPqr`` holds the incidence and sideslip commands fixed over the stencil and varies the roll and the
        airspeed linearly with their rates (and uses the measured tether roll), so its feed-forward lacks the incidence
        and sideslip rate terms (its own TODO) and only its pitch component is used by the controller.
        """
        h = 1e-3
        kitespeed = self.kite_speed(loop_angle) if kitespeed is None else kitespeed
        rate = -kitespeed/self.radius_m
        minus, plus = self.attitude_g2b(loop_angle-h), self.attitude_g2b(loop_angle+h)
        relative = plus@minus.T                                  # body(-h) components -> body(+h) components
        # DcmToAngle(..., kRotationOrderZyx) of a small rotation: the Euler angles are the rotation vector of the
        # transpose (a passive DCM I - [w]x describes the active rotation w); rate = vector * d(loop)/dt / (2h).
        return rotation_vector(relative.T)*rate/(2.0*h)

    def body_rates_to_world(self, loop_angle, pqr_body):
        """Makani body rates (Makani body components) as an angular velocity in the OpenFAWE inertial frame."""
        return MAKANI_G_TO_WORLD@self.attitude_g2b(loop_angle).T@np.asarray(pqr_body, dtype=float)

    # ----- OpenFAWE frames ---------------------------------------------------------------------------
    def pose_world(self, loop_angle):
        """Position (m) and body-to-world rotation of the body origin in the OpenFAWE inertial frame."""
        position, _, _ = self.loop_geometry(loop_angle)
        g2b = self.attitude_g2b(loop_angle)
        return MAKANI_G_TO_WORLD@position, MAKANI_G_TO_WORLD@g2b.T@MAKANI_BODY_TO_BODY.T

    def wind_world(self):
        return MAKANI_G_TO_WORLD@self.wind_g

    def ground_point_world(self):
        return np.zeros(3)

    # ----- loop angle in time ------------------------------------------------------------------------
    def _lap_time_between(self, angle_from, angle_to, order=24):
        """Time to fly from ``angle_from`` down to ``angle_to``: integral of R / kitespeed by one Gauss-Legendre rule.

        Accurate only when no table knot lies strictly inside the interval (the integrand is smooth between
        knots and only C2, or C0 with the linear reading, across them); the callers split at the knots.
        """
        nodes, weights = np.polynomial.legendre.leggauss(order)
        a, b = angle_to, angle_from
        theta = 0.5*(b-a)*nodes+0.5*(a+b)
        return 0.5*(b-a)*float(np.sum(weights*np.array([self.radius_m/self.kite_speed(t) for t in theta])))

    def _knot_times(self, start_angle):
        """Cumulative time at every table knot crossed in one lap, flying from ``start_angle`` downwards."""
        knots = []
        k = math.floor(start_angle/self.d_loop_angle)
        angle = start_angle
        bounds = [start_angle]
        while angle > start_angle-2.0*np.pi+1e-12:
            next_knot = k*self.d_loop_angle
            if abs(next_knot-angle) < 1e-12:
                k -= 1
                next_knot = k*self.d_loop_angle
            next_angle = max(next_knot, start_angle-2.0*np.pi)
            bounds.append(next_angle)
            knots.append(self._lap_time_between(angle, next_angle))
            angle = next_angle
            k -= 1
        return np.array(bounds), np.concatenate([[0.0], np.cumsum(knots)])

    def period_s(self, start_angle=0.0):
        """Time of one lap."""
        _, times = self._knot_times(start_angle)
        return float(times[-1])

    def loop_angle_at(self, time_s, start_angle=0.0, bounds=None, knot_times=None):
        """Loop angle (unwrapped, decreasing from ``start_angle``) at ``time_s`` >= 0 by Newton on the lap time."""
        if bounds is None:
            bounds, knot_times = self._knot_times(start_angle)
        period = knot_times[-1]
        laps, remainder = divmod(float(time_s), period)
        if remainder > period-1e-9*period:                      # the end of a lap: avoid a zero-length interval
            laps, remainder = laps+1.0, 0.0
        k = int(np.searchsorted(knot_times, remainder, side="right")-1)
        k = min(max(k, 0), len(knot_times)-2)
        lo, hi = bounds[k], bounds[k+1]                          # lo > hi: the angle decreases
        target = remainder-knot_times[k]
        angle = lo-(lo-hi)*target/max(knot_times[k+1]-knot_times[k], 1e-300)
        for _ in range(50):
            f = self._lap_time_between(lo, angle)-target
            step = f*self.kite_speed(angle)/self.radius_m    # d t / d angle = -R / speed
            angle = min(max(angle+step, hi), lo)
            if abs(f) < 1e-13*max(1.0, period):
                break
        return angle-2.0*np.pi*laps

    def interpolation_difference(self):
        """Largest difference between the spline and the linear reading of the three tables over the loop."""
        angles = np.linspace(0.0, 2.0*np.pi, 4901)
        out = {}
        for key in ("alpha_lookup", "beta_lookup", "airspeed_lookup"):
            values, _ = self._tables[key]
            spline = interpolate.CubicSpline(self.loop_angles, values, bc_type="periodic")
            linear = np.interp(angles, self.loop_angles, values)
            out[key] = float(np.max(np.abs(spline(angles)-linear)))
        return out

    # ----- the sampled motion ------------------------------------------------------------------------
    def motion(self, laps, steps_per_lap, *, start_angle=0.0, difference_step_s=1e-3, store_angular_acceleration=False):
        """``RigidMotion`` of ``laps`` laps with ``steps_per_lap`` equal time steps per lap, and the command table.

        The time step is the lap time divided by ``steps_per_lap``, so that the same phase of consecutive
        laps falls on a sample (the samples of lap n are those of lap 1 shifted by 2 pi). Velocity,
        acceleration, angular velocity and angular acceleration are fourth-order central differences of the
        law in time with the step ``difference_step_s`` (position and attitude are exact at the samples).
        The angular acceleration is kept in the table and stored with the motion only on request: with the
        spline reading of the tables the attitude is C2, so the angular acceleration is only continuous
        (kinks at the 49 table knots) and a difference of the sampled angular velocity over the run's
        time step does not reproduce it to the tolerance of the sample check (measured 1.7e-4 relative
        at 0.02 s against 1e-7 for the angular velocity); the residual account then takes it from the
        samples. The table holds, per sample, the loop angle and the commands (airspeed, kite speed,
        incidence, sideslip, tether roll before and after saturation) and the flags of the active
        saturations.
        """
        if int(laps) < 1 or int(steps_per_lap) < 2:
            raise ValueError("at least one lap and two steps per lap")
        laps, steps_per_lap = int(laps), int(steps_per_lap)
        bounds, knot_times = self._knot_times(start_angle)
        period = float(knot_times[-1])
        dt = period/steps_per_lap
        lap_angles = np.array([self.loop_angle_at(k*dt, start_angle, bounds, knot_times) for k in range(steps_per_lap)])
        angles = np.concatenate([lap_angles-2.0*np.pi*lap for lap in range(laps)]+[[start_angle-2.0*np.pi*laps]])
        times = np.arange(laps*steps_per_lap+1)*dt
        h = float(difference_step_s)
        rows = []

        def angle_at(time):
            # The loop angle is a function of time only; before the first sample the lap before is used.
            if time >= 0.0:
                return self.loop_angle_at(time, start_angle, bounds, knot_times)
            return self.loop_angle_at(time+period, start_angle, bounds, knot_times)+2.0*np.pi

        def omega_at(poses, k):
            # Fourth-order central difference of the attitude at stencil index k (poses at k-2 .. k+2).
            phi1 = rotation_vector(poses[k+1][1]@poses[k-1][1].T)
            phi2 = rotation_vector(poses[k+2][1]@poses[k-2][1].T)
            return (8.0*phi1-phi2)/(12.0*h)

        for angle, time in zip(angles, times):
            position, rotation = self.pose_world(angle)
            # Nine poses at t + m h, m = -4..4 (the centre is the sample itself): velocity and acceleration from the
            # inner five, the angular velocity from the inner five attitudes, the angular acceleration from the
            # angular velocities at m = -2..2, each of which needs its own five attitudes.
            poses = [self.pose_world(angle_at(time+m*h)) if m else (position, rotation) for m in range(-4, 5)]
            p = [pose[0] for pose in poses]
            velocity = (-p[6]+8.0*p[5]-8.0*p[3]+p[2])/(12.0*h)
            acceleration = (-p[6]+16.0*p[5]-30.0*p[4]+16.0*p[3]-p[2])/(12.0*h*h)
            omegas = [omega_at(poses, k) for k in range(2, 7)]
            omega = omegas[2]
            omega_dot = (-omegas[4]+8.0*omegas[3]-8.0*omegas[1]+omegas[0])/(12.0*h)
            alpha, fading = self.alpha_command(angle)
            roll, roll_raw = self.tether_roll_command(angle)
            rows.append(dict(position=position, rotation=rotation, velocity=velocity, acceleration=acceleration, omega=omega,
                             omega_dot=omega_dot,
                             loop_angle=wrap(angle, 0.0, 2.0*np.pi), airspeed=self.airspeed(angle), kite_speed=self.kite_speed(angle),
                             alpha=alpha, beta=self.beta_command(angle), roll=roll, roll_unsaturated=roll_raw,
                             alpha_fading=bool(fading), roll_saturated=bool(roll != roll_raw),
                             airspeed_saturated=bool(self.airspeed(angle) != self._lookup("airspeed_lookup", angle))))
        omega = np.array([r["omega"] for r in rows])
        omega_dot = np.array([r["omega_dot"] for r in rows])
        motion = RigidMotion(times, np.array([r["position"] for r in rows]), np.array([r["rotation"] for r in rows]),
                             np.array([r["velocity"] for r in rows]), omega, np.array([r["acceleration"] for r in rows]),
                             omega_dot if store_angular_acceleration else None,
                             description=(f"Makani nominal crosswind loop, wind {self.wind_speed:g} m/s, radius {self.radius_m:g} m, "
                                          f"{laps} laps of {steps_per_lap} steps, dt {dt:.6f} s, {self.interpolation} tables"))
        table = dict(time_s=times.tolist(), dt_s=dt, period_s=period, steps_per_lap=steps_per_lap, laps=laps,
                     loop_angle_rad=[r["loop_angle"] for r in rows], airspeed_m_s=[r["airspeed"] for r in rows],
                     kite_speed_m_s=[r["kite_speed"] for r in rows], alpha_cmd_rad=[r["alpha"] for r in rows],
                     beta_cmd_rad=[r["beta"] for r in rows], tether_roll_cmd_rad=[r["roll"] for r in rows],
                     tether_roll_unsaturated_rad=[r["roll_unsaturated"] for r in rows],
                     angular_acceleration_rad_s2=omega_dot.tolist(), angular_acceleration_stored_with_motion=bool(store_angular_acceleration),
                     samples_with_alpha_fading=int(sum(r["alpha_fading"] for r in rows)),
                     samples_with_roll_saturated=int(sum(r["roll_saturated"] for r in rows)),
                     samples_with_airspeed_saturated=int(sum(r["airspeed_saturated"] for r in rows)),
                     tension_for_unsaturated_roll_N=self.tension_for_unsaturated_roll(),
                     derivatives=f"fourth-order central differences of the law in time, step {h:g} s (angular acceleration by nested stencils)")
        return motion, table

    def tension_for_unsaturated_roll(self, samples=720):
        """Smallest tension at which the saturation band holds the whole unsaturated roll command (None: never)."""
        c = self.constants
        angles = np.linspace(0.0, 2.0*np.pi, samples, endpoint=False)
        needed = max(abs(self.tether_roll_command(a)[1]-c.tether_roll_nom_rad) for a in angles)
        low, high = c.tether_roll_max_excursion_low_rad, c.tether_roll_max_excursion_high_rad
        if needed <= low:
            return 0.0
        if needed > high:
            return None
        return float(c.tether_roll_tension_low_N+(needed-low)/(high-low)*(c.tether_roll_tension_high_N-c.tether_roll_tension_low_N))

    def settings(self):
        """What the law was evaluated with, for the receipts."""
        c = self.constants
        return dict(wind_g_m_s=self.wind_g.tolist(), wind_world_m_s=self.wind_world().tolist(), wind_speed_for_playbook_m_s=self.wind_speed,
                    radius_m=self.radius_m, elevation_rad=self.elevation_rad, azi_offset_rad=self.azi_offset_rad,
                    centre_distance_m=self.centre_distance_m, makani_centre_distance_m=c.tether_length_m,
                    kite_distance_from_ground_point_m=self.kite_on_sphere_m,
                    path_centre_world_m=(MAKANI_G_TO_WORLD@self.path_centre_g).tolist(), interpolation=self.interpolation,
                    interpolation_difference=self.interpolation_difference(), roll_limit_tension_N=self.roll_limit_tension_N,
                    roll_excursion_rad=self.roll_excursion(), tether_roll_nom_rad=c.tether_roll_nom_rad,
                    effective_mass_kg=c.effective_mass_kg, air_density_of_the_law_kg_m3=c.air_density_kg_m3,
                    rows_blended=self.entry.get("rows_blended"), loop_direction="clockwise seen from the ground station",
                    frames="world = diag(1,-1,-1) Makani g; body = diag(-1,1,-1) Makani body; sideslip sign reversed in the body axes")
