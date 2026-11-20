"""IEC 61400-1 deterministic extreme winds as Makani's simulator computes them (OF-0011 revision 7 addendum 5,
conference paper 2's P1; borrowed part, transcribed; opt-in, used by nothing by default).

Source: Makani snapshot 1d66923 (Apache License 2.0, Makani Technologies LLC):

- ``sim/physics/wind.cc:183-420``: ``IecWindModel`` - the reference wind speed and turbulence intensity of the class
  and category (IEC 61400-1 Table 1), the longitudinal turbulence scale (eq. 5), the normal turbulence sigma (eq. 11),
  the normal wind profile (eq. 10, alpha 0.2), the extreme wind speeds (eqs. 12-15), the extreme operating gust EOG
  (eqs. 17-18), the extreme direction change EDC (eqs. 20-21), the extreme coherent gust with direction change ECD (eqs.
  22-25) and the extreme wind shears (eqs. 26-27), each with the equation and page numbers that Makani's source gives;
- ``sim/models/environment.cc:51-95`` (the model's construction: turbulence category A, class I, the hub wind speed
  from ``phys_sim`` through ``CalcWindShear``), ``160-220`` (``CalcWind``: the mean-wind frame and back),
  ``337-414`` (``CalcIecMeanWind``) and ``416-446`` (the surrogate turbulence: zero for the normal wind profile, EOG,
  EDC, ECD and the shears; NOT zero for the extreme wind speeds, which Makani gives sigma_1 = 0.11 V_hub of white
  noise - this module computes their mean only, see addendum 5 result 1);
- ``sim/physics/wind_frame.c`` (``CalcDcmMwToG``, ``TransformGToMw``) and ``sim/models/model.cc:134-143`` (the
  environment's 0.1 s discrete time ``t_z1`` at which the wind is evaluated);
- ``config/m600/sim/iec_sim.py`` (the M600's analogue of a wind turbine: hub height sin 30 deg x 370 m + ground_z,
  rotor diameter 220 m, event start 10 s).

Reproduced as Makani has it and reported (addendum 5, I-2): ``IecWindModel``'s constructor passes
``(v_ref, I_ref, hub_wind_speed)`` to ``CalcExtremeTurbulenceModelSigma(hub_wind_speed, v_ref, I_ref)``; only Makani's ETM
surrogate turbulence uses it (``etm_sigma_1``), none of the deterministic cases.
"""
import math
import sys

import numpy as np

from .gs02 import angle_to_dcm
from .makani_wind import calc_wind_shear

EPS = sys.float_info.epsilon
DBL_MAX = sys.float_info.max
PI = math.pi
ENVIRONMENT_TS_S = 0.1                                 # Model("Environment", 0.1)
CASES = ("normal_wind_profile", "extreme_wind_speed_1year", "extreme_wind_speed_50year", "extreme_operating_gust",
         "extreme_direction_change", "extreme_coherent_gust_with_direction_change", "extreme_wind_shear_vertical",
         "extreme_wind_shear_horizontal")
# sim_types.h's IecDesignLoadCase numbers of these cases (kIecCaseNormalWindProfile = 0, ...)
CASE_NUMBERS = dict(normal_wind_profile=0, extreme_wind_speed_1year=2, extreme_wind_speed_50year=3,
                    extreme_operating_gust=4, extreme_direction_change=6,
                    extreme_coherent_gust_with_direction_change=7, extreme_wind_shear_vertical=8,
                    extreme_wind_shear_horizontal=9)
REFERENCE_SPEED = dict(I=50.0, II=42.5, III=37.5)       # wind.cc:183-196
TURBULENCE_INTENSITY = dict(A=0.16, B=0.14, C=0.12)     # wind.cc:198-213


def longitudinal_scale(hub_height_agl):
    """wind.cc:215-223 (IEC 61400-1 eq. 5)."""
    return 0.7*hub_height_agl if hub_height_agl <= 60.0 else 42.0


def ntm_sigma(i_ref, hub_wind_speed):
    """wind.cc:225-229 (eq. 11)."""
    b = 5.6
    return i_ref*(0.75*hub_wind_speed + b)


def ewm_sigma(hub_wind_speed):
    """wind.cc:231-234 (eq. 16)."""
    return 0.11*hub_wind_speed


def etm_sigma(hub_wind_speed, v_ref, i_ref):
    """wind.cc:236-245 (eqs. 9, 19)."""
    v_ave = 0.2*v_ref
    c = 2.0
    return c*i_ref*(0.072*(v_ave/c + 3.0)*(hub_wind_speed/c - 4.0) + 10.0)


class IecWindModel:
    """wind.cc:249-420, function by function."""

    def __init__(self, hub_height_agl, rotor_diameter, hub_wind_speed, turbulence_category="A", wind_turbine_class="I"):
        self.hub_height_agl = float(hub_height_agl)
        self.rotor_diameter = float(rotor_diameter)
        self.hub_wind_speed = float(hub_wind_speed)
        self.v_ref = REFERENCE_SPEED[wind_turbine_class]
        self.i_ref = TURBULENCE_INTENSITY[turbulence_category]
        self.lambda_1 = longitudinal_scale(self.hub_height_agl)
        self.ntm_sigma_1 = ntm_sigma(self.i_ref, self.hub_wind_speed)
        self.ewm_sigma_1 = ewm_sigma(self.hub_wind_speed)
        self.etm_sigma_1 = etm_sigma(self.v_ref, self.i_ref, self.hub_wind_speed)   # Makani's argument order (I-2)
        if not (0.0 < self.hub_height_agl and self.hub_height_agl > self.rotor_diameter/2.0 and self.hub_wind_speed >= 0.0):
            raise ValueError("IecWindModel: hub height, rotor diameter or hub wind speed out of range (wind.cc:264-266)")

    def normal_wind_profile(self, height_agl):
        return calc_wind_shear(height_agl, self.hub_height_agl, self.hub_wind_speed, 0.2)

    def extreme_wind_speed_50year(self, height_agl, is_turbulent):
        wind_speed = calc_wind_shear(height_agl, self.hub_height_agl, self.v_ref, 0.11)
        return wind_speed if is_turbulent else 1.4*wind_speed

    def extreme_wind_speed_1year(self, height_agl, is_turbulent):
        return 0.8*self.extreme_wind_speed_50year(height_agl, is_turbulent)

    def extreme_operating_gust(self, t, height_agl):
        e1 = self.extreme_wind_speed_1year(height_agl, False)
        gust = min(1.35*(e1 - self.hub_wind_speed), 3.3*(self.ntm_sigma_1/(1.0 + 0.1*self.rotor_diameter/self.lambda_1)))
        T = 10.5
        wind_speed = self.normal_wind_profile(height_agl)
        if 0.0 <= t <= T:
            wind_speed -= 0.37*gust*math.sin(3.0*PI*t/T)*(1.0 - math.cos(2.0*PI*t/T))
        return wind_speed

    def extreme_direction_change(self, t):
        sign = 1.0
        theta_e = sign*4.0*math.atan2(self.ntm_sigma_1, self.hub_wind_speed*(1.0 + 0.1*self.rotor_diameter/self.lambda_1))
        T = 6.0
        if t < 0.0:
            return 0.0
        if t <= T:
            return 0.5*theta_e*(1.0 - math.cos(PI*t/T))
        return theta_e

    def extreme_coherent_gust_with_direction_change(self, t, height_agl):
        """(speed, direction change)."""
        sign = 1.0
        coherent_gust_speed = 15.0
        direction_change = PI if self.hub_wind_speed < 4.0 else 4.0*PI/self.hub_wind_speed
        T = 10.0
        wind_speed = self.normal_wind_profile(height_agl)
        if t < 0.0:
            direction = 0.0
        elif t <= T:
            direction = sign*0.5*direction_change*(1.0 - math.cos(PI*t/T))
            wind_speed += 0.5*coherent_gust_speed*(1.0 - math.cos(PI*t/T))
        else:
            direction = sign*direction_change
            wind_speed += coherent_gust_speed
        return wind_speed, direction

    def _shear_amplitude(self):
        beta = 6.4
        return 2.5 + 0.2*beta*self.ntm_sigma_1*math.pow(self.rotor_diameter/self.lambda_1, 0.25)

    def extreme_wind_shear_vertical(self, t, height_agl):
        sign, T = 1.0, 12.0
        wind_speed = self.normal_wind_profile(height_agl)
        if 0.0 <= t <= T:
            wind_speed += (sign*((height_agl - self.hub_height_agl)/self.rotor_diameter)*self._shear_amplitude()
                           *(1.0 - math.cos(2.0*PI*t/T)))
        return wind_speed

    def extreme_wind_shear_horizontal(self, t, height_agl, y):
        sign, T = 1.0, 12.0
        wind_speed = self.normal_wind_profile(height_agl)
        if 0.0 <= t <= T:
            wind_speed += sign*(y/self.rotor_diameter)*self._shear_amplitude()*(1.0 - math.cos(2.0*PI*t/T))
        return wind_speed


def dcm_mw_to_g(mean_wind_direction):
    """wind_frame.c ``CalcDcmMwToG``: AngleToDcm(pi, 0, -(pi + direction), XYZ)."""
    return angle_to_dcm(PI, 0.0, -(PI + mean_wind_direction), "xyz")


def g_to_mw(point_g, dcm_mw2g, ground_z):
    """wind_frame.c ``TransformGToMw``: z less ground_z, then the transpose of dcm_mw2g."""
    p = np.array(point_g, dtype=float)
    p[2] -= ground_z
    return dcm_mw2g.T@p


class IecMeanWind:
    """``Environment::CalcIecMeanWind`` and the mw-to-g step of ``CalcWind`` for one load case (environment.cc:160-220,
    337-414), with the environment's discrete time: ``discrete_update(t)`` as model.cc:134-143 (0.1 s), ``wind_g``
    evaluated at ``t_z1``."""

    def __init__(self, load_case, *, hub_height_agl, rotor_diameter, event_t_start, wind_speed, wind_direction,
                 wind_shear_ref_height_agl, wind_shear_exponent, ground_z):
        if load_case not in CASES:
            raise ValueError(f"load case {load_case!r} is not one of {CASES}")
        want = 0.11 if load_case == "extreme_wind_speed_50year" else 0.2
        if abs(want - wind_shear_exponent) > 1e-3:                       # environment.cc:82-88
            raise ValueError(f"the IEC case {load_case} needs the wind shear exponent {want}")
        self.case = load_case
        self.t_start = float(event_t_start)
        self.ground_z = float(ground_z)
        self.model = IecWindModel(hub_height_agl, rotor_diameter,
                                  calc_wind_shear(hub_height_agl, wind_shear_ref_height_agl, wind_speed, wind_shear_exponent))
        self.dcm_mw2g = dcm_mw_to_g(wind_direction)
        self.t_z1 = -DBL_MAX

    def discrete_update(self, t):
        t_cmp = t*(1.0 + EPS)
        if t_cmp - self.t_z1 >= ENVIRONMENT_TS_S:
            self.t_z1 = t_cmp - math.fmod(t_cmp, ENVIRONMENT_TS_S)

    def speed_direction(self, t, pos_mw):
        m, z = self.model, pos_mw[2]
        direction = 0.0
        if self.case == "normal_wind_profile":
            speed = m.normal_wind_profile(z)
        elif self.case == "extreme_wind_speed_1year":
            speed = m.extreme_wind_speed_1year(z, False)
        elif self.case == "extreme_wind_speed_50year":
            speed = m.extreme_wind_speed_50year(z, False)
        elif self.case == "extreme_operating_gust":
            speed = m.extreme_operating_gust(t - self.t_start, z)
        elif self.case == "extreme_direction_change":
            speed = m.normal_wind_profile(z)
            direction = m.extreme_direction_change(t - self.t_start)
        elif self.case == "extreme_coherent_gust_with_direction_change":
            speed, direction = m.extreme_coherent_gust_with_direction_change(t - self.t_start, z)
        elif self.case == "extreme_wind_shear_vertical":
            speed = m.extreme_wind_shear_vertical(t - self.t_start, z)
        else:
            speed = m.extreme_wind_shear_horizontal(t - self.t_start, z, pos_mw[1])
        return speed, direction

    def wind_g(self, pos_g):
        pos_mw = g_to_mw(pos_g, self.dcm_mw2g, self.ground_z)
        speed, direction = self.speed_direction(self.t_z1, pos_mw)
        wind_mw = np.array([speed*math.cos(direction), -speed*math.sin(direction), 0.0])
        return self.dcm_mw2g@wind_mw


def from_params(params_raw, load_case, iec_sim=None, ground_z=None):
    """An ``IecMeanWind`` from a GS02 parameter file's raw dict (``phys_sim``, ``ground_frame``) and Makani's
    ``iec_sim`` values (default: ``config/m600/sim/iec_sim.py`` with that ground_z)."""
    phys = params_raw["phys_sim"]
    gz = float(params_raw["ground_frame"]["ground_z"]) if ground_z is None else float(ground_z)
    iec = iec_sim or dict(hub_height_agl=math.sin(math.radians(30.0))*370.0 + gz, rotor_diameter=220.0,
                          event_t_start=10.0)
    return IecMeanWind(load_case, hub_height_agl=iec["hub_height_agl"], rotor_diameter=iec["rotor_diameter"],
                       event_t_start=iec["event_t_start"], wind_speed=float(phys["wind_speed"]),
                       wind_direction=float(phys["wind_direction"]),
                       wind_shear_ref_height_agl=float(phys.get("wind_shear_ref_height_agl", 21.0)),
                       wind_shear_exponent=0.11 if load_case == "extreme_wind_speed_50year" else 0.2, ground_z=gz)


def _shear(z, ref_height, ref_speed, exponent):
    """``CalcWindShear`` on an array (numpy.power for math.pow: equal to rounding)."""
    return ref_speed*np.power(np.maximum(z/ref_height, 0.1), exponent)


class IecEventWind:
    """OF-0011 revision 7 addendum 6 (P2): the IEC winds of ``IecMeanWind`` as a wind field of ``run_turnkey`` (the
    interface of ``wind_shear.ShearWind`` and ``turbulent_wind.TurbulentWind``). ``set_time(t)`` - called by the flight
    at each period's start - advances the environment's 0.1 s discrete time (model.cc:134-143) and every evaluation of
    that period uses the resulting ``t_z1``; the event starts at ``event_t_start`` (Makani's ``iec_sim.event_t_start``).
    The vorticity is Makani's 10 m forward difference (environment.cc:122-158). The ground wind sensor is at its nominal
    place (``wind_shear.WIND_SENSOR_POS_P_M``), as for the other wind fields. The extreme wind speeds are their mean only
    (Makani adds its surrogate turbulence; P1 result 1)."""

    def __init__(self, load_case, params_raw, *, ground_z_g_m, event_t_start=10.0, rotor_diameter=220.0):
        phys = params_raw["phys_sim"]
        self.load_case = load_case
        self.ground_z_g_m = float(ground_z_g_m)
        self.speed_m_s = float(phys["wind_speed"])
        self.direction_rad = float(phys["wind_direction"])
        self.ref_height_agl_m = float(phys.get("wind_shear_ref_height_agl", 21.0))
        self.hub_height_agl_m = math.sin(math.radians(30.0))*370.0 + self.ground_z_g_m      # config/m600/sim/iec_sim.py
        self.mean = IecMeanWind(load_case, hub_height_agl=self.hub_height_agl_m, rotor_diameter=rotor_diameter,
                                event_t_start=event_t_start, wind_speed=self.speed_m_s, wind_direction=self.direction_rad,
                                wind_shear_ref_height_agl=self.ref_height_agl_m,
                                wind_shear_exponent=0.11 if load_case == "extreme_wind_speed_50year" else 0.2,
                                ground_z=self.ground_z_g_m)
        self.time_s = None

    def set_time(self, time_s):
        self.time_s = float(time_s)
        self.mean.discrete_update(self.time_s)

    def _ready(self):
        if self.time_s is None:
            raise RuntimeError("IecEventWind: set_time before the first evaluation")

    def settings(self):
        m = self.mean.model
        return dict(model="Makani CalcWind, kWindModelIec (deterministic mean; OF-0011 rev 7 add 5-6)",
                    load_case=self.load_case, case_number=CASE_NUMBERS[self.load_case], event_t_start_s=self.mean.t_start,
                    hub_height_agl_m=m.hub_height_agl, rotor_diameter_m=m.rotor_diameter, hub_wind_speed_m_s=m.hub_wind_speed,
                    speed_at_ref_m_s=self.speed_m_s, ref_height_agl_m=self.ref_height_agl_m,
                    direction_rad=self.direction_rad, ground_z_g_m=self.ground_z_g_m, environment_ts_s=ENVIRONMENT_TS_S,
                    turbulence_category="A", wind_turbine_class="I", sensor_height_agl_m=self.sensor_height_agl_m())

    def wind_g(self, pos_g):
        self._ready()
        return self.mean.wind_g(pos_g)

    def wind_world_at(self, pos_world):
        from openfawe.flight.makani_interface import G_TO_WORLD
        return G_TO_WORLD@self.wind_g(G_TO_WORLD@np.asarray(pos_world, dtype=float))

    def node_winds_world(self, nodes_world):
        nodes = np.asarray(nodes_world, dtype=float).reshape(-1, 3)
        return np.array([self.wind_world_at(p) for p in nodes], dtype=float).reshape(-1, 3)

    def winds_g(self, points_g):
        """Many points at once (vectorised ``IecMeanWind.wind_g``; equal to rounding)."""
        self._ready()
        p = np.asarray(points_g, dtype=float).reshape(-1, 3).copy()
        p[:, 2] -= self.ground_z_g_m
        mw = p@self.mean.dcm_mw2g                                      # rows: (dcm^T p)^T
        speed, direction = self._speeds(self.mean.t_z1 - self.mean.t_start, mw[:, 2], mw[:, 1])
        wind_mw = np.stack([speed*math.cos(direction), -speed*math.sin(direction), np.zeros_like(speed)], axis=1)
        return wind_mw@self.mean.dcm_mw2g.T

    def winds_world(self, points_world):
        from openfawe.flight.makani_interface import G_TO_WORLD
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        return self.winds_g(pts@G_TO_WORLD.T)@G_TO_WORLD.T

    def _speeds(self, t, z, y):
        m, case = self.mean.model, self.load_case
        nwp = _shear(z, m.hub_height_agl, m.hub_wind_speed, 0.2)
        direction = 0.0
        if case == "normal_wind_profile":
            return nwp, direction
        if case in ("extreme_wind_speed_1year", "extreme_wind_speed_50year"):
            e50 = 1.4*_shear(z, m.hub_height_agl, m.v_ref, 0.11)
            return (0.8*e50 if case == "extreme_wind_speed_1year" else e50), direction
        if case == "extreme_operating_gust":
            e1 = 0.8*(1.4*_shear(z, m.hub_height_agl, m.v_ref, 0.11))
            gust = np.minimum(1.35*(e1 - m.hub_wind_speed), 3.3*(m.ntm_sigma_1/(1.0 + 0.1*m.rotor_diameter/m.lambda_1)))
            T = 10.5
            if 0.0 <= t <= T:
                return nwp - 0.37*gust*math.sin(3.0*PI*t/T)*(1.0 - math.cos(2.0*PI*t/T)), direction
            return nwp, direction
        if case == "extreme_direction_change":
            return nwp, m.extreme_direction_change(t)
        if case == "extreme_coherent_gust_with_direction_change":         # as the scalar method, operation by operation
            change = PI if m.hub_wind_speed < 4.0 else 4.0*PI/m.hub_wind_speed
            T = 10.0
            if t < 0.0:
                return nwp, 0.0
            if t <= T:
                return nwp + 0.5*15.0*(1.0 - math.cos(PI*t/T)), 1.0*0.5*change*(1.0 - math.cos(PI*t/T))
            return nwp + 15.0, 1.0*change
        T = 12.0
        if not 0.0 <= t <= T:
            return nwp, direction
        a = m._shear_amplitude()
        if case == "extreme_wind_shear_vertical":
            return nwp + (1.0*((z - m.hub_height_agl)/m.rotor_diameter)*a*(1.0 - math.cos(2.0*PI*t/T))), direction
        return nwp + 1.0*(y/m.rotor_diameter)*a*(1.0 - math.cos(2.0*PI*t/T)), direction

    def wind_omega_g(self, pos_g):
        """Half the curl by forward differences of 10 m, operation by operation as environment.cc:122-158."""
        from .wind_shear import FINITE_DIFFERENCE_STEP_M
        x, y, z = (float(v) for v in pos_g)
        h = FINITE_DIFFERENCE_STEP_M
        w = self.wind_g((x, y, z))
        t = self.wind_g((x + h, y, z))
        dvdx, dwdx = (t[1] - w[1])/h, (t[2] - w[2])/h
        t = self.wind_g((x, y + h, z))
        dudy, dwdy = (t[0] - w[0])/h, (t[2] - w[2])/h
        t = self.wind_g((x, y, z + h))
        dudz, dvdz = (t[0] - w[0])/h, (t[1] - w[1])/h
        return np.array([0.5*(dwdy - dvdz), 0.5*(dudz - dwdx), 0.5*(dvdx - dudy)])

    def wind_rates_body(self, pose):
        from openfawe.flight.makani_interface import G_TO_WORLD
        omega_g = self.wind_omega_g(G_TO_WORLD@np.asarray(pose.position_m, dtype=float))
        return pose.rotation.T@(G_TO_WORLD@omega_g)

    def origin_wind_g(self, pose):
        from openfawe.flight.makani_interface import G_TO_WORLD
        return self.wind_g(G_TO_WORLD@np.asarray(pose.position_m, dtype=float))

    def sensor_height_agl_m(self):
        from .wind_shear import WIND_SENSOR_POS_P_M
        return self.ground_z_g_m - WIND_SENSOR_POS_P_M[2]

    def sensor_wind_g(self):
        from .wind_shear import WIND_SENSOR_POS_P_M
        return self.wind_g(np.array(WIND_SENSOR_POS_P_M, dtype=float))
