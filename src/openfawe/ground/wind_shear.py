"""Plan 3.9b: Makani's no-turbulence wind with the power-law shear (borrowed part, transcribed; opt-in, used by nothing by
default). Power-energy contract revision 1 addendum 5 (E5).

Source: Makani snapshot 1d66923 (Apache License 2.0, Makani Technologies LLC):

- ``sim/models/environment.cc:169-205`` ``Environment::CalcWind``, the ``kWindModelNoTurbulence`` branch: the point is
  taken to the mean-wind frame by ``TransformGToMw`` (``sim/physics/wind_frame.c:31-36``: subtract ``ground_z``, then the
  transpose of ``dcm_mw2g``, whose roll of pi makes the frame's z axis point up, so ``pos_mw.z`` is the height above
  ground ``ground_z - z_g``); the speed is ``CalcWindShear(h, wind_shear_ref_height_agl, wind_speed,
  wind_shear_exponent)`` (``openfawe.ground.makani_wind``); with ``wind_veer`` and ``wind_elevation`` zero
  (``config/common/sim/phys_sim.py``) the direction does not change with height and is the uniform wind's
  (``perch.makani_wind_g``, the same expression per unit speed).
- ``environment.cc:122-158`` ``CalcWind(pos_g, wind_g, wind_omega_g)``: half the curl of the wind by forward differences
  with ``kFiniteDifferenceStep`` = 10 m (``environment.h:119``), the wind-induced apparent rate that ``Wing`` adds to the
  body rates when it looks up the aerodynamic coefficients (``wing.cc:553-558``).
- ``config/base_station/wind_sensor.py``: the GSv2 (GS02) wind sensor at (2.32, 0.026, -14.9) m in the platform frame.

Ours, not Makani's (addendum 5 and the code addendum): the height is ``ground_z - z_g`` exactly; Makani's transpose
of ``dcm_mw2g`` carries ``sin(pi)`` ~ 1.2e-16 terms of x and y into ``pos_mw.z`` (a difference of order 1e-16 |x| m).
That the platform frame's origin and z axis are the ground frame's (the platform turns about the vertical axis through
the ground-frame origin) is our reading, so the sensor is 14.9 m above the ground frame's origin, 21.022 m above ground.
The wind is Makani's ``wind_g`` in magnitude and direction, not operation by operation (Makani rotates
``(v cos 0 cos 0, -v cos 0 sin 0, -v sin 0)`` by ``dcm_mw2g``; ``perch.makani_wind_g`` writes the result directly).

Where the wind is taken (``ground.turnkey`` with ``wind_shear``): the body origin for the aerodynamics and, carried by the
rigid motion, the rotors (Makani's ``CalcLocalApparentWindB`` uses the wing's ``wind_g()``, the wind at the origin); every
tether node at its height (``tether.cc:907``); the estimate's ground wind at the wind sensor, its wind aloft and
apparent wind at the body origin (``kite_estimate``).
"""
import math

import numpy as np

from openfawe.flight.makani_interface import G_TO_WORLD

from .makani_wind import calc_wind_shear

FINITE_DIFFERENCE_STEP_M = 10.0                 # Environment::kFiniteDifferenceStep (environment.h:119)
WIND_SENSOR_POS_P_M = (2.32, 0.026, -14.9)      # config/base_station/wind_sensor.py, GSv2
REF_HEIGHT_AGL_M = 21.0                         # config/common/sim/phys_sim.py, wind_shear_ref_height_agl


class ShearWind:
    """The wind of ``Environment::CalcWind`` without turbulence: ``speed_m_s`` at ``ref_height_agl_m`` above the ground,
    the power-law ``exponent``, coming from ``direction_rad`` (Makani's ``wind_direction``), the ground at
    ``ground_z_g_m`` (z of the ground in the NED ground frame g). World coordinates are the plant's (``G_TO_WORLD``)."""

    def __init__(self, speed_m_s, direction_rad, exponent, *, ref_height_agl_m=REF_HEIGHT_AGL_M, ground_z_g_m):
        self.speed_m_s, self.direction_rad = float(speed_m_s), float(direction_rad)
        self.exponent, self.ref_height_agl_m, self.ground_z_g_m = float(exponent), float(ref_height_agl_m), float(ground_z_g_m)
        if not (math.isfinite(self.exponent) and math.isfinite(self.speed_m_s) and self.ref_height_agl_m > 0.0
                and math.isfinite(self.ref_height_agl_m) and math.isfinite(self.ground_z_g_m)):
            raise ValueError("the wind shear needs a finite exponent, speed and ground height and a positive reference height")
        if self.speed_m_s <= 0.0:
            raise ValueError("the wind shear needs a positive reference speed")
        d = self.direction_rad
        self.unit_g = np.array([-math.cos(d), -math.sin(d), 0.0])       # perch.makani_wind_g per unit speed
        self.unit_world = G_TO_WORLD@self.unit_g

    @classmethod
    def from_params(cls, params, exponent, *, ref_height_agl_m=REF_HEIGHT_AGL_M, ground_z_g_m):
        """Speed and direction from the parameters' ``phys_sim`` (as ``perch.makani_wind_g``)."""
        phys = params.raw["phys_sim"]
        return cls(float(phys["wind_speed"]), float(phys["wind_direction"]), exponent, ref_height_agl_m=ref_height_agl_m,
                   ground_z_g_m=ground_z_g_m)

    def settings(self):
        return dict(model="Makani CalcWind, kWindModelNoTurbulence (power-law shear)", exponent=self.exponent,
                    ref_height_agl_m=self.ref_height_agl_m, speed_at_ref_m_s=self.speed_m_s, direction_rad=self.direction_rad,
                    ground_z_g_m=self.ground_z_g_m, sensor_pos_p_m=list(WIND_SENSOR_POS_P_M),
                    sensor_height_agl_m=self.sensor_height_agl_m(), sensor_speed_m_s=self.speed_at(self.sensor_height_agl_m()),
                    finite_difference_step_m=FINITE_DIFFERENCE_STEP_M)

    def speed_at(self, height_agl_m):
        """``CalcWindShear`` at the height above ground."""
        return calc_wind_shear(float(height_agl_m), self.ref_height_agl_m, self.speed_m_s, self.exponent)

    def wind_g(self, pos_g):
        """The wind in g coordinates at the point ``pos_g`` (g coordinates)."""
        return self.speed_at(self.ground_z_g_m - float(pos_g[2]))*self.unit_g

    def wind_world_at(self, pos_world):
        """The wind in world coordinates at ``pos_world`` (G_TO_WORLD is its own inverse)."""
        return G_TO_WORLD@self.wind_g(G_TO_WORLD@np.asarray(pos_world, dtype=float))

    def node_winds_world(self, nodes_world):
        """One wind per tether node at its own height (world coordinates; the world z axis points up, so the height is
        ``ground_z + z``). Equal bit for bit to ``wind_world_at`` of each node (the speed is positive and the products
        with the direction exact in the same way)."""
        nodes = np.asarray(nodes_world, dtype=float).reshape(-1, 3)
        speeds = np.array([self.speed_at(self.ground_z_g_m + z) for z in nodes[:, 2].tolist()], dtype=float)
        return speeds[:, None]*self.unit_world[None, :]

    def winds_world(self, points_world):
        """OF-0010 revision 19 addendum 6 (B1): the wind at many points at once, for the vortex method's wake, rings and
        sections (world coordinates, z up, height ``ground_z + z``); ``numpy.power`` in place of ``math.pow``, so equal
        to ``node_winds_world`` to rounding (a unit test bounds it at 1e-12 relative), not bit for bit."""
        points = np.asarray(points_world, dtype=float).reshape(-1, 3)
        ratio = np.maximum((self.ground_z_g_m + points[:, 2])/self.ref_height_agl_m, 0.1)
        speeds = self.speed_m_s*np.power(ratio, self.exponent)
        return speeds[:, None]*self.unit_world[None, :]

    def wind_omega_g(self, pos_g):
        """Half the curl of the wind by forward differences of 10 m, operation by operation as environment.cc:122-158."""
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
        """The wind-induced apparent rates in the plant's body axes at the body origin of ``pose`` (world-frame pose with
        the body-to-world ``rotation``): ``dcm_g2b wind_omega_g`` of wing.cc:555 in OpenFAWE body axes; the database
        (``aero.makani_database``) takes the angular velocity in these axes and turns it to Makani's."""
        omega_g = self.wind_omega_g(G_TO_WORLD@np.asarray(pose.position_m, dtype=float))
        return pose.rotation.T@(G_TO_WORLD@omega_g)

    def origin_wind_g(self, pose):
        """The wind at the body origin of ``pose`` (g coordinates)."""
        return self.wind_g(G_TO_WORLD@np.asarray(pose.position_m, dtype=float))

    def sensor_height_agl_m(self):
        return self.ground_z_g_m - WIND_SENSOR_POS_P_M[2]

    def sensor_wind_g(self):
        """The wind at the ground wind sensor (g coordinates): the estimate's ground wind."""
        return self.speed_at(self.sensor_height_agl_m())*self.unit_g
