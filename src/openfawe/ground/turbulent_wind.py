"""OF-0010 revision 23 (B2): IEC turbulence from a TurbSim full-field file as the flight's wind (opt-in; nothing uses it
by default; the uniform wind and ``ground.wind_shear`` are untouched).

The field is a TurbSim ``.bts`` file (OpenFAST v5.0.0 ``modules/turbsim``, Apache-2.0), generated with
``docs/tasks/OF-0010-platform20261022/b2/turbsim-b2.inp`` (revision 23 section 23.2). It is read and interpolated as
InflowWind reads and interpolates it, so that the two agree at every point (checks B2-R):

- the file format and the de-normalisation of the 16-bit integers in single precision: InflowWind
  ``InflowWind_IO.f90`` ``IfW_TurbSim_Init`` (header type at 579-590; ``(real(VelRaw, SiKi) - Voffset)/VSlope`` at
  about 744; the grid's half width ``0.5*(NYGrids-1)/InvDY`` and base at 638-648; periodic when the file id is 8, with
  ``InitXPosition`` 0, at 635 and 651-657); the same in openfast_io ``turbsim_file.py`` ``TurbSimFile.read``;
- the lookup: ``IfW_FlowField.f90`` ``Grid3DField_GetCell`` - the time shift ``Time + (InitXPosition - PosX)/MeanWS``
  (frozen turbulence, ``GetBoundsT`` 1255-1256), the periodic wrap by two moduli (1262-1268), the y and z cells with the
  edge tolerance ``GridTol`` = 1e-3 of a cell (``GetBoundsY`` 1093-1144, ``GetBoundsZ`` 1146-1238; outside the grid is
  an error, there is no tower and no extrapolation here), the eight corners in the order of ``GetCellInGrid`` (835-850)
  and the trilinear shape functions of ``Grid3DField_GetVelLinear`` (1308-1336);
- the flight's use: Makani snapshot 1d66923 (Apache-2.0) ``sim/physics/wind.cc:160-176``
  ``FrozenTurbulenceWindDatabase::CalcWind`` (the initial time offset ``t0``; the mean-wind frame of
  ``sim/physics/wind_frame.c``: x downwind, z up from the ground at ``ground_z``, the origin under the ground frame's),
  ``sim/models/environment.cc:122-158`` (the wind-induced rates: half the curl by 10 m forward differences, the same for
  every wind model) and ``sim/models/full_system.cc:657-661`` (the wind sensor samples the field at its own position).
  Unlike Makani the lookup does not wrap in y (InflowWind's behaviour: outside the grid raises).

Within one flight period every evaluation (the body origin, the tether nodes, the wake points, the rotors, the sensor)
sees the field at the period's start time, set by ``set_time`` (revision 23 section 23.3: a 0.01 s hold against the
field's 0.1 s step - an approximation, written). The interface is ``ground.wind_shear.ShearWind``'s plus ``set_time``.
"""
import hashlib
import math
import struct
from pathlib import Path

import numpy as np

from openfawe.flight.makani_interface import G_TO_WORLD

from .wind_shear import FINITE_DIFFERENCE_STEP_M, WIND_SENSOR_POS_P_M

GRID_TOL = 1.0e-3                       # IfW_FlowField.f90:36 GridTol (a fraction of a cell)
PERIODIC_ID, APERIODIC_ID = 8, 7        # TurbSim file ids (InflowWind_IO.f90:635)
HEADER = struct.Struct("<h4i3f3f6fi")   # id; nz ny ntower nt; dz dy dt; mws ref_height grid_base; (slope, offset) x 3; desc len


class TurbSimField:
    """A TurbSim ``.bts`` file held as its 16-bit integers (``raw[it, iz, iy, ic]``) with the single-precision slopes
    and offsets, decoded on lookup exactly as InflowWind decodes it (so grid values are bit for bit InflowWind's)."""

    def __init__(self, path):
        self.path = Path(path)
        data = self.path.read_bytes()
        self.sha256 = hashlib.sha256(data).hexdigest()
        (file_id, nz, ny, ntower, nt, dz, dy, dt, mws, ref_height, grid_base,
         sx, ox, sy, oy, sz, oz, desc_len) = HEADER.unpack_from(data, 0)
        if file_id not in (PERIODIC_ID, APERIODIC_ID):
            raise ValueError(f"{self.path}: not a TurbSim full-field file (id {file_id})")
        if ntower:
            raise ValueError(f"{self.path}: tower points ({ntower}) are not supported")
        start = HEADER.size+desc_len
        self.description = data[HEADER.size:start].decode("latin-1")
        count = nt*3*ny*nz
        raw = np.frombuffer(data, dtype="<i2", count=count, offset=start)
        if raw.size != count or len(data) != start+2*count:
            raise ValueError(f"{self.path}: {len(data)} bytes, expected {start+2*count}")
        self.raw = raw.reshape(nt, nz, ny, 3)       # Fortran VelRaw(3, NY, NZ) per step, i.e. component fastest
        self.slope = np.array([sx, sy, sz], dtype=np.float32)
        self.offset = np.array([ox, oy, oz], dtype=np.float32)
        self.periodic = file_id == PERIODIC_ID
        self.nz, self.ny, self.nt = int(nz), int(ny), int(nt)
        self.dt_s, self.dy_m, self.dz_m = float(dt), float(dy), float(dz)       # the file's float32 values as doubles
        self.rate = 1.0/self.dt_s                                               # G3D%Rate
        self.inv_dy, self.inv_dz = 1.0/self.dy_m, 1.0/self.dz_m
        self.y_half_width_m = 0.5*(self.ny-1)/self.inv_dy                       # G3D%YHWid
        self.z_half_height_m = 0.5*(self.nz-1)/self.inv_dz
        self.mean_speed_m_s = float(mws)                                        # G3D%MeanWS (the hub mean)
        self.inv_mean_speed = 1.0/self.mean_speed_m_s
        self.hub_height_m = float(ref_height)
        self.grid_base_m = float(grid_base)                                     # G3D%GridBase
        self.x_init_m = 0.0 if self.periodic else self.y_half_width_m            # G3D%InitXPosition
        self.total_time_s = (self.nt if self.periodic else self.nt-1)*self.dt_s

    def __getstate__(self):
        """Pickled by reference (checkpoints hold the hook and its wind field, OF-0010 revision 12): the path and the
        hash; unpickling reads the file again and refuses a different one."""
        return dict(path=str(self.path), sha256=self.sha256)

    def __setstate__(self, state):
        fresh = TurbSimField(state["path"])
        if fresh.sha256 != state["sha256"]:
            raise ValueError(f"{state['path']}: the file's hash {fresh.sha256} is not the pickled field's {state['sha256']}")
        self.__dict__.update(fresh.__dict__)

    @property
    def y_m(self):
        return -self.y_half_width_m+np.arange(self.ny)*self.dy_m

    @property
    def z_m(self):
        return self.grid_base_m+np.arange(self.nz)*self.dz_m

    @property
    def t_s(self):
        return np.arange(self.nt)*self.dt_s

    def decoded(self, it, iz, iy):
        """The velocities (m/s, float32 arithmetic as InflowWind's SiKi) at the 0-based grid indices (arrays)."""
        return ((self.raw[it, iz, iy].astype(np.float32)-self.offset)/self.slope).astype(np.float64)

    def grid_velocity(self):
        """The whole field decoded, ``(nt, nz, ny, 3)`` float64 (about 600 MB for the B2 field; for the checks)."""
        return ((self.raw.astype(np.float32)-self.offset)/self.slope).astype(np.float64)

    def velocity(self, time_s, x_m, y_m, z_m):
        """u, v, w (m/s) at the points of the field's frame (x downwind, y to the left looking downwind, z up from the
        grid's ground) at ``time_s`` (arrays of one shape or scalars), as ``Grid3DField_GetCell`` plus
        ``Grid3DField_GetVelLinear``. Raises ValueError where InflowWind would stop (outside the grid or the time)."""
        x = np.asarray(x_m, dtype=float); y = np.asarray(y_m, dtype=float); z = np.asarray(z_m, dtype=float)
        t = np.broadcast_to(np.asarray(time_s, dtype=float), x.shape)
        x, y, z, t = np.broadcast_arrays(x, y, z, t)
        shape = x.shape
        x, y, z, t = (a.reshape(-1) for a in (x, y, z, t))
        it_lo, it_hi, xi_t = self._bounds_t(x, t)
        iy_lo, iy_hi, xi_y = self._bounds(y+self.y_half_width_m, self.inv_dy, self.ny, "y", y)
        iz_lo, iz_hi, xi_z = self._bounds(z-self.grid_base_m, self.inv_dz, self.nz, "z", z)
        # the eight corners in GetCellInGrid's order (1-based indices in InflowWind, 0-based here)
        corners = [self.decoded(it_lo, iz_lo, iy_lo), self.decoded(it_lo, iz_lo, iy_hi),
                   self.decoded(it_lo, iz_hi, iy_lo), self.decoded(it_lo, iz_hi, iy_hi),
                   self.decoded(it_hi, iz_lo, iy_lo), self.decoded(it_hi, iz_lo, iy_hi),
                   self.decoded(it_hi, iz_hi, iy_lo), self.decoded(it_hi, iz_hi, iy_hi)]
        n = [(1.0-xi_y)*(1.0-xi_z), (1.0+xi_y)*(1.0-xi_z), (1.0-xi_y)*(1.0+xi_z), (1.0+xi_y)*(1.0+xi_z)]
        lo, hi = (1.0-xi_t)/8.0, (1.0+xi_t)/8.0
        weights = [w*lo for w in n]+[w*hi for w in n]
        out = corners[0]*weights[0][:, None]
        for c, w in zip(corners[1:], weights[1:]):
            out = out+c*w[:, None]
        return out.reshape(shape+(3,))

    def _bounds_t(self, x, t):
        shifted = t+(self.x_init_m-x)*self.inv_mean_speed          # GetBoundsT 1256
        grid = shifted*self.rate
        if self.periodic:
            grid = np.mod(grid, float(self.nt))
            grid = np.mod(grid, float(self.nt))
        lo = np.floor(grid).astype(np.int64)+1                     # 1-based IT_LO
        hi = lo+1
        xi = 2.0*(grid-np.trunc(grid))-1.0
        inside = (lo >= 1) & (hi <= self.nt)
        last = lo == self.nt
        if self.periodic:
            hi = np.where(last, 1, hi)
        else:
            near = last & (xi <= GRID_TOL)
            hi = np.where(near, lo, hi); xi = np.where(near, -1.0, xi)
            extrap = last & ~near
            lo = np.where(extrap, self.nt-1, lo); hi = np.where(extrap, self.nt, hi)
        bad = ~(inside | last)
        if np.any(bad):
            i = int(np.flatnonzero(bad)[0])
            raise ValueError(f"the wind field's time is exhausted at t = {t[i]} s (shifted {shifted[i]} s; the file holds "
                             f"{self.total_time_s} s{' periodic' if self.periodic else ''})")
        return lo-1, hi-1, xi

    def _bounds(self, offset, inv_step, count, axis, pos):
        grid = offset*inv_step+1.0                                 # GetBoundsY 1098 / GetBoundsZ 1152 (1-based)
        lo = np.floor(grid).astype(np.int64)
        hi = lo+1
        frac = grid-np.trunc(grid)
        inside = (lo >= 1) & (hi <= count)
        below = (lo == 0) & (frac >= 1.0-GRID_TOL)
        above = (lo == count) & (frac <= GRID_TOL)
        bad = ~(inside | below | above)
        if np.any(bad):
            i = int(np.flatnonzero(bad)[0])
            low = -self.y_half_width_m if axis == "y" else self.grid_base_m
            high = self.y_half_width_m if axis == "y" else self.grid_base_m+2*self.z_half_height_m
            raise ValueError(f"the wind field's grid is too small in {axis}: {axis} = {pos[i]} m, grid [{low}, {high}] m")
        xi = np.where(inside, 2.0*frac-1.0, np.where(below, -1.0, 1.0))
        lo = np.where(below, 1, np.where(above, count-1, lo))
        hi = np.where(below, 2, np.where(above, count, hi))
        return lo-1, hi-1, xi


class TurbulentWind:
    """The flight's wind from a ``TurbSimField``: ``ShearWind``'s interface plus ``set_time``. ``direction_rad`` is
    Makani's ``wind_direction`` (the mean wind comes from that azimuth; ``perch.makani_wind_g``), ``ground_z_g_m`` the
    ground's z in the NED ground frame g; ``t0_s`` Makani's ``wind_database_initial_time``. The field's hub mean must be
    the parameters' wind speed (``ground.turnkey`` checks ``speed_m_s``)."""

    def __init__(self, field, direction_rad, *, ground_z_g_m, t0_s=30.0, z_below="raise"):
        if z_below not in ("raise", "saturate"):
            raise ValueError("z_below is 'raise' (InflowWind) or 'saturate' (Makani's wind.cc: the bottom row below the grid)")
        self.z_below = z_below                           # OF-0010 revision 23 addendum 9; the default changes nothing
        if not isinstance(field, TurbSimField):
            field = TurbSimField(field)
        self.field = field
        self.direction_rad, self.ground_z_g_m, self.t0_s = float(direction_rad), float(ground_z_g_m), float(t0_s)
        if not (math.isfinite(self.direction_rad) and math.isfinite(self.ground_z_g_m) and math.isfinite(self.t0_s)):
            raise ValueError("the turbulent wind needs a finite direction, ground height and initial time")
        self.speed_m_s = field.mean_speed_m_s
        d = self.direction_rad
        # the mean-wind frame in g: x downwind (perch.makani_wind_g per unit speed), z up, y = z x x (right-handed;
        # InflowWind's y, to the left looking downwind)
        self.x_g = np.array([-math.cos(d), -math.sin(d), 0.0])
        self.z_g = np.array([0.0, 0.0, -1.0])
        self.y_g = np.cross(self.z_g, self.x_g)
        self.unit_g = self.x_g                                     # as ShearWind.unit_g (the mean wind's direction)
        self.unit_world = G_TO_WORLD@self.unit_g
        self.time_s = 0.0

    @classmethod
    def from_params(cls, params, field, *, ground_z_g_m, t0_s=30.0, z_below="raise"):
        phys = params.raw["phys_sim"]
        wind = cls(field, float(phys["wind_direction"]), ground_z_g_m=ground_z_g_m, t0_s=t0_s, z_below=z_below)
        if wind.speed_m_s != float(phys["wind_speed"]):
            raise ValueError(f"the field's hub mean {wind.speed_m_s} m/s is not the parameters' wind speed "
                             f"{float(phys['wind_speed'])} m/s")
        return wind

    def set_time(self, time_s):
        """The flight time (s) every evaluation uses until the next call (the period's start)."""
        self.time_s = float(time_s)

    def settings(self):
        f = self.field
        return dict(model="IEC 61400-1 Ed.3 NTM Kaimal (TurbSim IECKAI) full-field file, frozen turbulence (B2)",
                    file=f.path.name, sha256=f.sha256, description=f.description.strip(), periodic=f.periodic,
                    grid=dict(ny=f.ny, nz=f.nz, nt=f.nt, dy_m=f.dy_m, dz_m=f.dz_m, dt_s=f.dt_s, base_m=f.grid_base_m,
                              hub_m=f.hub_height_m, total_time_s=f.total_time_s),
                    mean_speed_m_s=self.speed_m_s, direction_rad=self.direction_rad, t0_s=self.t0_s,
                    ground_z_g_m=self.ground_z_g_m, sensor_pos_p_m=list(WIND_SENSOR_POS_P_M),
                    finite_difference_step_m=FINITE_DIFFERENCE_STEP_M, time_within_period="held at the period's start",
                    **({} if self.z_below == "raise" else dict(z_below=self.z_below)))

    # --- the field at points -------------------------------------------------------------------------------------
    def winds_g(self, points_g):
        """The wind (g coordinates) at many points (g coordinates), at ``time_s + t0_s``."""
        p = np.asarray(points_g, dtype=float).reshape(-1, 3)
        rel = p-np.array([0.0, 0.0, self.ground_z_g_m])               # TransformGToMw: subtract ground_z, then rotate
        x, y, z = rel@self.x_g, rel@self.y_g, rel@self.z_g
        if self.z_below == "saturate":                 # revision 23 addendum 9: Makani's Saturate(z / height, 0, 1)
            z = np.maximum(z, self.field.grid_base_m)
        uvw = self.field.velocity(self.time_s+self.t0_s, x, y, z)
        return uvw[:, 0:1]*self.x_g+uvw[:, 1:2]*self.y_g+uvw[:, 2:3]*self.z_g

    def wind_g(self, pos_g):
        return self.winds_g(np.asarray(pos_g, dtype=float).reshape(1, 3))[0]

    def winds_world(self, points_world):
        """The wind (world coordinates) at many world points: the vortex method's wake, rings and sections."""
        points = np.asarray(points_world, dtype=float).reshape(-1, 3)
        return self.winds_g(points@G_TO_WORLD)@G_TO_WORLD             # G_TO_WORLD is diagonal and its own inverse

    def node_winds_world(self, nodes_world):
        """One wind per tether node (world coordinates); the same evaluation as ``winds_world``."""
        return self.winds_world(nodes_world)

    def wind_world_at(self, pos_world):
        return self.winds_world(np.asarray(pos_world, dtype=float).reshape(1, 3))[0]

    def wind_omega_g(self, pos_g):
        """Half the curl of the wind by forward differences of 10 m, as environment.cc:122-158 (for every wind model)."""
        x, y, z = (float(v) for v in pos_g)
        h = FINITE_DIFFERENCE_STEP_M
        w, tx, ty, tz = self.winds_g([[x, y, z], [x+h, y, z], [x, y+h, z], [x, y, z+h]])
        dvdx, dwdx = (tx[1]-w[1])/h, (tx[2]-w[2])/h
        dudy, dwdy = (ty[0]-w[0])/h, (ty[2]-w[2])/h
        dudz, dvdz = (tz[0]-w[0])/h, (tz[1]-w[1])/h
        return np.array([0.5*(dwdy-dvdz), 0.5*(dudz-dwdx), 0.5*(dvdx-dudy)])

    def wind_rates_body(self, pose):
        omega_g = self.wind_omega_g(G_TO_WORLD@np.asarray(pose.position_m, dtype=float))
        return pose.rotation.T@(G_TO_WORLD@omega_g)

    def origin_wind_g(self, pose):
        return self.wind_g(G_TO_WORLD@np.asarray(pose.position_m, dtype=float))

    def sensor_height_agl_m(self):
        return self.ground_z_g_m-WIND_SENSOR_POS_P_M[2]

    def sensor_pos_g(self):
        """The GSv2 wind sensor in g: the platform frame's origin and z axis are the ground frame's (wind_shear.py)."""
        return np.array([WIND_SENSOR_POS_P_M[0], WIND_SENSOR_POS_P_M[1], WIND_SENSOR_POS_P_M[2]], dtype=float)

    def sensor_wind_g(self):
        """The field at the wind sensor at this time (full_system.cc:657-661: the sensor samples the environment)."""
        return self.wind_g(self.sensor_pos_g())
