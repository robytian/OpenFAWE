"""Off-body velocity fields of an archived vortex network on planes (wake probes).

Samples the induced velocity of the complete filament network (bound rings,
free wake rows, rotor rings when present) plus the ambient flow on a
rectangular grid, with the same Biot-Savart kernel, core model and per-edge
core radii as the computation itself; nothing is interpolated from the
solution. A probe point that happens to lie on a filament uses the regularised
on-filament value (``singularity='zero'``), which is the physical limit for a
positive core. Finite differences on the grid give the in-plane derivatives
(vorticity component normal to the plane); they are diagnostics of the sampled
field, not part of the computation.
"""

from dataclasses import dataclass

import numpy as np

from .vortex import segments_velocity


@dataclass(frozen=True)
class Plane:
    """Rectangular probe plane: origin, two in-plane unit axes, half-sizes and resolution."""
    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    half_u: float
    half_v: float
    count_u: int
    count_v: int

    def __post_init__(self):
        for name in ("origin", "axis_u", "axis_v"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float).reshape(3))
        for name in ("axis_u", "axis_v"):
            vector = getattr(self, name)
            norm = np.linalg.norm(vector)
            if norm == 0.0:
                raise ValueError(f"{name} must be non-zero")
            object.__setattr__(self, name, vector / norm)
        if abs(np.dot(self.axis_u, self.axis_v)) > 1e-12:
            raise ValueError("plane axes must be orthogonal")
        if self.count_u < 2 or self.count_v < 2 or self.half_u <= 0 or self.half_v <= 0:
            raise ValueError("plane needs at least a 2x2 grid with positive half-sizes")

    @property
    def normal(self):
        return np.cross(self.axis_u, self.axis_v)

    @property
    def spacing(self):
        return 2 * self.half_u / (self.count_u - 1), 2 * self.half_v / (self.count_v - 1)

    def coordinates(self):
        """In-plane coordinate vectors (u, v) and the (count_v, count_u, 3) points."""
        u = np.linspace(-self.half_u, self.half_u, self.count_u)
        v = np.linspace(-self.half_v, self.half_v, self.count_v)
        uu, vv = np.meshgrid(u, v)
        points = self.origin + uu[..., None] * self.axis_u + vv[..., None] * self.axis_v
        return u, v, points


def crossflow_plane(x, half_y, half_z, count_y, count_z, centre_y=0.0, centre_z=0.0):
    """Plane normal to x (the downstream axis) at station x: u = y, v = z."""
    return Plane([x, centre_y, centre_z], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], half_y, half_z, count_y, count_z)


def symmetry_plane(y, centre_x, half_x, centre_z, half_z, count_x, count_z):
    """Plane y = const: u = x, v = z (the pitch plane)."""
    return Plane([centre_x, y, centre_z], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], half_x, half_z, count_x, count_z)


def network_velocity(points, nodes, edges, gamma, *, model, core_radius, ambient=None, backend="numba",
                     tree=None, chunk=20000):
    """Ambient + induced velocity of the network at ``points`` ((n,3) -> (n,3))."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    velocity = np.zeros_like(points)
    if ambient is not None:
        velocity += np.asarray(ambient(points) if callable(ambient) else ambient, dtype=float)
    kernel = dict(backend=backend) if tree is None else dict(backend="tree", tree=tree)
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        velocity[start:start + chunk] += segments_velocity(block, nodes, edges, gamma, model=model,
                                                           core_radius=core_radius, singularity="zero", **kernel)
    return velocity


def plane_fields(plane, nodes, edges, gamma, *, model, core_radius, freestream, backend="numba", tree=None):
    """Velocity field on the plane and derived scalars.

    Returns a dict with ``u``/``v`` coordinates, ``points`` (nv, nu, 3), ``velocity``
    (nv, nu, 3), ``speed``, ``axial`` (component along the freestream direction),
    ``in_plane`` ((nv, nu, 2): components along axis_u, axis_v), ``normal_vorticity``
    (d v_v/du - d v_u/dv by central differences, 1/s) and the plane geometry.
    """
    freestream = np.asarray(freestream, dtype=float).reshape(3)
    u, v, points = plane.coordinates()
    flat = points.reshape(-1, 3)
    velocity = network_velocity(flat, nodes, edges, gamma, model=model, core_radius=core_radius,
                                ambient=np.tile(freestream, (len(flat), 1)), backend=backend, tree=tree)
    velocity = velocity.reshape(points.shape)
    direction = freestream / max(np.linalg.norm(freestream), 1e-300)
    axial = velocity @ direction
    in_plane = np.stack([velocity @ plane.axis_u, velocity @ plane.axis_v], axis=-1)
    du, dv = plane.spacing
    dvv_du = np.gradient(in_plane[..., 1], du, axis=1)
    dvu_dv = np.gradient(in_plane[..., 0], dv, axis=0)
    return dict(u=u, v=v, points=points, velocity=velocity, speed=np.linalg.norm(velocity, axis=-1), axial=axial,
                in_plane=in_plane, normal_vorticity=dvv_du - dvu_dv, plane=plane,
                freestream_speed=float(np.linalg.norm(freestream)))


def line_probe(start, end, count, nodes, edges, gamma, *, model, core_radius, freestream, backend="numba", tree=None):
    """Velocity along a straight line (e.g. the tail's quarter-chord line): (points, velocity)."""
    points = np.linspace(np.asarray(start, dtype=float), np.asarray(end, dtype=float), count)
    velocity = network_velocity(points, nodes, edges, gamma, model=model, core_radius=core_radius,
                                ambient=np.tile(np.asarray(freestream, dtype=float), (count, 1)), backend=backend,
                                tree=tree)
    return points, velocity


def structured_grid(fields):
    """PyVista StructuredGrid of a plane_fields result with the scalar/vector arrays attached."""
    import pyvista as pv
    points = fields["points"]
    nv, nu = points.shape[:2]
    grid = pv.StructuredGrid()
    grid.points = points.reshape(-1, 3)
    grid.dimensions = [nu, nv, 1]
    speed_ref = fields["freestream_speed"]
    grid.point_data["velocity_m_s"] = fields["velocity"].reshape(-1, 3)
    grid.point_data["speed_over_Uinf"] = fields["speed"].reshape(-1) / speed_ref
    grid.point_data["axial_over_Uinf"] = fields["axial"].reshape(-1) / speed_ref
    grid.point_data["normal_vorticity_1_s"] = fields["normal_vorticity"].reshape(-1)
    grid.point_data["in_plane_m_s"] = np.concatenate([fields["in_plane"].reshape(-1, 2),
                                                      np.zeros((nu * nv, 1))], axis=1)
    return grid
