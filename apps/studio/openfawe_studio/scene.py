"""The 3D scene (PyVista/VTK): the airframe at its pose, the wake filaments coloured by circulation, the rotor rings,
the wind and force arrows, the ground, the ground station and the tether. Style after the owner's reference video
(OF-0007 animate_loop.py): white background, wake in blue, fuselage grey, wind green, force red.

Drawn only (not in the aerodynamics of a studio run): the fuselage, the nacelles, the bridle, the tether (a straight
line from the bridle knot to the ground station: the runs have prescribed motion and no line) and the ground station
(winch drum and platform, schematic). On the M600 the eight rotors are the generators: their power is in the records
when the rotors are on."""
import json

import numpy as np

from . import resources
from .runstore import surface_quads, wake_lines

COLOURS = dict(surface="#2b3a4a", fuselage="#b8bfc9", disc="#8f9aa8", nacelle="#5d6673", ring="#e07b2a",
               wind="#2a9d8f", force="#d1495b", wake="#1f5f9e", tether="#2b2f36", station="#6b7785", ground="#eef1f4")
ELEVATION_DEG = 30.0      # the schematic tether's elevation: the ground station sits upwind at height / tan(30 deg)


def ground_point(spec):
    """Schematic ground station: upwind of the aircraft (fixed) or of the loop centre (circle), on the ground."""
    return np.array([-float(spec.height_m)/np.tan(np.radians(ELEVATION_DEG)), 0.0, 0.0])
MODES = {"blue": "Wake in blue (|Gamma|)", "signed": "Circulation, signed", "uniform": "Uniform"}


def _lines(points, edges):
    import pyvista as pv
    lines = np.column_stack([np.full(len(edges), 2), edges]).ravel() if len(edges) else np.zeros(0, dtype=np.int64)
    return pv.PolyData(np.asarray(points, dtype=float), lines=lines)


def _matrix(rotation, position):
    import vtk
    m = vtk.vtkMatrix4x4()
    for i in range(3):
        for j in range(3):
            m.SetElement(i, j, float(rotation[i, j]))
        m.SetElement(i, 3, float(position[i]))
    return m


class Scene:
    """Draws frames into a pyvista plotter (QtInteractor in the window, an off-screen Plotter for videos)."""

    def __init__(self, plotter, *, rotors=False, fuselage=True, ground=None, station=True):
        self.p = plotter
        self.station = station
        self.ground = None if ground is None else np.asarray(ground, float)
        self.tether_actor = None
        self.mode = "blue"
        self.show_wake = self.show_rings = self.show_arrows = True
        self.colour_surfaces = False
        self.follow = True
        self.body_actors = []
        self.wake_actor = self.ring_actor = self.surface_actor = None
        self.arrows = []
        self.limit = None
        self.p.set_background("white")
        self._airframe(rotors, fuselage)
        self._ground()
        self.station_drawn = False
        self.text = self.p.add_text("", position="upper_left", font_size=9, color="#333333")
        self.note = self.p.add_text("drawn only: fuselage, nacelles, bridle, tether, ground station",
                                    position="lower_right", font_size=7, color="#888888")
        self.p.enable_anti_aliasing("ssaa") if hasattr(self.p, "enable_anti_aliasing") else None
        self.first = True

    def _airframe(self, rotors, fuselage):
        import pyvista as pv
        if fuselage:
            try:
                with np.load(resources.path(resources.FUSELAGE)) as data:
                    nodes, quads = np.asarray(data["nodes"], float), np.asarray(data["quads"], np.int64)
                mesh = pv.PolyData(nodes, faces=np.column_stack([np.full(len(quads), 4), quads]).ravel())
                self.body_actors.append(self.p.add_mesh(mesh, color=COLOURS["fuselage"], smooth_shading=True,
                                                        ambient=0.35, diffuse=0.7, specular=0.15))
            except FileNotFoundError:
                pass
        if rotors:
            data = json.loads(resources.path(resources.ROTOR_SET).read_text(encoding="utf-8"))
            radius = 0.5*float(data["diameter_m"])
            for disc in data["rotors"]:
                centre, axis = np.asarray(disc["centre_m"], float), np.asarray(disc["axis"], float)
                self.body_actors.append(self.p.add_mesh(pv.Disc(center=centre, inner=0.12, outer=radius, normal=axis,
                                                                c_res=48), color=COLOURS["disc"], opacity=0.35,
                                                        lighting=False))
                self.body_actors.append(self.p.add_mesh(pv.Cylinder(center=centre-0.45*axis, direction=axis, radius=0.16,
                                                                    height=1.0), color=COLOURS["nacelle"]))

    def _ground(self):
        import pyvista as pv
        bridle = json.loads(resources.path(resources.RIGID_BODY).read_text(encoding="utf-8"))["bridle"]
        self.bridle = (np.asarray(bridle["attachments_m"]["port"], float), np.asarray(bridle["attachments_m"]["starboard"], float),
                       np.asarray(bridle["pivot_m"], float), float(bridle["radius_m"]))


    def _station(self):
        import pyvista as pv
        if self.ground is None or self.station_drawn or not self.station:
            return
        self.station_drawn = True
        g = self.ground
        plane = pv.Plane(center=(g[0]+250.0, g[1], g[2]-0.05), direction=(0, 0, 1), i_size=900.0, j_size=600.0,
                         i_resolution=36, j_resolution=24)
        self.p.add_mesh(plane, color=COLOURS["ground"], show_edges=True, edge_color="#d5dbe1", lighting=False)
        self.p.add_mesh(pv.Cylinder(center=g+[0, 0, 1.5], direction=(0, 0, 1), radius=6.0, height=3.0),
                        color=COLOURS["station"], smooth_shading=True)                       # platform
        self.p.add_mesh(pv.Cylinder(center=g+[0, 0, 4.5], direction=(0, 1, 0), radius=1.8, height=6.0),
                        color="#4a5260", smooth_shading=True)                                # winch drum
        self.p.add_mesh(pv.Cylinder(center=g+[2.5, 0, 8.0], direction=(0, 0, 1), radius=0.4, height=10.0),
                        color="#4a5260", smooth_shading=True)                                # perch/levelwind mast
        self.p.add_point_labels([g+[0, 0, 16.0]], ["ground station (schematic)"], font_size=10, text_color="#555555",
                                shape_opacity=0.0, show_points=False, always_visible=True)

    def _tether(self, rotation, position, line=None):
        import pyvista as pv
        port, starboard, pivot, radius = self.bridle
        place = lambda pts: np.asarray(pts, float)@rotation.T+position   # noqa: E731
        if line is not None and len(line) > 1:              # the computed lumped-mass line: nodes from the ground point
            if self.tether_actor is not None:
                for actor in self.tether_actor:
                    self.p.remove_actor(actor, render=False)
            knot = np.asarray(line[-1], float)
            self.tether_actor = [self.p.add_mesh(pv.lines_from_points(np.asarray(line, float)), color=COLOURS["tether"],
                                                 line_width=2.5, render=False),
                                 self.p.add_mesh(pv.lines_from_points(np.vstack([place([port]), knot, place([starboard])])),
                                                 color=COLOURS["tether"], line_width=1.5, render=False)]
            return
        if self.ground is not None:
            direction = rotation.T@(self.ground-place([pivot])[0])
            direction /= np.linalg.norm(direction)
        else:
            direction = np.array([0., 0., -1.])
        knot = pivot+radius*direction
        points = [place([port, knot, starboard])]
        if self.ground is not None:
            points.append(np.vstack([place([knot]), self.ground[None]]))
        if self.tether_actor is not None:
            for actor in self.tether_actor:
                self.p.remove_actor(actor, render=False)
        self.tether_actor = [self.p.add_mesh(pv.lines_from_points(pts), color=COLOURS["tether"], line_width=2.0,
                                             render=False) for pts in points]

    def _colours(self, gamma):
        if self.mode == "uniform" or len(gamma) == 0:
            return None, None, None
        if self.limit is None:
            scale = float(np.percentile(np.abs(gamma), 98)) or 1.0
        else:
            scale = float(self.limit)
        if self.mode == "signed":
            return gamma, "RdBu_r", (-scale, scale)
        return np.abs(gamma), "Blues", (0.0, scale)

    def draw(self, frame, *, label=""):
        import pyvista as pv
        p = self.p
        rotation, position = frame["rotation"], frame["position"]
        for actor in self.body_actors:
            actor.SetUserMatrix(_matrix(rotation, position))
        for actor in (self.wake_actor, self.ring_actor, self.surface_actor, *self.arrows):
            if actor is not None:
                p.remove_actor(actor, render=False)
        self.arrows = []
        points, quads = surface_quads(frame)
        surface = pv.PolyData(points, faces=np.column_stack([np.full(len(quads), 4), quads]).ravel())
        if self.colour_surfaces:
            surface.cell_data["bound Gamma [m2/s]"] = frame["bound_gamma"]
            self.surface_actor = p.add_mesh(surface, scalars="bound Gamma [m2/s]", cmap="viridis", show_edges=True,
                                            edge_color="#222222", render=False, show_scalar_bar=False)
        else:
            self.surface_actor = p.add_mesh(surface, color=COLOURS["surface"], show_edges=True, edge_color="#11181f",
                                            smooth_shading=False, render=False)
        edges, gamma, ring = wake_lines(frame)
        self.wake_actor = self.ring_actor = None
        if self.show_wake and np.any(~ring):
            mesh = _lines(frame["nodes"], edges[~ring])
            values, cmap, clim = self._colours(gamma[~ring])
            if values is None:
                self.wake_actor = p.add_mesh(mesh, color=COLOURS["wake"], line_width=2.0, render_lines_as_tubes=True,
                                             render=False)
            else:
                mesh.cell_data["Gamma [m2/s]"] = values
                self.wake_actor = p.add_mesh(mesh, scalars="Gamma [m2/s]", cmap=cmap, clim=clim, line_width=2.0,
                                             render_lines_as_tubes=True, render=False, scalar_bar_args=dict(title="Gamma [m2/s]", color="#333333",
                                                                                vertical=True, height=0.5,
                                                                                position_x=0.86, position_y=0.25, title_font_size=12, label_font_size=11))
        if self.show_rings and np.any(ring):
            self.ring_actor = p.add_mesh(_lines(frame["nodes"], edges[ring]), color=COLOURS["ring"], line_width=1.5,
                                         render_lines_as_tubes=True, opacity=0.8, render=False)
        ground = frame.get("ground")
        if ground is not None and np.all(np.isfinite(ground)):
            self.ground = np.asarray(ground, float)
        self._station()
        line = frame.get("line")
        computed = line is not None and len(line) > 1
        self._tether(rotation, position, line if computed else None)
        self.note.SetText(1, "drawn only: fuselage, nacelles, ground station (tether computed)" if computed else
                          "drawn only: fuselage, nacelles, bridle, tether, ground station")
        extent = [frame["nodes"], position[None]] + ([self.ground[None]] if self.ground is not None else [])
        extent = np.vstack(extent)
        self.extent = (extent.min(axis=0), extent.max(axis=0))
        if self.show_arrows:
            span = float(np.ptp(frame["leading_edge"][:, 1])) or 25.0
            force = np.asarray(frame["force"], float)
            if np.linalg.norm(force) > 0:
                self.arrows.append(p.add_mesh(pv.Arrow(start=position, direction=force/np.linalg.norm(force),
                                                       scale=0.25*span, tip_radius=0.06, shaft_radius=0.02),
                                              color=COLOURS["force"], render=False))
            wind = np.asarray(frame.get("wind", (1.0, 0.0, 0.0)), float).reshape(3)       # ST-07: the run's wind direction
            direction = wind/np.linalg.norm(wind) if np.linalg.norm(wind) > 0 else np.array([1.0, 0.0, 0.0])
            upstream = position-0.6*span*direction
            self.arrows.append(p.add_mesh(pv.Arrow(start=upstream-0.3*span*direction, direction=tuple(direction),
                                                   scale=0.25*span, tip_radius=0.06, shaft_radius=0.02),
                                          color=COLOURS["wind"], render=False))
        self.text.SetText(2, f"{label}t = {frame['time_s']:.2f} s   step {frame['step']}")
        if self.first:
            self.view("iso", position)
            self.first = False
        elif self.follow:
            camera = p.camera
            focal = np.asarray(camera.focal_point)
            shift = position-focal
            camera.focal_point = tuple(position)
            camera.position = tuple(np.asarray(camera.position)+shift)
        p.render()

    def view(self, name, centre=None):
        p = self.p
        centre = np.asarray(p.camera.focal_point if centre is None else centre, float)
        distance = 70.0
        if name == "far" and getattr(self, "extent", None) is not None:   # wake, aircraft, tether and ground station
            low, high = self.extent
            d = np.array([-1.0, -1.2, 0.8])/np.linalg.norm([-1.0, -1.2, 0.8])
            p.camera.focal_point = tuple(0.5*(low+high))
            p.camera.position = tuple(0.5*(low+high)+1000.0*d)
            p.camera.up = (0.0, 0.0, 1.0)
            pad = 0.08*(high-low)+5.0
            p.renderer.ResetCamera(*np.ravel(np.column_stack([low-pad, high+pad])))
            p.render()
            return
        if name == "near":                       # the filaments next to the surfaces
            distance = 18.0
            name = "iso"
        directions = dict(iso=(-1.0, -1.2, 0.8), top=(0.0, 0.0, 1.0), side=(0.0, -1.0, 0.0), front=(-1.0, 0.0, 0.0),
                          rear=(1.0, 0.0, 0.3))
        d = np.asarray(directions[name], float)
        d /= np.linalg.norm(d)
        p.camera.focal_point = tuple(centre)
        p.camera.position = tuple(centre+distance*d)
        p.camera.up = (0.0, 1.0, 0.0) if name == "top" else (0.0, 0.0, 1.0)
        p.render()
