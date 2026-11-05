"""The complete M600 airframe for the 3D figures of the paper (author 2026-10-19 17:46: every visualisation shows the
complete airframe). Display only: of these parts only the wing and the two tails (and, with rotors, the rotor discs) enter
the computations of the runs; the fuselage and the pylons are drawn for orientation (captions say so).

Parts, all in the body frame of the aerodynamic case (x towards the trailing edge, y along the span, z towards the lift
side; the frame of cases/qualification/m600_makani_reconstructed.json, to which the fuselage mesh is registered):
  wing, horizontal and vertical tail, four pylons   mean surfaces of cases/qualification/m600_makani_reconstructed_pylons.json
  fuselage                                         paper1/crosswind/mesh/fuselage-s015-r24.npz (registered
                                                   to the same case, registration.json there)
  rotors                                           eight discs of 2.3 m with nacelles, cases/rotors/m600_rotors_makani.json
  bridle and tether                                bridle legs from the two attachment points of cases/aircraft/
                                                   m600_makani_rigid_body.json to the bridle point, and a straight line to
                                                   the ground point (schematic: the computed line sags and is not drawn)
"""
import json
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[1]
SURFACES = REPO/"cases/qualification/m600_makani_reconstructed_pylons.json"
FUSELAGE = TASK/"mesh/fuselage-s015-r24.npz"
ROTORS = REPO/"cases/rotors/m600_rotors_makani.json"
RIGID = REPO/"cases/aircraft/m600_makani_rigid_body.json"
COLOURS = dict(surface="#d9dde3", edge="#4a5260", fuselage="#b8bfc9", disc="#8f9aa8", nacelle="#5d6673", tether="#2b2f36")


def _parts():
    from openfawe.aero.aircraft import load_aircraft
    wing = load_aircraft(SURFACES).wing
    with np.load(FUSELAGE) as data:
        fuselage = (np.asarray(data["nodes"], dtype=float), np.asarray(data["quads"], dtype=np.int64))
    rotors = json.loads(ROTORS.read_text(encoding="utf-8"))
    bridle = json.loads(RIGID.read_text(encoding="utf-8"))["bridle"]
    return wing, fuselage, rotors, bridle


_CACHE = {}


def _basis(axis):
    """Rotation whose third column is ``axis`` (maps the x-y plane of pv.Circle onto the disc plane)."""
    z = np.asarray(axis, dtype=float)/np.linalg.norm(axis)
    helper = np.array([0., 0., 1.]) if abs(z[2]) < 0.9 else np.array([1., 0., 0.])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def parts():
    if not _CACHE:
        _CACHE["value"] = _parts()
    return _CACHE["value"]


def add_airframe(plot, rotation, position, *, ground_point=None, rotors=True, scale_lines=1.0, disc_outline=False):
    """Add the airframe at the pose (rotation body to inertial, position of the body origin) to a PyVista plotter."""
    import pyvista as pv
    rotation, position = np.asarray(rotation, dtype=float), np.asarray(position, dtype=float)
    place = lambda pts: np.asarray(pts, dtype=float)@rotation.T+position   # noqa: E731
    wing, (fnodes, fquads), rotor_set, bridle = parts()
    for surface in wing.wings:
        le, te = place(surface.leading_edge), place(surface.trailing_edge)
        n = len(le)
        faces = np.hstack([[4, k, k+1, n+k+1, n+k] for k in range(n-1)])
        mesh = pv.PolyData(np.vstack([le, te]), faces=faces)
        plot.add_mesh(mesh, color=COLOURS["surface"], smooth_shading=True, ambient=0.35, diffuse=0.7, specular=0.15)
        outline = np.vstack([le, te[::-1], le[:1]])
        plot.add_mesh(pv.lines_from_points(outline), color=COLOURS["edge"], line_width=1.2*scale_lines, lighting=False)
    faces = np.column_stack([np.full(len(fquads), 4), fquads]).ravel()
    plot.add_mesh(pv.PolyData(place(fnodes), faces=faces), color=COLOURS["fuselage"], smooth_shading=True, ambient=0.35,
                  diffuse=0.7, specular=0.15)
    if rotors:
        radius = 0.5*float(rotor_set["diameter_m"])
        for disc in rotor_set["rotors"]:
            centre = place([disc["centre_m"]])[0]
            axis = rotation@np.asarray(disc["axis"], dtype=float)
            plot.add_mesh(pv.Disc(center=centre, inner=0.12, outer=radius, normal=axis, r_res=1, c_res=48),
                          color=COLOURS["disc"], opacity=0.45, lighting=False)
            plot.add_mesh(pv.Cylinder(center=centre-0.45*axis, direction=axis, radius=0.16, height=1.0, resolution=24),
                          color=COLOURS["nacelle"], smooth_shading=True)
            if disc_outline:                                   # runs with rotor rings: the disc edge in black
                ring = pv.Circle(radius=radius, resolution=72)
                ring.points = ring.points@_basis(axis).T+centre
                plot.add_mesh(ring.extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False),
                              color="black", line_width=2.0*scale_lines, lighting=False)
    attach = [bridle["attachments_m"]["port"], bridle["attachments_m"]["starboard"]]
    pivot = np.asarray(bridle["pivot_m"], dtype=float)
    if ground_point is not None:
        direction = rotation.T@(np.asarray(ground_point, dtype=float)-place([pivot])[0])
        direction /= np.linalg.norm(direction)
    else:
        direction = np.array([0., 0., -1.])
    knot = pivot+float(bridle["radius_m"])*direction
    legs = place([attach[0], knot, attach[1]])
    plot.add_mesh(pv.lines_from_points(legs), color=COLOURS["tether"], line_width=1.0*scale_lines, lighting=False)
    if ground_point is not None:
        plot.add_mesh(pv.lines_from_points(np.vstack([place([knot]), np.asarray(ground_point, dtype=float)[None]])),
                      color=COLOURS["tether"], line_width=1.4*scale_lines, lighting=False)


GS02 = REPO/"cases/ground/makani_gs02_parker_ranch.json"
GS_COLOURS = dict(structure="#aeb4bd", drum="#5d6673", housing="#c9cdd3", edge="#4a5260")


def ground_station_geometry():
    """Dimensions of the Makani GS02 ground station used for drawing (cases/ground/makani_gs02_parker_ranch.json):
    platform height above the ground ``ground_frame.ground_z``, drum radius and centre ``drum_origin_p``, tether exit at the
    level wind ``levelwind_origin_p`` and the perched-wing position ``perched_wing_pos_p`` (platform frame, z down there,
    converted to z up here). Tower radius, drum length, platform plan and the housing are schematic."""
    g = json.loads(GS02.read_text(encoding="utf-8"))
    s = g["ground_station"]["gs02"]
    up = lambda v: np.array([v[0], v[1], -v[2]], dtype=float)   # noqa: E731
    return dict(platform_height_m=float(g["ground_frame"]["ground_z"]), drum_radius_m=float(s["drum_radius"]),
                drum_centre_m=up(s["drum_origin_p"]), levelwind_m=up(s["levelwind_origin_p"]), perch_m=up(s["perched_wing_pos_p"]),
                tower_radius_m=1.2, drum_length_m=3.0, housing_m=(6.0, 2.5, 2.6))


def add_ground_station(plot, anchor, toward=(1.0, 0.0, 0.0)):
    """Draw the ground station so that its tether exit (the level wind) is at ``anchor``, the tether ground point of the
    computation, with the platform facing ``toward`` (horizontal). Display only; nothing of it enters the computation.
    Returns the height of the ground below the anchor."""
    import pyvista as pv
    gs = ground_station_geometry()
    ex = np.array([toward[0], toward[1], 0.0], dtype=float)
    ex /= np.linalg.norm(ex)
    ey = np.cross([0.0, 0.0, 1.0], ex)
    R = np.column_stack([ex, ey, [0.0, 0.0, 1.0]])
    origin = np.asarray(anchor, dtype=float)-R@gs["levelwind_m"]                 # platform origin
    ground_z = origin[2]-gs["platform_height_m"]
    at = lambda p: R@np.asarray(p, dtype=float)+origin                         # noqa: E731
    style = dict(smooth_shading=True, ambient=0.35, diffuse=0.7, specular=0.1)
    h = gs["platform_height_m"]
    plot.add_mesh(pv.Cylinder(center=at([0., 0., -0.5*h]), direction=(0, 0, 1), radius=gs["tower_radius_m"], height=h, resolution=48),
                  color=GS_COLOURS["structure"], **style)
    reach = max(gs["perch_m"][1]+1.0, 4.0)
    box = pv.Box(bounds=(-3.0, 3.0, -2.5, reach, -0.5, 0.0))
    box.points = box.points@R.T+origin
    plot.add_mesh(box, color=GS_COLOURS["structure"], **style)
    plot.add_mesh(pv.Cylinder(center=at(gs["drum_centre_m"]), direction=R@np.array([0., 1., 0.]), radius=gs["drum_radius_m"],
                              height=gs["drum_length_m"], resolution=64), color=GS_COLOURS["drum"], **style)
    for side in (-1.0, 1.0):                                                    # drum supports
        foot = gs["drum_centre_m"]+np.array([0., side*0.5*gs["drum_length_m"], 0.])
        plot.add_mesh(pv.lines_from_points(np.vstack([at(foot), at([foot[0], foot[1], 0.])])), color=GS_COLOURS["edge"],
                      line_width=4.0, lighting=False)
    lw = gs["levelwind_m"]
    plot.add_mesh(pv.lines_from_points(np.vstack([at([lw[0], lw[1], 0.]), at(lw)])), color=GS_COLOURS["edge"], line_width=5.0,
                  lighting=False)
    plot.add_mesh(pv.Sphere(radius=0.35, center=at(lw)), color=GS_COLOURS["edge"])
    hx, hy, hz = gs["housing_m"]
    house = pv.Box(bounds=(-gs["tower_radius_m"]-1.0-hx, -gs["tower_radius_m"]-1.0, -0.5*hy, 0.5*hy, -h, -h+hz))
    house.points = house.points@R.T+origin
    plot.add_mesh(house, color=GS_COLOURS["housing"], **style)
    return float(anchor[2]-ground_z)
