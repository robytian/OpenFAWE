# SPDX-License-Identifier: Apache-2.0
# Transcribed from google/makani commit 1d66923 (Apache-2.0): sim/models/ground_station_util.cc (ContactPerchPanel),
# sim/physics/contactor.{h,cc} (Contactor, ContactCircle, ContactTwoCircles), sim/math/util.cc (IntersectCircles),
# common/c_math/vec2.c, vec3.c (norms and Normalize); parameters from config/base_station/sim/gs02_sim.py and
# config/m600/sim/contact_sim.py. Modifications (2026): Python/NumPy; a helper that builds Makani's wing frame from the
# plant's rigid-body state.
"""The GS02 perch: the panel surface and the wing's penalty contactors (plan 3.7 part 2, criterion G5).

Makani holds the perched wing with three contactors on the wing (two talons, one peg; contact_sim.py) pushing
against the panel surface of the platform (``GroundStationV2Base::panel_surface``); the forces act on the wing only
(full_system.cc:734-745). The surface (ContactPerchPanel, ground_station_util.cc:43-117) is, in the panel frame, the
union of two vertical cylinders (radius 4 m) cut by the plane of the panel top (z_p = -4.332 in the platform frame);
a contactor inside the union and below the top plane collides, and the collision point is the nearer of its radial
projection onto the cylinder it is in and its projection onto the top plane. The contact force (Contactor::
CalcContactForce, contactor.cc:60-97) is a spring along the penetration, a damper on the normal velocity only while
the contactor moves further in, and friction proportional to the spring force: viscous below a sliding speed of
1 m/s, Coulomb above.

As coded, not repaired: B8, a misplaced parenthesis in the choice of the top plane (ground_station_util.cc:95-96;
harmless here, both panel tops are at the same height); the bottom of the panels (``z_extents_p[1]``) is never used;
a point within 0.4 m of a cylinder axis stops the process in the source (here a ``RuntimeError``).

Frames are Makani's (``gs02.Frame``; NED root); the force is returned in the wing's body frame (Makani's body axes)
at the contactor's body position, as the source returns it.
"""

import math

import numpy as np

from openfawe.vec3 import cross3   # OF-0014: numpy.cross for 3-vectors, bit for bit

from .gs02 import Frame


def _norm2(v):
    return math.hypot(v[0], v[1])                       # Vec2Norm


def _norm3(v):
    return math.hypot(v[0], math.hypot(v[1], v[2]))     # Vec3Norm


def _normalize3(v):
    """Vec3Normalize (vec3.c:140-178) for finite input."""
    v_max = max(abs(v[0]), max(abs(v[1]), abs(v[2])))
    if v_max > 0.0:
        denom = math.sqrt((v[0]/v_max)*(v[0]/v_max) + (v[1]/v_max)*(v[1]/v_max) + (v[2]/v_max)*(v[2]/v_max))
        return np.array([(v[0]/v_max)/denom, (v[1]/v_max)/denom, (v[2]/v_max)/denom])
    return np.zeros(3)


def _normalize2(v):
    """Vec2Normalize (vec2.c:103-135) for finite, nonzero input."""
    v_max = max(abs(v[0]), abs(v[1]))
    if not v_max > 0.0:
        raise ValueError("Vec2Normalize: vector is not normalisable")
    denom = math.sqrt((v[0]/v_max)*(v[0]/v_max) + (v[1]/v_max)*(v[1]/v_max))
    return np.array([(v[0]/v_max)/denom, (v[1]/v_max)/denom])


def intersect_circles(d, r1, r2):
    """IntersectCircles (sim/math/util.cc:67-86): (intersect, x, y)."""
    if abs(d) > 0.0:
        c = (-d + r1 - r2)*(-d - r1 + r2)*(-d + r1 + r2)*(d + r1 + r2)
        return c >= 0.0, (d*d + r1*r1 - r2*r2)/(2.0*d), math.sqrt(max(c, 0.0))/(2.0*d)
    return abs(r1 - r2) < min(r1, r2)*np.finfo(float).eps, r1, 0.0


def contact_circle(pos, radius, center):
    """ContactCircle (contactor.cc:99-116): (collides, collision point)."""
    if not radius > 0.0:
        raise ValueError("The radius must be positive.")
    offset = np.asarray(pos, dtype=float) - np.asarray(center, dtype=float)
    offset_norm = _norm2(offset)
    if offset_norm < 0.1*radius:
        raise RuntimeError("The contact point is too close to the center of the circle.")      # LOG(FATAL)
    if offset_norm >= radius:
        return False, None
    return True, 1.0*np.asarray(center, dtype=float) + (radius/offset_norm)*offset


def contact_two_circles(pos, radius1, center1, radius2, center2):
    """ContactTwoCircles (contactor.cc:118-158)."""
    collides_1, collision_1 = contact_circle(pos, radius1, center1)
    collides_2, collision_2 = contact_circle(pos, radius2, center2)
    if collides_1 and collides_2:
        center1, center2 = np.asarray(center1, dtype=float), np.asarray(center2, dtype=float)
        circle_offset = center2 - center1
        circles_intersect, x, y = intersect_circles(_norm2(circle_offset), radius1, radius2)
        if not circles_intersect:
            raise RuntimeError("Circles do not intersect.")                                    # DCHECK
        direction = _normalize2(circle_offset)
        perpendicular = np.array([-direction[1], direction[0]])
        contactor_offset1 = np.asarray(pos, dtype=float) - center1
        dot = contactor_offset1[0]*perpendicular[0] + contactor_offset1[1]*perpendicular[1]
        sign_y = float(int(0.0 < dot) - int(dot < 0.0))
        collision = 1.0*center1 + x*direction + (sign_y*y)*perpendicular
    elif collides_1:
        collision = collision_1
    else:
        collision = collision_2
    return collides_1 or collides_2, collision


def contact_perch_panel(contactor_pos_panel, platform_frame, panel_frame, panel):
    """ContactPerchPanel (ground_station_util.cc:43-117): (collides, collision point in the panel frame)."""
    port, starboard = panel["port"], panel["starboard"]
    if not (port["z_extents_p"][0] < port["z_extents_p"][1] and starboard["z_extents_p"][0] < starboard["z_extents_p"][1]):
        raise ValueError("panel z extents are not ordered")                                    # DCHECKs
    contactor_pos_panel = np.asarray(contactor_pos_panel, dtype=float)
    contactor_pos_p = panel_frame.transform_to(platform_frame, "position", contactor_pos_panel)
    if contactor_pos_p[2] < port["z_extents_p"][0]:
        return False, None                                     # above the panel top
    xy = contactor_pos_panel[:2]
    if contactor_pos_p[2] < starboard["z_extents_p"][0]:
        collides, xy_collision = contact_circle(xy, port["radius"], port["center_panel"])
    else:
        collides, xy_collision = contact_two_circles(xy, port["radius"], port["center_panel"], starboard["radius"],
                                                     starboard["center_panel"])
    if not collides:
        return False, None
    port_offset = xy_collision - np.asarray(port["center_panel"], dtype=float)
    starboard_offset = xy_collision - np.asarray(starboard["center_panel"], dtype=float)
    top_collision_pos_p = np.array([contactor_pos_p[0], contactor_pos_p[1], 0.0])
    # B8: as coded (95-96), the parenthesis closes before "- radius" on the right-hand side.
    if abs(_norm2(port_offset) - port["radius"]) < abs(_norm2(starboard_offset)) - starboard["radius"]:
        top_collision_pos_p[2] = port["z_extents_p"][0]
    else:
        top_collision_pos_p[2] = starboard["z_extents_p"][0]
    top_collision_pos_panel = platform_frame.transform_to(panel_frame, "position", top_collision_pos_p)
    cylinder_collision_pos_panel = np.array([xy_collision[0], xy_collision[1], contactor_pos_panel[2]])
    dist_top = _norm3(contactor_pos_panel - top_collision_pos_panel)
    dist_cylinder = _norm3(contactor_pos_panel - cylinder_collision_pos_panel)
    return True, (top_collision_pos_panel if dist_top < dist_cylinder else cylinder_collision_pos_panel)


def contact_branch(contactor_pos_panel, platform_frame, panel_frame, panel):
    """Which branch of ``contact_perch_panel`` (and of the circle helpers it calls) a contactor is in, as a hashable
    tuple (OF-0011 revision 7 addendum 4 supplement 4). The collision point, and with it the spring force, jumps where
    the branch changes - e.g. from inside the first circle only (the projection on it) to inside both (the circles'
    intersection) - so an event-located step watches it. The same tests in the same order as
    ``contact_perch_panel``; that function is not changed.

    ``("above",)`` above the panel top; otherwise ``(zone, inside)`` with no collision, or
    ``(zone, inside, side, surface, top)``: zone "port" (one circle) or "two"; ``inside`` per circle; ``side`` the
    intersection's side (``sign_y``) when inside both, else 0.0; surface "top" or "cylinder"; ``top`` the panel whose
    top height is taken (B8 as coded)."""
    port, starboard = panel["port"], panel["starboard"]
    contactor_pos_panel = np.asarray(contactor_pos_panel, dtype=float)
    contactor_pos_p = panel_frame.transform_to(platform_frame, "position", contactor_pos_panel)
    if contactor_pos_p[2] < port["z_extents_p"][0]:
        return ("above",)
    xy = contactor_pos_panel[:2]
    side = 0.0
    if contactor_pos_p[2] < starboard["z_extents_p"][0]:
        zone = "port"
        hit, xy_collision = contact_circle(xy, port["radius"], port["center_panel"])
        inside = (bool(hit),)
    else:
        zone = "two"
        hit_1, _ = contact_circle(xy, port["radius"], port["center_panel"])
        hit_2, _ = contact_circle(xy, starboard["radius"], starboard["center_panel"])
        inside = (bool(hit_1), bool(hit_2))
        hit, xy_collision = contact_two_circles(xy, port["radius"], port["center_panel"], starboard["radius"],
                                                starboard["center_panel"])
        if hit_1 and hit_2:
            center1 = np.asarray(port["center_panel"], dtype=float)
            direction = _normalize2(np.asarray(starboard["center_panel"], dtype=float) - center1)
            offset = np.asarray(xy, dtype=float) - center1
            dot = offset[0]*(-direction[1]) + offset[1]*direction[0]
            side = float(int(0.0 < dot) - int(dot < 0.0))
    if not hit:
        return (zone, inside)
    port_offset = xy_collision - np.asarray(port["center_panel"], dtype=float)
    starboard_offset = xy_collision - np.asarray(starboard["center_panel"], dtype=float)
    top_collision_pos_p = np.array([contactor_pos_p[0], contactor_pos_p[1], 0.0])
    if abs(_norm2(port_offset) - port["radius"]) < abs(_norm2(starboard_offset)) - starboard["radius"]:
        top, top_collision_pos_p[2] = "port", port["z_extents_p"][0]
    else:
        top, top_collision_pos_p[2] = "starboard", starboard["z_extents_p"][0]
    top_collision_pos_panel = platform_frame.transform_to(panel_frame, "position", top_collision_pos_p)
    cylinder_collision_pos_panel = np.array([xy_collision[0], xy_collision[1], contactor_pos_panel[2]])
    dist_top = _norm3(contactor_pos_panel - top_collision_pos_panel)
    dist_cylinder = _norm3(contactor_pos_panel - cylinder_collision_pos_panel)
    return (zone, inside, side, "top" if dist_top < dist_cylinder else "cylinder", top)


def _circle_point(pos, radius, center):
    """ContactCircle's collision point (contactor.cc:99-116) without the inside test, so that it also extends outside
    the circle (OF-0011 revision 7 addendum 4 supplement 5); the same arithmetic and the same stop near the axis."""
    if not radius > 0.0:
        raise ValueError("The radius must be positive.")
    offset = np.asarray(pos, dtype=float) - np.asarray(center, dtype=float)
    offset_norm = _norm2(offset)
    if offset_norm < 0.1*radius:
        raise RuntimeError("The contact point is too close to the center of the circle.")      # LOG(FATAL)
    return 1.0*np.asarray(center, dtype=float) + (radius/offset_norm)*offset


def contact_point_held(contactor_pos_panel, platform_frame, panel_frame, panel, branch):
    """The collision point (panel frame) of ``contact_perch_panel`` worked out for a given branch (``contact_branch``)
    rather than for the branch the contactor is in: within an event-located step the contact force keeps the mode of
    the step's start and extends smoothly past the branch's edges (OF-0011 revision 7 addendum 4 supplement 5). None
    for a branch without a collision. Where ``branch`` is the contactor's own branch, the arithmetic is that of
    ``contact_perch_panel`` and the point is the same, bit for bit."""
    if len(branch) != 5:
        return None
    zone, inside, side, surface, top = branch
    port, starboard = panel["port"], panel["starboard"]
    contactor_pos_panel = np.asarray(contactor_pos_panel, dtype=float)
    contactor_pos_p = panel_frame.transform_to(platform_frame, "position", contactor_pos_panel)
    xy = contactor_pos_panel[:2]
    if zone == "port" or tuple(inside) == (True, False):
        xy_collision = _circle_point(xy, port["radius"], port["center_panel"])
    elif tuple(inside) == (False, True):
        xy_collision = _circle_point(xy, starboard["radius"], starboard["center_panel"])
    else:                                                       # inside both: ContactTwoCircles' intersection
        center1 = np.asarray(port["center_panel"], dtype=float)
        center2 = np.asarray(starboard["center_panel"], dtype=float)
        circle_offset = center2 - center1
        circles_intersect, x, y = intersect_circles(_norm2(circle_offset), port["radius"], starboard["radius"])
        if not circles_intersect:
            raise RuntimeError("Circles do not intersect.")                                    # DCHECK
        direction = _normalize2(circle_offset)
        perpendicular = np.array([-direction[1], direction[0]])
        xy_collision = 1.0*center1 + x*direction + (float(side)*y)*perpendicular
    if surface == "top":
        top_collision_pos_p = np.array([contactor_pos_p[0], contactor_pos_p[1], 0.0])
        top_collision_pos_p[2] = port["z_extents_p"][0] if top == "port" else starboard["z_extents_p"][0]
        return platform_frame.transform_to(panel_frame, "position", top_collision_pos_p)
    return np.array([xy_collision[0], xy_collision[1], contactor_pos_panel[2]])


def contact_force(contactor, contactor_pos, contactor_vel, collision_pos, collision_vel):
    """Contactor::CalcContactForce (contactor.cc:60-97), all vectors in the surface frame."""
    contact_offset = np.asarray(contactor_pos, dtype=float) - np.asarray(collision_pos, dtype=float)
    unit_contact_offset = contact_offset*(1.0/max(_norm3(contact_offset), 1e-9))
    contact_vel = np.asarray(contactor_vel, dtype=float) - np.asarray(collision_vel, dtype=float)
    contact_vel_perp = unit_contact_offset*float(np.dot(contact_vel, unit_contact_offset))
    contact_vel_para = contact_vel - contact_vel_perp
    spring_force = contact_offset*(-contactor["spring_const"])
    damping_force = np.zeros(3)
    if float(np.dot(contact_vel_perp, contact_offset)) > 0.0:
        damping_force = contact_vel_perp*(-contactor["damping_coeff"])
    if _norm3(contact_vel_para) > 1.0:
        contact_vel_para = _normalize3(contact_vel_para)
    friction_force = contact_vel_para*(-contactor["friction_coeff"]*_norm3(spring_force))
    return spring_force + damping_force + friction_force


def contactor_force(contactor, wing_frame, platform_frame, panel_frame, panel, branch=None):
    """Contactor::CalcContactForceMomentPos (contactor.cc:27-58) against the panel surface.

    ``wing_frame``: Makani's wing frame (``gs02.Frame``: root NED, origin the body origin, root-to-body DCM, angular
    velocity in root coordinates). Returns (force in body axes, contactor position in body axes, collided).
    ``branch`` (OF-0011 revision 7 addendum 4 supplement 5; None: as before): the collision point of that branch
    (``contact_point_held``) in place of the contactor's own.
    """
    pos_b = np.asarray(contactor["pos"], dtype=float)
    frame = wing_frame.child(pos_b)
    pos_surf = panel_frame.from_root("position", frame.pos)
    if branch is None:
        collides, collision_pos_surf = contact_perch_panel(pos_surf, platform_frame, panel_frame, panel)
    else:
        collision_pos_surf = contact_point_held(pos_surf, platform_frame, panel_frame, panel, branch)
        collides = collision_pos_surf is not None
    if not collides:
        return np.zeros(3), pos_b, False
    vel_surf = panel_frame.from_root("velocity", frame.vel)
    collision_frame = panel_frame.child(collision_pos_surf)
    collision_vel_surf = panel_frame.from_root("velocity", collision_frame.vel)
    force_surf = contact_force(contactor, pos_surf, vel_surf, collision_pos_surf, collision_vel_surf)
    return panel_frame.rotate_to(wing_frame, force_surf), pos_b, True


def makani_wing_frame(position_origin_ned, velocity_origin_ned, dcm_g2b, omega_body_makani):
    """Makani's wing frame (wing.cc:531-534): ReferenceFrame(ground, Xg, Vg, dcm_g2b, omega in ground axes)."""
    dcm_g2b = np.asarray(dcm_g2b, dtype=float)
    omega_g = dcm_g2b.T@np.asarray(omega_body_makani, dtype=float)
    return Frame.root().child(position_origin_ned, dcm_g2b, velocity_origin_ned, omega_g)


# ----- the wing on the perch: GS02, variable tether, rigid wing, contacts (criteria G5-G7) ---------------------------
def quat_to_dcm(q):
    """QuatToDcm (common/c_math/quaternion.c:275-296), scalar first."""
    q = np.asarray(q, dtype=float)
    q0, q1, q2, q3 = q/np.linalg.norm(q)
    return np.array([[q0*q0 + q1*q1 - q2*q2 - q3*q3, 2.0*(q1*q2 + q0*q3), 2.0*(q1*q3 - q0*q2)],
                     [2.0*(q1*q2 - q0*q3), q0*q0 - q1*q1 + q2*q2 - q3*q3, 2.0*(q2*q3 + q0*q1)],
                     [2.0*(q1*q3 + q0*q2), 2.0*(q2*q3 - q0*q1), q0*q0 - q1*q1 - q2*q2 + q3*q3]])


def default_rigid_body_case():
    from pathlib import Path
    return Path(__file__).resolve().parents[3]/"cases"/"aircraft"/"m600_makani_rigid_body.json"


class PerchedSystem:
    """The ground station, the variable-length tether and the rigid wing with its perch contactors, as one ODE.

    Frames: the wing and the tether in the plant's z-up world frame (``flight.makani_interface.G_TO_WORLD`` maps
    Makani's g frame to it); the ground station in Makani's NED. Loads on the wing: gravity, the three contactors
    against the panel surface (``contactor_force``), the tether's end force at the bridle point; no aerodynamics, no
    rotor. The tether sees no wind (its drag acts on the node velocities only). One evaluation follows the order of
    the simulator's connections (full_system.cc:696-745): ground-station state -> free length and anchor -> tether
    indices -> bridle point from the last node -> tether forces -> wing loads.
    """
    held_branches = None        # OF-0011 rev 7 add 4 supplement 5: each contactor's branch held within an event step

    def __init__(self, params, body=None, bridle=None, line=None, tether_ground_contact=False, ground_z_g_m=None,
                 floating=None):
        """``tether_ground_contact`` (opt-in, default off; full-cycle contract revision 7, option 甲): the tether nodes
        touch the ground plane at ``ground_z_g_m`` (Makani's g frame, z down) with Makani's contactor template
        (``tether.variable_line.makani_ground_contact``); only with the line built here.

        ``floating`` (OF-0011 revision 6 addendum 73, opt-in, default None: nothing changes): a floating base of
        ``floating.turnkey_base``. Its state ``base_x`` is integrated with the others (``step``); every evaluation of
        ``rates`` first sets GS02's vessel frame from it and, after the tether, puts the base's state derivative in
        the diagnostics (``base_rate``). Its ``body``, when given, is the wing's rigid body (unless ``body`` is)."""
        from openfawe.flight.residual import load_rigid_body
        from openfawe.tether.bridle import load_bridle
        from openfawe.tether.lumped_line import TetherProperties, TetherSettings
        from openfawe.tether.variable_line import StagingSettings, VariableLengthTether, makani_ground_contact
        if tether_ground_contact and (line is not None or ground_z_g_m is None):
            raise ValueError("tether ground contact needs the line built here and ground_z_g_m")
        if not tether_ground_contact and ground_z_g_m is not None:
            raise ValueError("ground_z_g_m is used only with tether_ground_contact")
        from .gs02 import GroundStationV2
        raw = params.raw
        case = default_rigid_body_case()
        self.params = params
        if body is None and floating is not None and floating.body is not None:
            body = floating.body                    # addendum 73: e.g. wing serial 05 offshore
        self.body = load_rigid_body(case) if body is None else body
        self.bridle = load_bridle(case) if bridle is None else bridle
        t, ts = raw["tether"], raw["tether_sim"]
        if line is None:
            props = TetherProperties(t["length"], t["linear_density"], t["outer_diameter"], t["tensile_stiffness"],
                                     t["bending_stiffness"], t["section_drag_coeff"])
            settings = TetherSettings(num_nodes=ts["num_nodes"], longitudinal_damping_ratio=ts["longitudinal_damping_ratio_active"],
                                      bending_damping_ratio=ts["bending_damping_ratio_active"],
                                      air_density_kg_m3=raw["phys_sim"]["air_density"], gravity_m_s2=raw["phys"]["g"])
            staging = StagingSettings(ts["stiff_len_lim"], ts["longitudinal_damping_ratio_staged"])
            if tether_ground_contact:
                from openfawe.flight.makani_interface import G_TO_WORLD
                height = float((G_TO_WORLD@np.array([0.0, 0.0, float(ground_z_g_m)]))[2])     # z_g = 6.122 -> -6.122
                line = VariableLengthTether(props, settings, staging, ground=makani_ground_contact(props, settings, height))
            else:
                line = VariableLengthTether(props, settings, staging)
        self.line = line
        self.gs = GroundStationV2(params)
        self.gravity = np.array([0.0, 0.0, -raw["phys"]["g"]])
        self.contactors = raw["perch_contactors"]
        self.panel = raw["gs02_sim"]["panel"]
        self.inverse_inertia = np.linalg.inv(self.body.inertia_body_kg_m2)
        self.tether_on = True
        self.floating = floating
        if floating is not None:
            self.base_x = np.array(floating.x0, dtype=float)
            self.set_base(self.base_x)

    def set_base(self, x):
        """Addendum 73: GS02's vessel frame from the floating base's state ``x`` (the ground frame for None)."""
        from .gs02 import Frame
        frame = self.floating.frame(x)
        self.gs.set_vessel_frame(Frame.root() if frame is None else frame)

    # ----- conversions --------------------------------------------------------------------------------------------
    def initial_wing_state(self):
        """Makani's initial wing state (config/common/sim/wing_sim.py: ``perched_wing_pos_p`` moved 6 cm up and 6 cm
        towards the platform, at rest). As coded, the talons are then about 21 cm above the panel top (15 cm already at
        ``perched_wing_pos_p``), so the wing first falls onto the perch."""
        from openfawe.flight.makani_interface import G_TO_WORLD, MAKANI_BODY_TO_BODY
        from openfawe.flight.kinematics import Pose
        from openfawe.flight.rigid_body import state_from_pose
        init = self.params.raw["wing_sim_initial"]
        if any(init["Vb_0"]) or any(init["omega_0"]):
            raise ValueError("the initial wing velocity is expected to be zero")
        dcm_g2b = quat_to_dcm(init["q_0"])
        rotation = G_TO_WORLD@dcm_g2b.T@MAKANI_BODY_TO_BODY
        pose = Pose(0.0, G_TO_WORLD@np.asarray(init["Xg_0"], dtype=float), rotation, np.zeros(3), np.zeros(3))
        return state_from_pose(pose, self.body)

    def anchor_world(self, gsx):
        from openfawe.flight.makani_interface import G_TO_WORLD
        pos, vel = self.gs.anchor_pos_vel_ned(gsx)
        return G_TO_WORLD@pos, G_TO_WORLD@vel

    def makani_frame(self, pose):
        """Makani's wing frame of a plant pose (origin of body axes)."""
        from openfawe.flight.makani_interface import G_TO_WORLD, MAKANI_BODY_TO_BODY
        dcm_g2b = MAKANI_BODY_TO_BODY@pose.rotation.T@G_TO_WORLD
        omega_b = MAKANI_BODY_TO_BODY@(pose.rotation.T@pose.angular_velocity_rad_s)
        return makani_wing_frame(G_TO_WORLD@pose.position_m, G_TO_WORLD@pose.velocity_m_s, dcm_g2b, omega_b)

    # ----- loads ----------------------------------------------------------------------------------------------------
    def contact_loads(self, state, gsx):
        """Sum of the contactor forces (world) and their moment about the centre of mass (world); per-contactor forces."""
        from openfawe.flight.makani_interface import MAKANI_BODY_TO_BODY
        pose = state.pose(self.body)
        frame = self.makani_frame(pose)
        frames = self.gs.frames(gsx)
        force, moment, each = np.zeros(3), np.zeros(3), []
        held = self.held_branches                       # supplement 5: None except within a "held" event step
        for i, contactor in enumerate(self.contactors):
            f_b, pos_b, hit = contactor_force(contactor, frame, frames.platform, frames.panel, self.panel,
                                              None if held is None else held[i])
            f_world = pose.rotation@(MAKANI_BODY_TO_BODY@f_b)
            arm = pose.rotation@(MAKANI_BODY_TO_BODY@pos_b - self.body.centre_of_mass_body_m)
            force += f_world
            moment += cross3(arm, f_world)
            each.append((f_world, hit))
        return force, moment, each

    def contact_branches(self, state, gsx, depth_m=None):
        """Each contactor's ``contact_branch`` at a state, at the positions ``contactor_force`` uses (OF-0011
        revision 7 addendum 4 supplement 4; read only, for the event signature). ``depth_m`` (supplement 6; None: as
        before): each element is (branch, deep) instead, ``deep`` whether the contactor is at least ``depth_m`` into the
        panel (``|offset|`` of ``contact_force`` against ``contact_perch_panel``'s point; False without a collision)."""
        pose = state.pose(self.body)
        frame = self.makani_frame(pose)
        frames = self.gs.frames(gsx)
        branches = []
        for contactor in self.contactors:
            pos_surf = frames.panel.from_root("position", frame.child(np.asarray(contactor["pos"], dtype=float)).pos)
            branch = contact_branch(pos_surf, frames.platform, frames.panel, self.panel)
            if depth_m is None:
                branches.append(branch)
                continue
            hit, where = contact_perch_panel(pos_surf, frames.platform, frames.panel, self.panel)
            deep = bool(hit) and _norm3(np.asarray(pos_surf, dtype=float) - np.asarray(where, dtype=float)) >= depth_m
            branches.append((branch, deep))
        return tuple(branches)

    def tether_evaluation(self, state, nodes, speeds, gsx):
        pose = state.pose(self.body)
        start, start_vel = self.anchor_world(gsx)
        free_length = self.gs.free_length(float(gsx[self.gs.THETA]))
        idx = self.line.indices(free_length)
        last_x, last_v = self.line.last_node(idx, nodes, speeds, start, start_vel)
        end, end_vel = self.bridle.point(pose, last_x, last_v)
        dx, dv, out = self.line.derivatives(free_length, nodes, speeds, start, start_vel, end, end_vel)
        out.update(start_point=start, start_velocity=start_vel, end_point=end, end_velocity=end_vel, free_length=free_length)
        return dx, dv, out

    def rates(self, state, nodes, speeds, gsx):
        """All derivatives at one stage: (cm velocity, cm acceleration, body angular acceleration, node rates, gs rates,
        diagnostics)."""
        if self.floating is not None:                   # addendum 73: the vessel frame of this stage's base state
            self.set_base(self.base_x)
        force, moment, each = self.contact_loads(state, gsx)
        if self.tether_on:
            dx, dv, out = self.tether_evaluation(state, nodes, speeds, gsx)
            end_force = out["end"]
            force = force + end_force
            moment = moment + cross3(out["end_point"] - state.position_cm_m, end_force)
        else:
            dx, dv, out = np.zeros_like(nodes), np.zeros_like(speeds), None
        omega = state.angular_velocity_body_rad_s
        inertia = self.body.inertia_body_kg_m2
        acceleration = force/self.body.mass_kg + self.gravity
        spin = self.inverse_inertia@(state.rotation.T@moment - cross3(omega, inertia@omega))
        diag = dict(tether=out, contacts=each, contact_force=force)
        if self.floating is not None:
            diag["base_rate"] = self.floating.rates(state.time_s, self.base_x, out)
        return state.velocity_cm_m_s, acceleration, spin, dx, dv, self.gs.derivatives(gsx), diag

    def step(self, state, nodes, speeds, h):
        """One RK4 step of all states (the attitude by rotation-vector increments, as ``flight.rigid_body``)."""
        if self.floating is not None:
            return self._step_floating(state, nodes, speeds, h)
        from openfawe.flight.kinematics import rotation_from_vector
        from openfawe.flight.rigid_body import RigidState, _theta_rate
        t0, p0, r0, v0, w0 = (state.time_s, state.position_cm_m, state.rotation, state.velocity_cm_m_s,
                              state.angular_velocity_body_rad_s)
        g0 = self.gs.x.copy()
        zero = np.zeros(3)

        def stage(time, dp, dv, theta, dw, dx, du, dg):
            rotation = r0 if theta is zero else r0@rotation_from_vector(theta)
            omega = w0 + dw
            s = RigidState(time, p0 + dp, rotation, v0 + dv, omega)
            velocity, acceleration, spin, node_x, node_v, gs_rate, _ = self.rates(s, nodes + dx, speeds + du, g0 + dg)
            return velocity, acceleration, _theta_rate(theta, omega), spin, node_x, node_v, gs_rate

        k1 = stage(t0, zero, zero, zero, zero, 0.0, 0.0, 0.0)
        k2 = stage(t0 + 0.5*h, *(0.5*h*k for k in k1))
        k3 = stage(t0 + 0.5*h, *(0.5*h*k for k in k2))
        k4 = stage(t0 + h, *(h*k for k in k3))
        dp, dv, theta, dw, dx, du, dg = (h/6.0*(a + 2.0*b + 2.0*c + d) for a, b, c, d in zip(k1, k2, k3, k4))
        new = RigidState(t0 + h, p0 + dp, r0@rotation_from_vector(theta), v0 + dv, w0 + dw)
        nodes, speeds = nodes + dx, speeds + du
        self.gs.x = g0 + dg
        if not (np.isfinite(nodes).all() and np.isfinite(speeds).all() and np.isfinite(self.gs.x).all()):
            raise FloatingPointError("a state left the finite range")
        return new, nodes, speeds

    def _step_floating(self, state, nodes, speeds, h):
        """``step`` with the floating base's states in the same RK4 (addendum 73); the other states' arithmetic is
        ``step``'s, term for term."""
        from openfawe.flight.kinematics import rotation_from_vector
        from openfawe.flight.rigid_body import RigidState, _theta_rate
        t0, p0, r0, v0, w0 = (state.time_s, state.position_cm_m, state.rotation, state.velocity_cm_m_s,
                              state.angular_velocity_body_rad_s)
        g0 = self.gs.x.copy()
        b0 = self.base_x.copy()
        zero = np.zeros(3)

        def stage(time, dp, dv, theta, dw, dx, du, dg, db):
            rotation = r0 if theta is zero else r0@rotation_from_vector(theta)
            omega = w0 + dw
            s = RigidState(time, p0 + dp, rotation, v0 + dv, omega)
            self.base_x = b0 + db
            velocity, acceleration, spin, node_x, node_v, gs_rate, diag = self.rates(s, nodes + dx, speeds + du, g0 + dg)
            return velocity, acceleration, _theta_rate(theta, omega), spin, node_x, node_v, gs_rate, diag["base_rate"]

        k1 = stage(t0, zero, zero, zero, zero, 0.0, 0.0, 0.0, 0.0)
        k2 = stage(t0 + 0.5*h, *(0.5*h*k for k in k1))
        k3 = stage(t0 + 0.5*h, *(0.5*h*k for k in k2))
        k4 = stage(t0 + h, *(h*k for k in k3))
        dp, dv, theta, dw, dx, du, dg, db = (h/6.0*(a + 2.0*b + 2.0*c + d) for a, b, c, d in zip(k1, k2, k3, k4))
        new = RigidState(t0 + h, p0 + dp, r0@rotation_from_vector(theta), v0 + dv, w0 + dw)
        nodes, speeds = nodes + dx, speeds + du
        self.gs.x = g0 + dg
        self.base_x = b0 + db
        self.set_base(self.base_x)                      # the frame of the step's end, for what follows the step
        if not (np.isfinite(nodes).all() and np.isfinite(speeds).all() and np.isfinite(self.gs.x).all()
                and np.isfinite(self.base_x).all()):
            raise FloatingPointError("a state left the finite range")
        return new, nodes, speeds


def perched_initialisation(params, system=None):
    """InitGs02Tether (full_system.cc:866-914): bisection on the drum angle between MinDrumAngle and 0 for a tether
    end tension within [L rho g/100, L rho g/2], with the wing at its initial pose; then the ground station and the
    tether are left initialised at that drum angle. Returns the result and the system."""
    system = PerchedSystem(params) if system is None else system
    raw = params.raw
    L, rho, g = raw["tether"]["length"], raw["tether"]["linear_density"], raw["phys"]["g"]
    tension_min, tension_max = L*rho*g/100.0, L*rho*g/2.0
    state = system.initial_wing_state()
    pose = state.pose(system.body)
    gs, line = system.gs, system.line
    azimuth = raw["gs02_sim"]["initial_platform_azi"]
    low, high = gs.min_drum_angle(), 0.0
    nodes = speeds = None
    iterations = 0
    while True:
        if not high - low > 1e-8:
            raise RuntimeError("Failed to converge.")                                          # CHECK_GT
        mid = (low + high)/2.0
        gs.init(azimuth, mid)
        start, start_vel = system.anchor_world(gs.x)
        nodes, speeds, end, end_vel, start_force = line.init(start, start_vel, lambda x, v: system.bridle.point(pose, x, v),
                                                             gs.free_length(), nodes, speeds)
        iterations += 1
        tension = float(np.linalg.norm(start_force))
        if tension < tension_min:
            high = mid
        elif tension > tension_max:
            low = mid
        else:
            break
    return dict(drum_angle_rad=mid, free_length_m=gs.free_length(), tension_N=tension, iterations=iterations,
                tension_window_N=[tension_min, tension_max], platform_azimuth_rad=azimuth,
                nodes=nodes, node_velocities=speeds, wing_state=state, indices=line.indices(gs.free_length())), system


def makani_wind_g(params):
    """The simulator's mean wind (environment.cc:90, 411-412; wind_frame.c:24-29): speed along the mean-wind x axis,
    which points opposite to ``wind_direction`` (the direction the wind comes from), in g (NED) coordinates."""
    phys = params.raw["phys_sim"]
    d = float(phys["wind_direction"])
    return float(phys["wind_speed"])*np.array([-math.cos(d), -math.sin(d), 0.0])


def kite_estimate(api, system, state, acceleration_cm, spin_body, tether_force_world, wind_g, *, sensor_wind_g=None):
    """The kite's fields of a ``StateEstimate`` from the plant's truth, as ``flight.closed_loop`` fills them; the
    wind is the simulator's (given in g). ``acceleration_cm`` and ``spin_body`` are the plant's accelerations at
    ``state`` (part 2: no aerodynamics; part 3, ``ground.turnkey``: with the loads held over the period).

    ``sensor_wind_g`` (plan 3.9b, default None): the wind at the ground wind sensor for the ground wind estimate
    ``wind_g`` (Makani's EstimatorWindStep, estimator.c:196-198); ``wind_g`` is then the wind at the kite, for the
    apparent wind and the wind aloft (the Pitot solution of EstimatorWindAloftStep, estimator_wind.c:162-199). Without it
    both estimates carry ``wind_g``."""
    import ctypes
    from openfawe.flight.makani_interface import G_TO_WORLD, makani_truth, origin_acceleration, wind_direction
    origin_acc = origin_acceleration(system.body, state, acceleration_cm, spin_body)
    truth = makani_truth(system.body, state, G_TO_WORLD@wind_g, acceleration_origin_m_s2=origin_acc,
                         tether_force_inertial_N=tether_force_world)
    vec = lambda v: api.Vec3(float(v[0]), float(v[1]), float(v[2]))        # noqa: E731
    est = api.StateEstimate()
    for i in range(3):
        for j in range(3):
            est.dcm_g2b.d[i][j] = float(truth.dcm_g2b[i][j])
    est.pqr, est.pqr_f = vec(truth.pqr), vec(truth.pqr)
    est.Xg, est.Vg, est.Vg_f, est.Ag = vec(truth.Xg), vec(truth.Vg), vec(truth.Vg), vec(truth.Ag)
    vb = truth.dcm_g2b@truth.Vg
    gravity_g = np.array([0.0, 0.0, float(system.params.raw["phys"]["g"])])
    # Ab_f is the kinematic acceleration in body axes, Makani's definition (CalcAlternativeCoordinates,
    # estimator_nav.c:107-113: Ab_f = dcm_g2b g_g + acc_b_f, the accelerometers' specific force plus gravity), as
    # flight.closed_loop fills it; acc_norm_f is the norm of the specific force (estimator_nav.c:87-88). Until the
    # full-cycle contract's revision 6 (K4) this field held the specific force dcm_g2b (Ag - g_g); the only reader in
    # the controller is crosswind_path.c, so part 2's perched check G7 did not depend on it.
    est.Vb, est.Vb_f, est.Ab_f = vec(vb), vec(vb), vec(truth.dcm_g2b@truth.Ag)
    est.acc_norm_f = float(np.linalg.norm(truth.Ag - gravity_g))
    est.gps_active = True
    est.rho = float(api.GetSystemParams().contents.phys.rho)
    wind_b = vec(truth.apparent_wind_b)
    est.apparent_wind.solution_type = api.kApparentWindSolutionTypePitot
    est.apparent_wind.vector = wind_b
    api.ApparentWindCartToSph(ctypes.byref(wind_b), ctypes.byref(est.apparent_wind.sph))
    api.ApparentWindCartToSph(ctypes.byref(wind_b), ctypes.byref(est.apparent_wind.sph_f))
    force_b = vec(truth.tether_force_b)
    est.tether_force_b.valid = True
    est.tether_force_b.vector = est.tether_force_b.vector_f = force_b
    api.TetherForceCartToSph(ctypes.byref(force_b), ctypes.byref(est.tether_force_b.sph))
    est.tether_force_b.tension_f = est.tether_force_b.sph.tension
    offset = float(api.GetSystemParams().contents.ground_station.azi_ref_offset)
    # The ground wind at the sensor goes through the same two transformations as the kite's (makani_truth's wind_g).
    ground = truth.wind_g if sensor_wind_g is None else G_TO_WORLD@(G_TO_WORLD@np.asarray(sensor_wind_g, dtype=float).reshape(3))
    for target, wind in ((est.wind_g, ground), (est.wind_aloft_g, truth.wind_g)):
        target.valid = True
        target.vector, target.vector_f, target.vector_f_slow = (vec(wind) for _ in range(3))
        target.dir_f = target.dir_f_playbook = wind_direction(wind, offset)
        target.speed_f = target.speed_f_playbook = float(np.linalg.norm(wind))
    return est


def run_perched_turnkey(params, api, *, duration_s=11.0, step_s=1e-3, switch_time_s=10.1):
    """Criterion G7: the perched wing, the ground station, the variable tether and part 1's full controller.

    The joystick is TurnKey's first seconds (config/common/sim/joystick_sim.py:110-131): switch up, throttle 0; the
    switch goes to the middle at 10.1 s; the throttle stays 0 (TurnKey's 0.81 at 11.0 s is not applied). One
    controller period (0.01 s) follows the simulator's step (kite_sim.cc:173-204 with the connections of
    full_system.cc): the sensors and the ground station's derived inputs (levelwind engagement, tether force) from the
    state at the start of the period, then the ground station's discrete step, then the controller's command of the
    previous period reaches the ground station (it arrives after the integration of that period, so the discrete
    step of a period sees the command of two periods before), then the controller runs on this period's sensors,
    then ten RK4 steps. Returns a record of every period.
    """
    import time
    from openfawe.flight.makani_full import FullController, Joystick, gs02_sensors
    from openfawe.flight.makani_interface import G_TO_WORLD
    from .gs02 import MODE_REEL, sensor_readings
    init, system = perched_initialisation(params)
    gs, line = system.gs, system.line
    state, nodes, speeds = init["wing_state"], init["nodes"], init["node_velocities"]
    fc = FullController(api)
    cal = fc.system.winch.velocity_cmd_cal
    if (cal.scale, cal.bias) != (1.0, 0.0):
        raise ValueError("the winch velocity command calibration is expected to be the identity")
    wind_g = makani_wind_g(params)
    per = int(round(0.01/step_s))
    if abs(per*step_s - 0.01) > 1e-15:
        raise ValueError("the step must divide the controller period")
    pending, records = None, []
    wall = time.time()
    for k in range(int(round(duration_s/0.01))):
        t = k*0.01
        velocity, acceleration, spin, _, _, _, diag = system.rates(state, nodes, speeds, gs.x)
        out = diag["tether"]
        direction_ned = G_TO_WORLD@line.start_direction(out["indices"], nodes, out["start_point"], out["end_point"])
        ele, azi, yoke, termination = gs.tether_dir_to_angles(direction_ned)
        gs.set_levelwind_engaged(ele)
        gs.set_tether_force(G_TO_WORLD@out["start"])
        readings = sensor_readings(gs, ele, yoke, termination)
        gs.discrete_update(t)
        if pending is not None:
            gs.set_from_command(**pending)
        sensors = gs02_sensors(api, fc.system, drum_angle_rad=readings["drum_angle_rad"],
                               platform_azimuth_rad=readings["platform_azimuth_rad"], gs_mode=readings["gs_mode"],
                               transform_stage=readings["transform_stage"], proximity=readings["proximity"],
                               detwist_angle_rad=readings["detwist_angle_rad"],
                               levelwind_elevation_rad=readings["levelwind_elevation_rad"],
                               gsg_yoke_rad=readings["gsg_yoke_rad"], gsg_termination_rad=readings["gsg_termination_rad"],
                               platform_azimuth_rate_rad_s=gs.platform_azi_vel)
        joystick = Joystick(switch_position=api.kJoystickSwitchPositionUp if t < switch_time_s
                            else api.kJoystickSwitchPositionMiddle, throttle=0.0)
        est = kite_estimate(api, system, state, acceleration, spin, out["end"], wind_g)
        est, _, _ = fc.ground_estimate(est, sensors, joystick)
        ran, output = fc.step(est)
        pending = dict(gs_azi_target=float(output.gs_azi_cmd.target), gs_azi_dead_zone=float(output.gs_azi_cmd.dead_zone),
                       winch_velocity=float(output.winch_vel_cmd), detwist_position=float(output.detwist_cmd),
                       gs_mode_request=int(output.gs_mode_request), gs_unpause_transform=bool(output.gs_unpause_transform))
        records.append(dict(t=t, ran=bool(ran), flight_mode=fc.flight_mode, gs_mode=gs.mode.val, mode_request=pending["gs_mode_request"],
                            winch_velocity_cmd=pending["winch_velocity"], drum_vel_cmd=gs.reel_controller.drum_vel_cmd.val,
                            drum_angle=gs.drum_angle, drum_omega=gs.drum_omega, platform_azi=gs.platform_azi,
                            levelwind_engaged=gs.levelwind_engaged, tether_elevation=ele, tension=float(np.linalg.norm(out["start"])),
                            proximity=bool(gs.prox_sensor_active.val), free_length=out["free_length"],
                            position_cm=state.position_cm_m.copy(), rotation=state.rotation.copy(),
                            contacts=[bool(c[1]) for c in diag["contacts"]], payout=float(est.winch.payout),
                            wall_s=time.time() - wall))
        for _ in range(per):
            state, nodes, speeds = system.step(state, nodes, speeds, step_s)
    return dict(init={k: v for k, v in init.items() if k not in ("nodes", "node_velocities", "wing_state")},
                records=records, wall_s=time.time() - wall)


def drift(records, reference_time_s):
    """Largest centre-of-mass displacement and attitude change (deg) after ``reference_time_s``."""
    ref = next(r for r in records if r["t"] >= reference_time_s - 1e-9)
    later = [r for r in records if r["t"] >= ref["t"]]
    displacement = max(float(np.linalg.norm(r["position_cm"] - ref["position_cm"])) for r in later)
    attitude = max(math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(ref["rotation"].T@r["rotation"]) - 1.0)/2.0))))
                   for r in later)
    return displacement, attitude
