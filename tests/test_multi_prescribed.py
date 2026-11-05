"""Several bodies on prescribed motions in one wake (OF-0009 contract, case C3: identity and superposition checks)."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero.aircraft import PanelPolars, WingAssembly
from openfawe.aero.methods import AeroBackend
from openfawe.aero.vortex import TreeSettings
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing
from openfawe.flight.kinematics import circular_motion, uniform_translation
from openfawe.flight.multi_prescribed import PrescribedBody, ring_settings, run_prescribed_multi, structural_velocity_field
from openfawe.flight.prescribed import PrescribedRotors, PrescribedTether, run_prescribed
from openfawe.rotor.actuator_disc import ActuatorDisc
from openfawe.rotor.ring_wake import RingWake
from openfawe.rotor.table import RotorTable
from openfawe.tether.bridle import Bridle
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings

NEWTON = SolveSettings(solver="newton", max_iterations=60)
RADIUS, SPEED, WIND, ALPHA, DT = 130.0, 47.0, 9.0, 3.0, 0.02


def backend(steps, dt=DT, **options):
    return AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=dt, max_rows=steps, core_radius_m=0.15, solve=NEWTON, **options))


def discs():
    """Two discs ahead of the wing, as in ``test_prescribed_motion``: a small table in the Makani layout."""
    omegas, speeds = np.linspace(40., 260., 12), np.linspace(0.1, 90., 9)
    w, v = np.meshgrid(omegas, speeds, indexing="ij")
    table = RotorTable("synthetic", 2.3, omegas, speeds, 0.18-0.002*v-1e-5*w*v/60., -0.08+0.0015*v)
    axis = [-np.cos(np.radians(3.)), 0., -np.sin(np.radians(3.))]
    return (ActuatorDisc("upper", [-1.9, 1.2, 1.2], axis, 1, table), ActuatorDisc("lower", [-1.6, -3.6, -1.6], axis, -1, table))


class SingleBodyIdentityTests(unittest.TestCase):
    def test_one_body_reproduces_run_prescribed_bit_for_bit(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 5
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        single = run_prescribed(backend(steps), wing, polar, motion, [WIND, 0., 0.], keep_states=(steps,))
        multi = run_prescribed_multi(backend(steps), [PrescribedBody("k", wing, polar, motion)], [WIND, 0., 0.], keep_states=(steps,))
        self.assertEqual(multi.status, "complete")
        self.assertEqual(len(multi.records), steps+1)
        for a, b in zip(single.records, multi.records):
            mine = b["bodies"]["k"]
            for name in ("force_inertial_N", "moment_origin_inertial_Nm", "force_body_N", "moment_origin_body_Nm"):
                self.assertEqual(a[name], mine[name], name)
            self.assertEqual(a["airspeed_m_s"], mine["airspeed_m_s"])
            self.assertEqual(a["alpha_deg"], mine["alpha_deg"])
            self.assertEqual(a["aero_power_W"], mine["aero_power_W"])
            self.assertEqual(a["iterations"], b["iterations"])
        assert_allclose(single.states[steps].solution.circulation, multi.states[steps].solution.circulation, rtol=0., atol=0.)
        self.assertEqual(multi.panel_slices["k"], slice(0, wing.panels))

    def test_an_assembly_body_keeps_its_surface_names_prefixed(self):
        wing = Wing.ellipse(6, span=10.0, root_chord=1.0)
        tail = Wing.ellipse(4, span=4.0, root_chord=0.6).transformed(np.eye(3), np.array([5., 0., 0.5]))
        assembly = WingAssembly((wing, tail), ("wing", "tail"))
        polar = PanelPolars((SectionPolar.linear(limit_deg=60.),)*assembly.panels)
        steps = 3
        motion = uniform_translation(np.arange(steps+1)*DT, [-30., 0., 0.])
        run = run_prescribed_multi(backend(steps), [PrescribedBody("a", assembly, polar, motion)], np.zeros(3))
        self.assertEqual(run.status, "complete")
        self.assertEqual(run.panel_slices["a"], slice(0, assembly.panels))
        reference = run_prescribed(backend(steps), assembly, polar, motion, np.zeros(3))
        for a, b in zip(reference.records, run.records):
            self.assertEqual(a["force_inertial_N"], b["bodies"]["a"]["force_inertial_N"])


class SuperpositionTests(unittest.TestCase):
    def test_two_bodies_far_apart_reproduce_two_single_runs(self):
        """S1 of the contract: at 20 km separation the joint solve equals the two single runs within the solver tolerance."""
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 4
        times = np.arange(steps+1)*DT
        near = circular_motion(times, RADIUS, SPEED, WIND, ALPHA)
        far = circular_motion(times, RADIUS, SPEED, WIND, ALPHA, phase_rad=1.0)
        shift = np.array([20000., 0., 0.])
        far = type(far)(far.time_s, far.position_m+shift, far.rotation, far.velocity_m_s, far.angular_velocity_rad_s,
                        far.acceleration_m_s2, far.angular_acceleration_rad_s2, description="shifted")
        bodies = [PrescribedBody("near", wing, polar, near), PrescribedBody("far", wing, polar, far)]
        joint = run_prescribed_multi(backend(steps), bodies, [WIND, 0., 0.])
        self.assertEqual(joint.status, "complete")
        alone = {name: run_prescribed(backend(steps), wing, polar, motion, [WIND, 0., 0.]) for name, motion in (("near", near), ("far", far))}
        for name in ("near", "far"):
            for a, b in zip(alone[name].records, joint.records):
                for key in ("force_inertial_N", "moment_origin_inertial_Nm"):
                    assert_allclose(b["bodies"][name][key], a[key], rtol=1e-9, atol=1e-9*np.linalg.norm(a[key]), err_msg=f"{name} {key}")

    def test_structural_velocity_is_that_of_the_nearest_body_and_close_bodies_are_refused(self):
        times = np.arange(3)*DT
        a = uniform_translation(times, [-10., 0., 0.], origin=(0., 0., 0.))
        b = uniform_translation(times, [0., 5., 0.], origin=(300., 0., 0.))
        wing, polar = Wing.ellipse(4), SectionPolar.linear()
        field = structural_velocity_field([PrescribedBody("a", wing, polar, a), PrescribedBody("b", wing, polar, b)], 50.)
        out = field(np.array([[1., 0., 0.], [299., 0., 0.]]), 0.)
        assert_allclose(out, [[-10., 0., 0.], [0., 5., 0.]])
        close = uniform_translation(times, [0., 5., 0.], origin=(20., 0., 0.))
        field = structural_velocity_field([PrescribedBody("a", wing, polar, a), PrescribedBody("b", wing, polar, close)], 50.)
        with self.assertRaises(ValueError):
            field(np.zeros((1, 3)), 0.)
        with self.assertRaises(ValueError):
            run_prescribed_multi(backend(2), [PrescribedBody("a", wing, polar, a), PrescribedBody("b", wing, polar, close)], np.zeros(3))

    def test_validation(self):
        wing, polar = Wing.ellipse(4), SectionPolar.linear()
        motion = uniform_translation(np.arange(3)*DT, [-10., 0., 0.])
        with self.assertRaises(ValueError):
            run_prescribed_multi(backend(2), [], np.zeros(3))
        with self.assertRaises(ValueError):
            run_prescribed_multi(backend(2), [PrescribedBody("a", wing, polar, motion), PrescribedBody("a", wing, polar, motion)], np.zeros(3))
        other = uniform_translation(np.arange(4)*DT, [-10., 0., 0.], origin=(500., 0., 0.))
        with self.assertRaises(ValueError):
            run_prescribed_multi(backend(2), [PrescribedBody("a", wing, polar, motion), PrescribedBody("b", wing, polar, other)], np.zeros(3))
        with self.assertRaises(ValueError):
            run_prescribed_multi(AeroBackend("vsm", wake=WakeConfig(dt_s=DT, max_rows=2, core_radius_m=0.15, solve=NEWTON)),
                                 [PrescribedBody("a", wing, polar, motion)], np.zeros(3))


class RotorTests(unittest.TestCase):
    def test_one_body_with_rotors_reproduces_run_prescribed_bit_for_bit(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 4
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3)
        single = run_prescribed(backend(steps), wing, polar, motion, [WIND, 0., 0.], rotors=rotors, keep_states=(steps,))
        multi = run_prescribed_multi(backend(steps), [PrescribedBody("k", wing, polar, motion, rotors=rotors)], [WIND, 0., 0.],
                                     keep_states=(steps,))
        self.assertEqual((single.status, multi.status), ("complete", "complete"), multi.error)
        for a, b in zip(single.records, multi.records):
            mine = b["bodies"]["k"]
            for name in ("force_inertial_N", "moment_origin_inertial_Nm", "rotor_force_inertial_N", "rotor_moment_origin_body_Nm",
                         "rotor_power_on_body_W", "rotor_shaft_power_W", "rotor_thrust_N", "rotor_axial_freestream_m_s",
                         "ring_nodes", "ring_nodes_within_surface_core", "ring_surface_core_m"):
                self.assertEqual(a[name], mine[name], name)
        assert_allclose(single.states[steps].rings.nodes, multi.states[steps].rings.nodes, rtol=0., atol=0.)
        self.assertEqual(multi.states[steps].rings.count, (steps+1)*2*8)

    def test_two_bodies_with_rotors_share_one_ring_set_and_reproduce_two_single_runs_far_apart(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 3
        times = np.arange(steps+1)*DT
        near = circular_motion(times, RADIUS, SPEED, WIND, ALPHA)
        far = circular_motion(times, RADIUS, SPEED, WIND, ALPHA, phase_rad=1.0)
        far = type(far)(far.time_s, far.position_m+np.array([20000., 0., 0.]), far.rotation, far.velocity_m_s, far.angular_velocity_rad_s,
                        far.acceleration_m_s2, far.angular_acceleration_rad_s2, description="shifted")
        rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3)
        bodies = [PrescribedBody("near", wing, polar, near, rotors=rotors), PrescribedBody("far", wing, polar, far, rotors=rotors)]
        joint = run_prescribed_multi(backend(steps), bodies, [WIND, 0., 0.], keep_states=(steps,))
        self.assertEqual(joint.status, "complete", joint.error)
        self.assertEqual(joint.states[steps].rings.count, (steps+1)*4*8)          # one ring set: both bodies' rings
        alone = {name: run_prescribed(backend(steps), wing, polar, motion, [WIND, 0., 0.], rotors=rotors)
                 for name, motion in (("near", near), ("far", far))}
        for name in ("near", "far"):
            for a, b in zip(alone[name].records, joint.records):
                for key in ("force_inertial_N", "rotor_force_inertial_N", "rotor_shaft_power_W"):
                    assert_allclose(b["bodies"][name][key], a[key], rtol=1e-9, atol=1e-9*np.linalg.norm(a[key]), err_msg=f"{name} {key}")
                # crossings are counted per body at its own load points; the joint count covers both bodies
                self.assertEqual(b["bodies"][name]["ring_nodes_within_surface_core"], a["ring_nodes_within_surface_core"])
                self.assertGreaterEqual(b["bodies"][name]["ring_nodes_within_surface_core_all_bodies"], b["bodies"][name]["ring_nodes_within_surface_core"])
        # unequal ring settings are refused: one ring set carries one surface core
        other = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.6)
        with self.assertRaises(ValueError):
            ring_settings([PrescribedBody("near", wing, polar, near, rotors=rotors), PrescribedBody("far", wing, polar, far, rotors=other)])
        self.assertIsNone(ring_settings([PrescribedBody("near", wing, polar, near)]))


class FrozenRingTests(unittest.TestCase):
    """``PrescribedRotors.free_convection_ring_steps``: rings older than the limit drift with the ambient wind, nothing is deleted."""

    def test_limit_beyond_the_run_changes_nothing_and_old_rings_drift_with_the_wind(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 5
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind = np.array([WIND, 0., 0.])
        plain = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3)
        wide = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3, free_convection_ring_steps=steps+5)
        narrow = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3, free_convection_ring_steps=2)
        runs = {name: run_prescribed(backend(steps), wing, polar, motion, wind, rotors=rotors, keep_states=tuple(range(steps+1)))
                for name, rotors in (("plain", plain), ("wide", wide), ("narrow", narrow))}
        for run in runs.values():
            self.assertEqual(run.status, "complete", run.error)
        assert_allclose(runs["wide"].states[steps].rings.nodes, runs["plain"].states[steps].rings.nodes, rtol=0., atol=0.)
        self.assertEqual(runs["wide"].records[-1]["ring_free_convection_steps"], steps+5)
        # narrow (K = 2): the state of step k holds the rings of steps 0..k, newest first, each of ``per_step`` nodes. During the
        # advance from step k-1 to step k (state time t_{k-1}) the rings of steps k-1 and k-2 are free and those of steps <= k-3
        # are frozen: they move by exactly wind * dt, whereas the ring of step k-1 is induced as well.
        per_step = 2*8
        for k in range(3, steps+1):
            before, after = runs["narrow"].states[k-1].rings, runs["narrow"].states[k].rings
            self.assertEqual(before.count+per_step, after.count)
            assert_allclose(after.nodes[3*per_step:], before.nodes[2*per_step:]+wind*DT, rtol=0., atol=1e-9, err_msg=f"step {k}")
            moving = after.nodes[per_step:2*per_step]-before.nodes[:per_step]
            self.assertGreater(np.max(np.abs(moving-wind*DT)), 1e-6)
        self.assertEqual(runs["narrow"].states[steps].rings.count, runs["plain"].states[steps].rings.count)   # nothing deleted
        self.assertEqual(runs["narrow"].records[-1]["ring_free_convection_steps"], 2)
        # the loads differ from the complete convection only through the positions of the far rings: a finite, small difference
        a, b = runs["narrow"].records[-1], runs["plain"].records[-1]
        gap = np.linalg.norm(np.subtract(a["force_inertial_N"], b["force_inertial_N"]))/np.linalg.norm(b["force_inertial_N"])
        self.assertTrue(np.isfinite(gap) and gap < 5e-2, gap)

    def test_frozen_rows_and_frozen_rings_together(self):
        """The setting of the runs: WakeConfig.free_convection_rows and the rings' own limit at the same time."""
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 5
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind = np.array([WIND, 0., 0.])
        rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3, free_convection_ring_steps=2)
        run = run_prescribed(backend(steps, free_convection_rows=2), wing, polar, motion, wind, rotors=rotors,
                             keep_states=tuple(range(steps+1)))
        self.assertEqual(run.status, "complete", run.error)
        per_step = 2*8
        for k in range(3, steps+1):
            before, after = run.states[k-1], run.states[k]
            assert_allclose(after.rings.nodes[3*per_step:], before.rings.nodes[2*per_step:]+wind*DT, rtol=0., atol=1e-9, err_msg=f"rings {k}")
            # rows: free_convection_rows counts the trailing-edge row too, so the stored rows older than one free row are frozen
            old_before, old_after = before.free_rows[1:], after.free_rows[2:]
            assert_allclose(old_after, old_before+wind*DT, rtol=0., atol=1e-9, err_msg=f"rows {k}")

    def test_near_surface_count_matches_the_full_test_on_the_boundary(self):
        rings = RingWake(np.array([[0.3, 0., 0.], [0., 0.3, 0.], [1e5+0.3, 0., 0.], [0.1, 0.1, 0.1]]), np.array([[0, 1], [1, 3], [2, 3], [3, 0]]),
                         np.ones(4), np.zeros(4), np.zeros(4, dtype=int))
        points = np.array([[0., 0., 0.], [1e5, 0., 0.]])
        full = int(np.count_nonzero(np.any(np.sum((rings.nodes[:, None, :]-points[None])**2, axis=2) <= 0.3**2, axis=1)))
        self.assertEqual(rings.near_surface_count(points, 0.3), full)

    def test_ring_wake_free_node_count_and_validation(self):
        # 0 is allowed since 2026-10-22 18:55 BST (model A': every ring drifts with the ambient flow from its release)
        with self.assertRaises(ValueError):
            RingWake(free_convection_steps=-1)
        with self.assertRaises(ValueError):
            PrescribedRotors(discs(), 120., free_convection_ring_steps=-1)
        self.assertEqual(RingWake(free_convection_steps=0).free_convection_steps, 0)
        with self.assertRaises(ValueError):
            PrescribedRotors(discs(), 120., retained_ring_steps=3, free_convection_ring_steps=4)
        rings = RingWake(free_convection_steps=2)
        self.assertIsNone(RingWake().free_node_count(1., 0.1))
        self.assertEqual(rings.free_node_count(1., 0.1), 0)
        # three rings released at 0.3, 0.2, 0.1 (newest first), 4 nodes each
        nodes = np.zeros((12, 3)); edges = np.array([[i, (i+1) % 4+4*(i//4)] for i in range(12)])
        times = np.repeat([0.3, 0.2, 0.1], 4)
        stored = RingWake(nodes, edges, np.ones(12), times, np.zeros(12, dtype=int), free_convection_steps=2)
        self.assertEqual(stored.free_node_count(0.3, 0.1), 8)
        self.assertEqual(stored.free_node_count(0.4, 0.1), 4)
        self.assertEqual(stored.free_node_count(0.6, 0.1), 0)
        unordered = RingWake(nodes, edges, np.ones(12), np.repeat([0.1, 0.3, 0.2], 4), np.zeros(12, dtype=int), free_convection_steps=2)
        with self.assertRaises(ValueError):
            unordered.free_node_count(0.3, 0.1)


class RingCoreTests(unittest.TestCase):
    """``PrescribedRotors.ring_core_m`` / ``RingWake.core_m``: the core of the ring filaments away from the lifting surfaces."""

    def test_core_equal_to_the_wake_core_is_bit_identical_and_a_larger_core_changes_the_rings(self):
        from openfawe.aero.wake import _edge_radii_extended, extended_network
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 4
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        wind = np.array([WIND, 0., 0.])
        runs = {}
        for name, core in (("default", None), ("same", 0.15), ("larger", 0.6)):
            rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3, ring_core_m=core)
            runs[name] = run_prescribed(backend(steps), wing, polar, motion, wind, rotors=rotors, keep_states=tuple(range(steps+1)))
            self.assertEqual(runs[name].status, "complete", runs[name].error)
        for a, b in zip(runs["default"].records, runs["same"].records):
            for key in ("force_inertial_N", "moment_origin_inertial_Nm", "rotor_force_inertial_N", "rotor_shaft_power_W",
                        "rotor_axial_freestream_m_s", "ring_nodes_within_surface_core"):
                self.assertEqual(a[key], b[key], key)
            self.assertNotIn("ring_core_m", a)
            self.assertEqual(b["ring_core_m"], 0.15)
        assert_allclose(runs["same"].states[steps].rings.nodes, runs["default"].states[steps].rings.nodes, rtol=0., atol=0.)
        assert_allclose(runs["same"].states[steps].free_rows, runs["default"].states[steps].free_rows, rtol=0., atol=0.)
        # a larger core moves the rings differently (their self-induction and the inflow at the hubs change) ...
        self.assertGreater(np.max(np.abs(runs["larger"].states[steps].rings.nodes-runs["default"].states[steps].rings.nodes)), 1e-6)
        # ... and the ring edges carry it in the convection, the wing edges keep the wake core
        state = runs["larger"].states[steps]
        self.assertEqual(state.rings.core_m, 0.6)
        nodes, edges, gamma, ring_first = extended_network(state)
        radii = np.broadcast_to(_edge_radii_extended(state, edges, ring_first, backend(steps).wake), (len(edges),))
        ring = np.all(edges >= ring_first, axis=1)
        assert_allclose(radii[ring], 0.6, rtol=0., atol=0.)
        assert_allclose(radii[~ring], 0.15, rtol=0., atol=0.)

    def test_core_is_carried_by_every_ring_operation_and_validated(self):
        nodes = np.zeros((8, 3)); edges = np.array([[i, (i+1) % 4+4*(i//4)] for i in range(8)])
        rings = RingWake(nodes, edges, np.ones(8), np.repeat([0.2, 0.1], 4), np.zeros(8, dtype=int), core_m=0.4)
        self.assertEqual(rings.moved(nodes+1.).core_m, 0.4)
        self.assertEqual(rings.released_since(0.15).core_m, 0.4)
        for bad in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                RingWake(core_m=bad)
            with self.assertRaises(ValueError):
                PrescribedRotors(discs(), 120., ring_core_m=bad)
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        motion = circular_motion(np.arange(3)*DT, RADIUS, SPEED, WIND, ALPHA)
        bodies = [PrescribedBody("a", wing, polar, motion, rotors=PrescribedRotors(discs(), 120., ring_core_m=0.3)),
                  PrescribedBody("b", wing, polar, motion, rotors=PrescribedRotors(discs(), 120., ring_core_m=0.6))]
        with self.assertRaises(ValueError):
            ring_settings(bodies)


class RingCoreRoutingTests(unittest.TestCase):
    """Review RC-04 of commit 3cc4374: where ``RingWake.core_m`` is read and where it is not."""

    @staticmethod
    def _state(core):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        motion = circular_motion(np.arange(3)*DT, RADIUS, SPEED, WIND, ALPHA)
        rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3, ring_core_m=core)
        run = run_prescribed(backend(2), wing, polar, motion, [WIND, 0., 0.], rotors=rotors, keep_states=(2,))
        return run, run.states[2]

    def test_hub_inflow_uses_the_ring_core(self):
        from dataclasses import replace as dc_replace
        from openfawe.aero.cores import edge_radii
        from openfawe.aero.vortex import segments_velocity
        from openfawe.aero.wake import ambient_velocity, kernel_options, wake_network
        from openfawe.rotor.coupled import hub_apparent_wind
        run, state = self._state(0.6)
        config = backend(2).wake
        motion = circular_motion(np.arange(3)*DT, RADIUS, SPEED, WIND, ALPHA)
        disc = discs()[0].placed(motion.rotation[2], motion.position_m[2])       # the hub where the rings are
        wind = np.array([WIND, 0., 0.])
        got = hub_apparent_wind(state, config, disc, 0, wind)
        hub = disc.centre_m[None, :]
        nodes, edges, gamma = wake_network(state)
        expected = ambient_velocity(wind, hub, state.time_s).astype(float).copy()
        expected += segments_velocity(hub, nodes, edges, gamma, model=config.core_model,
                                      core_radius=edge_radii(state.wing, edges, state.cores, config.core_radius_m), **kernel_options(config))
        expected += state.rings.induced_velocity(hub, config.core_model, 0.6, exclude_source=0, singularity="zero", **kernel_options(config))
        assert_allclose(got, expected[0], rtol=0., atol=1e-12)
        # the other disc's rings are 3 m and more from this hub, where cores of 0.15 and 0.6 m give the same velocity;
        # a core of 3 m reaches the hub, so the routing shows in the value
        plain = hub_apparent_wind(dc_replace(state, rings=dc_replace(state.rings, core_m=None)), config, disc, 0, wind)
        wide = hub_apparent_wind(dc_replace(state, rings=dc_replace(state.rings, core_m=3.0)), config, disc, 0, wind)
        self.assertGreater(np.max(np.abs(plain-wide)), 1e-6)

    def test_lifting_surface_points_use_the_surface_core_not_the_ring_core(self):
        from dataclasses import replace as dc_replace
        from openfawe.aero.wake import ring_induction
        _, state = self._state(0.6)
        config = backend(2).wake
        with_core = ring_induction(state.rings, state.wing.cp, config)
        without = ring_induction(dc_replace(state.rings, core_m=None), state.wing.cp, config)
        assert_allclose(with_core, without, rtol=0., atol=0.)

    def test_the_multi_body_run_carries_the_ring_core(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        motion = circular_motion(np.arange(3)*DT, RADIUS, SPEED, WIND, ALPHA)
        rotors = PrescribedRotors(discs(), 120., nodes_per_ring=8, surface_core_m=0.3, ring_core_m=0.6)
        multi = run_prescribed_multi(backend(2), [PrescribedBody("k", wing, polar, motion, rotors=rotors)], [WIND, 0., 0.], keep_states=(2,))
        self.assertEqual(multi.status, "complete", multi.error)
        self.assertEqual(multi.states[2].rings.core_m, 0.6)
        self.assertEqual(multi.records[-1]["bodies"]["k"]["ring_core_m"], 0.6)
        single, _ = self._state(0.6)
        assert_allclose(multi.states[2].rings.nodes, single.states[2].rings.nodes, rtol=0., atol=0.)

    def test_tree_code_honours_mixed_per_segment_cores(self):
        from openfawe.aero.vortex import CoreModel, segments_velocity
        rng = np.random.default_rng(7)
        nodes = rng.uniform(-20., 20., (600, 3))
        edges = np.column_stack([np.arange(0, 599), np.arange(1, 600)])
        gamma = rng.uniform(-10., 10., len(edges))
        radii = np.where(np.arange(len(edges)) < 300, 0.15, 0.9)                 # wing-like and ring-like cores
        points = rng.uniform(-20., 20., (400, 3))
        exact = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=radii, backend="numba")
        fine = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=radii, backend="tree",
                                 tree=TreeSettings(theta=1e-9, order=2, leaf_size=32))
        scale = np.max(np.linalg.norm(exact, axis=1))
        self.assertLess(np.max(np.linalg.norm(fine-exact, axis=1))/scale, 1e-12)
        usual = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=radii, backend="tree",
                                  tree=TreeSettings(theta=0.2, order=2, leaf_size=32))
        self.assertLess(np.max(np.linalg.norm(usual-exact, axis=1))/scale, 1e-3)
        # and the cores matter: a uniform core gives a different field
        uniform = segments_velocity(points, nodes, edges, gamma, model=CoreModel.LAMB_OSEEN, core_radius=0.15, backend="numba")
        self.assertGreater(np.max(np.linalg.norm(uniform-exact, axis=1))/scale, 1e-6)


class TetherTests(unittest.TestCase):
    def test_each_body_carries_its_own_tether_and_the_records_hold_its_fields(self):
        wing, polar = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.)
        steps = 3
        times = np.arange(steps+1)*DT
        motion_a = circular_motion(times, RADIUS, SPEED, WIND, ALPHA)
        motion_b = circular_motion(times, RADIUS, SPEED, WIND, ALPHA, phase_rad=2.0)
        motion_b = type(motion_b)(times, motion_b.position_m+np.array([600., 0., 0.]), motion_b.rotation, motion_b.velocity_m_s,
                                  motion_b.angular_velocity_rad_s, motion_b.acceleration_m_s2, None, description="b")
        line = LumpedTether(TetherProperties(length_m=400.), TetherSettings(num_nodes=8))
        bridle = Bridle(np.array([0., 0., -0.2]), 1.0)
        tether_a = PrescribedTether(line, np.array([-380., 0., 0.]), bridle, substeps=10)
        tether_b = PrescribedTether(line, np.array([220., 0., 0.]), bridle, substeps=10)
        bodies = [PrescribedBody("a", wing, polar, motion_a, tether_a), PrescribedBody("b", wing, polar, motion_b, tether_b)]
        run = run_prescribed_multi(backend(steps), bodies, [WIND, 0., 0.], keep_tether_states=(steps,))
        self.assertEqual(run.status, "complete", run.error)
        self.assertEqual(sorted(run.tether_settings), ["a", "b"])
        last = run.records[-1]["bodies"]
        for name in ("a", "b"):
            self.assertIn("tether_tension_kite_N", last[name])
            self.assertIn("tether_force_body_N", last[name])
            self.assertEqual(run.tether_states[name][steps][0].shape, (8, 3))
        # the single-body run with the same tether gives the same tether fields for body a (its wake does not reach b in 3 steps
        # at this separation only to the tree/solver tolerance, so compare the tether, which sees the ambient wind only)
        single = run_prescribed(backend(steps), wing, polar, motion_a, [WIND, 0., 0.], tether=tether_a)
        assert_allclose(last["a"]["tether_tension_kite_N"], single.records[-1]["tether_tension_kite_N"], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
