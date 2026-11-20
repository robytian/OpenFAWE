"""Opt-in far-wake truncation of the rotor rings (``PrescribedRotors.retained_ring_steps``; ring contract of the G5 task)."""
import json
import unittest

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero.methods import AeroBackend
from openfawe.aero.vortex import CoreModel
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing
from openfawe.flight.free_flight import Release, run_free
from openfawe.flight.kinematics import circular_motion
from openfawe.flight.prescribed import PrescribedRotors, run_prescribed
from openfawe.flight.residual import RigidBody
from openfawe.flight.rigid_body import state_from_pose
from openfawe.rotor.actuator_disc import ActuatorDisc, ShedRing, ring_edges, ring_nodes
from openfawe.rotor.ring_wake import RingWake
from openfawe.rotor.table import RotorTable

NEWTON = SolveSettings(solver="newton", max_iterations=60)
RADIUS, SPEED, WIND, ALPHA, DT = 130.0, 47.0, 9.0, 3.0, 0.02
NODES = 8


def backend(steps):
    return AeroBackend("vsm_free_wake", wake=WakeConfig(dt_s=DT, max_rows=steps, core_radius_m=0.15, solve=NEWTON))


def discs():
    """Two discs ahead of the wing (the layout of the free-flight tests)."""
    omegas, speeds = np.linspace(40., 260., 12), np.linspace(0.1, 90., 9)
    w, v = np.meshgrid(omegas, speeds, indexing="ij")
    table = RotorTable("synthetic", 2.3, omegas, speeds, 0.18-0.002*v-1e-5*w*v/60., -0.08+0.0015*v)
    axis = [-np.cos(np.radians(3.)), 0., -np.sin(np.radians(3.))]
    return (ActuatorDisc("upper", [-1.9, 1.2, 1.2], axis, 1, table), ActuatorDisc("lower", [-1.6, -3.6, -1.6], axis, 1, table))


def run(steps, **options):
    motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
    rotors = PrescribedRotors(discs(), [150., 120.], nodes_per_ring=NODES, **options)
    out = run_prescribed(backend(steps), Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.), motion,
                         np.array([WIND, 0., 0.]), rotors=rotors, keep_states=(steps,))
    assert out.status == "complete", out.error
    return out


def column(count, spacing, radius=1.15, circulation=2.0, nodes=24):
    """``count`` equal rings on the x axis, the newest at x = 0 and released last, each one step older than the one before."""
    wake = RingWake()
    for k in reversed(range(count)):                         # the oldest first: ``shed`` puts the newest in front
        ring = ring_nodes([k*spacing, 0., 0.], [-1., 0., 0.], radius, nodes)
        wake = wake.shed(ShedRing(ring, ring_edges(nodes), circulation, -k*DT, circulation/spacing, spacing), disc_index=k % 2)
    return wake


class RingSetTests(unittest.TestCase):
    def test_older_rings_are_removed_whole_and_the_rest_is_untouched(self):
        wake = column(6, 0.5, nodes=5)
        kept = wake.released_since(-2.5*DT)                  # the rings of ages 0, 1, 2
        self.assertEqual((kept.count, len(kept.edges)), (15, 15))
        self.assertTrue(np.array_equal(kept.nodes, wake.nodes[:15]))
        self.assertTrue(np.array_equal(kept.edges, wake.edges[:15]))
        self.assertTrue(np.array_equal(kept.circulation, wake.circulation[:15]))
        self.assertTrue(np.array_equal(kept.release_times_s, wake.release_times_s[:15]))
        self.assertTrue(np.array_equal(kept.source, wake.source[:15]))
        self.assertEqual(sorted(np.bincount(kept.edges.ravel(), minlength=15).tolist()), [2]*15)     # closed loops
        self.assertEqual(kept.surface_core_m, wake.surface_core_m)

    def test_nothing_older_returns_the_same_object_and_everything_older_an_empty_set(self):
        wake = column(4, 0.5, nodes=5)
        self.assertIs(wake.released_since(-10.0), wake)
        empty = wake.released_since(1.0)
        self.assertEqual((empty.count, len(empty.edges), len(empty.circulation)), (0, 0, 0))
        assert_allclose(empty.induced_velocity(np.zeros((1, 3)), CoreModel.LAMB_OSEEN, 0.1), 0.0)

    def test_the_order_of_the_nodes_does_not_matter(self):
        wake = column(5, 0.5, nodes=6)
        order = np.random.default_rng(3).permutation(wake.count)
        inverse = np.argsort(order)
        mixed = RingWake(wake.nodes[order], inverse[wake.edges], wake.circulation, wake.release_times_s[order], wake.source[order])
        kept, reference = mixed.released_since(-1.5*DT), wake.released_since(-1.5*DT)
        point = np.array([[0.3, 0.2, -0.1], [-1.0, 0.5, 0.4]])
        assert_allclose(kept.induced_velocity(point, CoreModel.LAMB_OSEEN, 0.05), reference.induced_velocity(point, CoreModel.LAMB_OSEEN, 0.05),
                        rtol=1e-13, atol=1e-15)
        self.assertEqual(kept.count, 12)

    def test_a_ring_with_nodes_of_two_release_times_is_refused(self):
        wake = column(2, 0.5, nodes=5)
        times = wake.release_times_s.copy()
        times[7] = 0.0                                       # one node of the older ring claims the newer time
        broken = RingWake(wake.nodes, wake.edges, wake.circulation, times, wake.source)
        with self.assertRaises(ValueError):
            broken.released_since(-0.5*DT)

    def test_the_cut_column_induces_the_share_of_the_cylinder_formula(self):
        """On the axis at the newest ring: a cylinder of length L gives L/sqrt(L^2+R^2) of the semi-infinite one."""
        radius, spacing, circulation = 1.15, 0.05, 2.0
        full = column(4000, spacing, radius, circulation, nodes=48)                      # 200 m: 174 radii
        point = np.zeros((1, 3))
        whole = full.induced_velocity(point, CoreModel.LAMB_OSEEN, 1e-3)[0, 0]
        for kept_rings in (100, 400, 1600):
            cut = full.released_since(-(kept_rings-0.5)*DT).induced_velocity(point, CoreModel.LAMB_OSEEN, 1e-3)[0, 0]
            length = (kept_rings-0.5)*spacing                # the sheet represented by the kept rings, from the disc plane
            expected = length/np.hypot(length, radius)
            self.assertAlmostEqual(cut/whole, expected, delta=2e-3)
        self.assertLess(abs(1.0-full.released_since(-(1600-0.5)*DT).induced_velocity(point, CoreModel.LAMB_OSEEN, 1e-3)[0, 0]/whole), 2e-4)


class RotorStepTests(unittest.TestCase):
    def test_the_default_and_a_limit_never_reached_leave_the_records_as_they_are(self):
        steps = 5
        plain, roomy = run(steps), run(steps, retained_ring_steps=steps+1)
        for a, b in zip(plain.records, roomy.records):
            self.assertNotIn("ring_retained_steps", a)
            self.assertEqual(b.pop("ring_retained_steps"), steps+1)
            self.assertEqual(json.dumps(a, sort_keys=True, default=str), json.dumps(b, sort_keys=True, default=str))

    def test_a_limit_keeps_the_rings_of_the_newest_steps(self):
        steps, keep = 7, 3
        plain, cut = run(steps), run(steps, retained_ring_steps=keep)
        per_step = 2*NODES
        self.assertEqual([r["ring_nodes"] for r in plain.records], [per_step*(k+1) for k in range(steps+1)])
        self.assertEqual([r["ring_nodes"] for r in cut.records], [per_step*min(k+1, keep) for k in range(steps+1)])
        rings, reference = cut.states[steps].rings, plain.states[steps].rings
        self.assertEqual(sorted(set(np.round(rings.release_times_s/DT).astype(int).tolist())), [steps-2, steps-1, steps])
        self.assertEqual(sorted(np.bincount(rings.edges.ravel(), minlength=rings.count).tolist()), [2]*rings.count)
        # Up to the first removal nothing differs; afterwards the loads move a little (the rings act from the next step on).
        for k in range(keep+1):
            self.assertEqual(json.dumps({key: v for key, v in plain.records[k].items() if key != "ring_nodes"}, sort_keys=True, default=str),
                             json.dumps({key: v for key, v in cut.records[k].items() if key not in ("ring_nodes", "ring_retained_steps")},
                                        sort_keys=True, default=str))
        force, cut_force = np.array(plain.records[-1]["force_inertial_N"]), np.array(cut.records[-1]["force_inertial_N"])
        self.assertGreater(np.linalg.norm(force-cut_force), 0.0)
        self.assertLess(np.linalg.norm(force-cut_force), 0.05*np.linalg.norm(force))
        # The kept rings sit where the same rings of the untruncated run sit, to the induction of the removed ones.
        assert_allclose(rings.nodes, reference.nodes[:rings.count], rtol=0, atol=0.05)

    def test_bad_limits_are_refused(self):
        for value in (0, -2, 1.5, True, "3"):
            with self.assertRaises(ValueError):
                PrescribedRotors(discs(), 150., retained_ring_steps=value)
        self.assertEqual(PrescribedRotors(discs(), 150., retained_ring_steps=np.int64(4)).retained_ring_steps, 4)


class FreeFlightTests(unittest.TestCase):
    """The closed loop runs through ``run_free`` (added after review RR3-10): the truncated set is the one carried."""

    def test_the_free_flight_driver_carries_the_truncated_set_and_matches_the_prescribed_run(self):
        steps, keep = 7, 3
        motion = circular_motion(np.arange(steps+1)*DT, RADIUS, SPEED, WIND, ALPHA)
        body = RigidBody(1700., [0.08, 0.01, -0.13], np.array([[32000., -20., 35.], [-20., 9700., -15.], [35., -15., 38700.]]))
        wing, polar, wind = Wing.ellipse(8, span=25.0, root_chord=1.3), SectionPolar.linear(limit_deg=60.), np.array([WIND, 0., 0.])

        def fly(**options):
            rotors = PrescribedRotors(discs(), [150., 120.], nodes_per_ring=NODES, **options)
            out = run_free(backend(steps), wing, polar, body, state_from_pose(motion.pose(0.), body), wind, times=motion.time_s,
                           pin_to=motion, external=Release(body, motion), rotors=rotors)
            self.assertEqual(out.status, "complete", out.error)
            return out
        plain, cut = fly(), fly(retained_ring_steps=keep)
        per_step = 2*NODES
        self.assertEqual([r["ring_nodes"] for r in plain.records], [per_step*(k+1) for k in range(steps+1)])
        self.assertEqual([r["ring_nodes"] for r in cut.records], [per_step*min(k+1, keep) for k in range(steps+1)])
        rings = cut.aero_state.rings
        self.assertEqual(sorted(set(np.round(rings.release_times_s/DT).astype(int).tolist())), [steps-2, steps-1, steps])
        for k in range(keep+1):                       # nothing removed yet: the records of the two runs are the same
            for key in ("force_inertial_N", "moment_origin_inertial_Nm", "rotor_force_inertial_N", "rotor_thrust_N"):
                self.assertEqual(plain.records[k][key], cut.records[k][key])
        self.assertNotEqual(plain.records[-1]["force_inertial_N"], cut.records[-1]["force_inertial_N"])
        reference = run(steps, retained_ring_steps=keep)  # the prescribed-motion driver with the same horizon
        for mine, theirs in zip(cut.records, reference.records):
            for key in ("force_inertial_N", "rotor_force_inertial_N"):
                scale = np.linalg.norm(theirs[key])
                assert_allclose(mine[key], theirs[key], rtol=0, atol=1e-9*scale)
            self.assertEqual(mine["ring_nodes"], theirs["ring_nodes"])


if __name__ == "__main__":
    unittest.main()
