"""The plant over one aerodynamic step: rigid body and tether advanced together (G5 plant contract, step 1b part one)."""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from numpy.testing import assert_allclose

from openfawe.flight.kinematics import RigidMotion, circular_motion, rotation_from_vector, rotation_vector
from openfawe.flight.plant import HeldLoads, PlantAccount, PlantTether, plant_step, system_energy, system_momentum
from openfawe.flight.prescribed import PrescribedTether, run_tether_alone
from openfawe.flight.residual import RigidBody, load_rigid_body
from openfawe.flight.rigid_body import RigidState, propagate, state_from_pose
from openfawe.tether.bridle import load_bridle
from openfawe.tether.lumped_line import LumpedTether, TetherProperties, TetherSettings

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT/"cases/aircraft/m600_makani_rigid_body.json"
BODY, BRIDLE = load_rigid_body(CASE), load_bridle(CASE)
R0 = rotation_from_vector(0.7*np.array([1., 2., 3.])/np.sqrt(14.))
G = np.array([0., 0., -9.81])
HELD = HeldLoads([-2000., 500., 60000.], [300., -800., 200.])
WIND = np.array([9., 0., 0.])


def angle(a, b):
    return float(np.linalg.norm(rotation_vector(a.T@b)))


def short_line(**settings):
    """A 60 m line of the M600 section with six nodes (stability bound of the sub-step 2.6 ms)."""
    return LumpedTether(TetherProperties(length_m=60., **settings.pop("properties", {})), TetherSettings(num_nodes=6, **settings))


def straight_start(tether, body, state):
    pose = state.pose(body)
    pivot, _ = tether.bridle.pivot(pose)
    towards = tether.anchor_m-pivot
    towards = towards-np.dot(pose.rotation[:, 1], towards)*pose.rotation[:, 1]
    return tether.line.straight_state(tether.anchor_m, pivot+tether.bridle.radius_m*towards/np.linalg.norm(towards))


def hanging(body, tether):
    """Exact equilibrium of a body whose centre of mass is on the bridle pivot, hanging below the anchor."""
    line = tether.line
    n = line.s.num_nodes
    weight, node_weight = body.mass_kg*line.s.gravity_m_s2, line.node_mass*line.s.gravity_m_s2
    length = line.natural*(1.+(weight+node_weight*(n-np.arange(n+1)))/line.p.tensile_stiffness_N)
    depth = np.cumsum(length)
    nodes = tether.anchor_m+np.column_stack([np.zeros(n), np.zeros(n), -depth[:n]])
    pivot = tether.anchor_m+np.array([0., 0., -depth[n]-tether.bridle.radius_m])
    return RigidState(0., pivot, np.eye(3), np.zeros(3), np.zeros(3)), nodes, np.zeros_like(nodes)


TEST_BODY = RigidBody(BODY.mass_kg, BRIDLE.pivot_body_m, np.diag(np.diag(BODY.inertia_body_kg_m2)))


class WithoutTetherTests(unittest.TestCase):
    def test_without_a_tether_a_step_is_the_rigid_body_scheme(self):
        anchor = np.array([10., -20., 300.])
        start = RigidState(0., anchor+np.array([5., -3., 2.]), R0, np.array([1., -2., 0.5]), np.array([0.1, -0.2, 0.15]))

        def spring(time, state, model=None):
            return -4.*BODY.mass_kg*(state.position_cm_m-anchor), np.zeros(3)
        alone = propagate(BODY, start, lambda t, s: (s.rotation@HELD.force_body_N+spring(t, s)[0], s.rotation@HELD.moment_cm_body_Nm),
                          0.002, 100)
        state = start
        for k in range(10):
            state, nodes, speeds, account = plant_step(BODY, state, step_s=0.02, substeps=10, held=HELD, external=spring)
            other = alone[(k+1)*10]
            assert_allclose(state.position_cm_m, other.position_cm_m, rtol=1e-13, atol=1e-12)
            assert_allclose(state.velocity_cm_m_s, other.velocity_cm_m_s, rtol=1e-13, atol=1e-12)
            assert_allclose(state.angular_velocity_body_rad_s, other.angular_velocity_body_rad_s, rtol=1e-13, atol=1e-14)
            self.assertLess(angle(other.rotation, state.rotation), 1e-13)
            self.assertAlmostEqual(state.time_s, other.time_s, places=13)      # the caller adds the steps up; propagate counts them
        self.assertIsNone(nodes)
        self.assertIsInstance(account, PlantAccount)
        assert_allclose(account.impulse_N_s["anchor"], 0., atol=0)             # no line: nothing from it in the accounts
        self.assertEqual(account.work_J["tether"], 0.)

    def test_the_work_account_is_the_change_of_the_kinetic_energy_of_the_body(self):
        state = RigidState(0., np.zeros(3), R0, np.array([-30., 4., 8.]), np.array([0.1, -0.2, 0.15]))
        before = system_energy(BODY, state)
        new, _, _, account = plant_step(BODY, state, step_s=0.02, substeps=10, held=HELD)
        after = system_energy(BODY, new)
        kinetic = (after["body_translation"]+after["body_rotation"])-(before["body_translation"]+before["body_rotation"])
        self.assertAlmostEqual(kinetic, account.total_work_J(), delta=1e-9*abs(account.work_J["held"]))
        # Gravity's work is minus the change of the potential the energy function uses.
        self.assertAlmostEqual(account.work_J["gravity"], before["body_gravity"]-after["body_gravity"], delta=1e-9*abs(account.work_J["gravity"]))
        self.assertGreater(abs(account.work_J["held"]), 1e3)

    def test_the_held_loads_turn_with_the_body(self):
        # A held body-axis force on a body spinning about its principal y axis is the body-fixed thrust of P1k.
        body = RigidBody(BODY.mass_kg, BODY.centre_of_mass_body_m, np.diag(np.diag(BODY.inertia_body_kg_m2)))
        spin, thrust, v0 = 0.5, 4000., np.array([-40., 5., 12.])
        state = RigidState(0., np.zeros(3), R0, v0, np.array([0., spin, 0.]))
        for _ in range(100):
            state, _, _, _ = plant_step(body, state, step_s=0.02, substeps=2, held=HeldLoads([thrust, 0., 0.]))
        t, a = state.time_s, thrust/body.mass_kg
        exact = v0*t+0.5*G*t*t+a*(R0@np.array([(1.-np.cos(spin*t))/spin**2, 0., (np.sin(spin*t)-spin*t)/spin**2]))
        assert_allclose(state.position_cm_m, exact, rtol=0, atol=1e-9)

    def test_a_held_moment_is_in_body_axes(self):
        # From rest at the attitude R0, a held moment about the principal body y axis spins the body about that axis and no
        # other: the body rate is M t / I_yy exactly. Read as inertial components, the same numbers would act about
        # R0^T (0, M, 0), which has parts along all three body axes.
        body = RigidBody(BODY.mass_kg, BODY.centre_of_mass_body_m, np.diag(np.diag(BODY.inertia_body_kg_m2)))
        moment = -800.
        state = RigidState(0., np.zeros(3), R0, np.zeros(3), np.zeros(3))
        for _ in range(10):
            state, _, _, _ = plant_step(body, state, step_s=0.02, substeps=5, held=HeldLoads([0., 0., 0.], [0., moment, 0.]))
        rate = moment*state.time_s/body.inertia_body_kg_m2[1, 1]
        assert_allclose(state.angular_velocity_body_rad_s, [0., rate, 0.], rtol=1e-12, atol=1e-16)
        self.assertLess(angle(R0@rotation_from_vector([0., 0.5*rate*state.time_s, 0.]), state.rotation), 1e-13)
        wrong = np.linalg.solve(body.inertia_body_kg_m2, R0.T@np.array([0., moment, 0.]))*state.time_s
        self.assertGreater(np.linalg.norm(wrong-[0., rate, 0.]), 0.1*abs(rate))     # the other reading is far from what was found

    def test_set_up_is_checked(self):
        state = RigidState(0., np.zeros(3), np.eye(3), np.zeros(3), np.zeros(3))
        tether = PlantTether(short_line(), np.zeros(3), BRIDLE)
        nodes = np.zeros((6, 3))
        with self.assertRaises(ValueError):                                     # a node state without a line
            plant_step(BODY, state, step_s=0.02, nodes=nodes, node_velocities=nodes)
        with self.assertRaises(ValueError):                                     # a line without its node state
            plant_step(BODY, state, step_s=0.02, tether=tether)
        with self.assertRaises(ValueError):                                     # the sub-step above the line's stability bound
            plant_step(BODY, state, step_s=0.02, substeps=2, tether=tether, nodes=nodes+np.arange(6)[:, None], node_velocities=nodes)
        with self.assertRaises(ValueError):                                     # the line's gravity is not the body's
            plant_step(BODY, state, step_s=0.02, substeps=20, tether=tether, nodes=nodes, node_velocities=nodes, gravity_m_s2=(0., 0., -1.62))
        with self.assertRaises(ValueError):
            plant_step(BODY, state, step_s=0.02, held=dict(force=[0., 0., 0.]))
        with self.assertRaises(ValueError):
            plant_step(BODY, state, step_s=-0.02)
        with self.assertRaises(ValueError):
            PlantTether("line", np.zeros(3), BRIDLE)
        with self.assertRaises(FloatingPointError):
            HeldLoads([np.nan, 0., 0.])
        with self.assertRaises(FloatingPointError):                             # a load outside the finite range fails the step
            plant_step(BODY, state, step_s=0.02, external=lambda t, s, m: (np.array([np.inf, 0., 0.]), np.zeros(3)))


class TetherCouplingTests(unittest.TestCase):
    def test_a_body_hanging_in_equilibrium_stays_where_it_is(self):
        tether = PlantTether(short_line(), [0., 0., 100.], BRIDLE)
        state, nodes, speeds = hanging(TEST_BODY, tether)
        start = state.position_cm_m.copy()
        out = tether.loads(TEST_BODY, state, nodes, speeds, None)[0]
        self.assertAlmostEqual(out["tension"][-1], TEST_BODY.mass_kg*9.81, delta=1e-9*TEST_BODY.mass_kg*9.81)
        for _ in range(25):
            state, nodes, speeds, account = plant_step(TEST_BODY, state, step_s=0.02, substeps=20, tether=tether, nodes=nodes,
                                                       node_velocities=speeds)
        self.assertLess(np.linalg.norm(state.position_cm_m-start), 1e-10)
        # The weight is carried by the line: over a step the two impulses on the body cancel, and the anchor carries everything.
        total = TEST_BODY.mass_kg*9.81+tether.line.node_mass*9.81*6
        assert_allclose(account.impulse_N_s["anchor"], [0., 0., total*0.02], rtol=1e-9, atol=1e-9)

    def test_the_momentum_of_body_and_line_changes_by_the_impulses_of_the_external_forces(self):
        tether = PlantTether(short_line(), np.zeros(3), BRIDLE)
        elevation = np.radians(40.)
        origin = 64.*np.array([np.cos(elevation), 0., np.sin(elevation)])
        state = RigidState(0., origin+R0@BODY.centre_of_mass_body_m, R0, np.array([-6., 4., 8.]), np.array([0.1, -0.2, 0.15]))
        nodes, speeds = straight_start(tether, BODY, state)
        taut = 0
        for _ in range(60):
            before = system_momentum(BODY, state, tether, speeds)
            state, nodes, speeds, account = plant_step(BODY, state, step_s=0.02, substeps=20, tether=tether, nodes=nodes,
                                                       node_velocities=speeds, held=HELD, wind=WIND)
            change = system_momentum(BODY, state, tether, speeds)-before
            scale = sum(np.linalg.norm(v) for v in account.impulse_N_s.values())
            self.assertLess(np.linalg.norm(change-account.total_impulse_N_s()), 1e-10*scale)
            taut += np.linalg.norm(account.impulse_N_s["anchor"]) > 1.
        self.assertGreater(taut, 5)                                             # the line did pull on its anchor
        self.assertGreater(np.linalg.norm(account.impulse_N_s["line_aero"]), 0.)

    def test_one_step_of_n_sub_steps_is_n_steps_of_one_sub_step(self):
        # Criteria C4 and C5 are run one sub-step per call; the plant is used with several sub-steps per call.
        tether = PlantTether(short_line(), np.zeros(3), BRIDLE)
        elevation = np.radians(40.)
        origin = 64.*np.array([np.cos(elevation), 0., np.sin(elevation)])
        start = RigidState(0., origin+R0@BODY.centre_of_mass_body_m, R0, np.array([-6., 4., 8.]), np.array([0.1, -0.2, 0.15]))
        nodes, speeds = straight_start(tether, BODY, start)
        together = plant_step(BODY, start, step_s=0.02, substeps=20, tether=tether, nodes=nodes, node_velocities=speeds, held=HELD, wind=WIND)
        state, impulse, work = start, np.zeros(3), 0.
        for _ in range(20):
            state, nodes, speeds, account = plant_step(BODY, state, step_s=0.001, substeps=1, tether=tether, nodes=nodes,
                                                       node_velocities=speeds, held=HELD, wind=WIND)
            impulse = impulse+account.total_impulse_N_s()
            work += account.total_work_J()
        assert_allclose(state.position_cm_m, together[0].position_cm_m, rtol=1e-13, atol=1e-12)
        assert_allclose(state.velocity_cm_m_s, together[0].velocity_cm_m_s, rtol=1e-12, atol=1e-12)
        assert_allclose(state.angular_velocity_body_rad_s, together[0].angular_velocity_body_rad_s, rtol=1e-12, atol=1e-14)
        self.assertLess(angle(together[0].rotation, state.rotation), 1e-13)
        assert_allclose(nodes, together[1], rtol=1e-13, atol=1e-12)
        assert_allclose(speeds, together[2], rtol=1e-11, atol=1e-11)
        assert_allclose(impulse, together[3].total_impulse_N_s(), rtol=1e-11, atol=1e-9)
        self.assertAlmostEqual(work, together[3].total_work_J(), delta=1e-10*abs(work)+1e-9)
        self.assertAlmostEqual(state.time_s, together[0].time_s, places=13)

    def test_body_and_line_together_converge_at_fourth_order_when_the_line_is_resolved(self):
        # The one-node soft line of criterion C8c with the body swinging about the bridle pivot, for 2 s: errors of the pitch
        # angle and of the position against a fine run fall by about 16 on halving the sub-step. A tether force or a bridle
        # point taken from the start of the sub-step would give first order.
        line = LumpedTether(TetherProperties(length_m=60., tensile_stiffness_N=2e5, bending_stiffness_Nm2=0.),
                            TetherSettings(num_nodes=1, longitudinal_damping_ratio=0., bending_damping_ratio=0., air_density_kg_m3=0.))
        tether = PlantTether(line, [0., 0., 100.], BRIDLE)
        offset = np.array([-0.10, 0., -0.30])
        body = RigidBody(TEST_BODY.mass_kg, TEST_BODY.centre_of_mass_body_m+offset, TEST_BODY.inertia_body_kg_m2)

        def pulse(time, state, model):
            return (5000.*np.sin(0.5*np.pi*time)**2*np.array([0.6, 0., -0.8]) if time <= 2. else np.zeros(3)), np.zeros(3)

        def run(substeps):
            hung, nodes, speeds = hanging(TEST_BODY, tether)
            state = RigidState(0., hung.position_cm_m+offset, np.eye(3), np.zeros(3), np.zeros(3))
            for _ in range(50):
                state, nodes, speeds, _ = plant_step(body, state, step_s=0.04, substeps=substeps, tether=tether, nodes=nodes,
                                                     node_velocities=speeds, external=pulse)
            return rotation_vector(state.rotation)[1], state.position_cm_m
        fine = run(16)
        runs = [run(count) for count in (1, 2, 4)]
        pitch = [abs(r[0]-fine[0]) for r in runs]
        place = [np.linalg.norm(r[1]-fine[1]) for r in runs]
        self.assertGreater(abs(fine[0]), 0.05)                                  # the body did swing
        for errors in (pitch, place):
            self.assertGreater(errors[2], 1e-12)                                # above rounding
            for coarse, finer in zip(errors[:-1], errors[1:]):
                self.assertGreater(np.log2(coarse/finer), 3.5)
                self.assertLess(np.log2(coarse/finer), 4.5)

    def test_a_callback_that_cancels_the_line_sees_its_force_with_the_right_sign_and_the_body_falls_freely(self):
        tether = PlantTether(short_line(), [0., 0., 100.], BRIDLE)
        state, nodes, speeds = hanging(TEST_BODY, tether)
        before = system_momentum(TEST_BODY, state, tether, speeds)
        cut = lambda t, s, model: (-model["tether_force_inertial_N"], -model["tether_moment_cm_inertial_Nm"])      # noqa: E731
        new, _, new_speeds, account = plant_step(TEST_BODY, state, step_s=0.02, substeps=20, tether=tether, nodes=nodes,
                                                 node_velocities=speeds, external=cut)
        # The line's force on the body is exactly undone: the body falls under gravity alone (a wrong sign of the model
        # force handed to the callback would double the line's pull instead and the body would stay up).
        assert_allclose(new.velocity_cm_m_s, G*0.02, rtol=1e-12, atol=1e-13)
        # The cancelling force is booked as external and the account still closes.
        change = system_momentum(TEST_BODY, new, tether, new_speeds)-before
        assert_allclose(change, account.total_impulse_N_s(), rtol=0, atol=1e-8)
        self.assertGreater(account.impulse_N_s["external"][2], -1.05*TEST_BODY.mass_kg*9.81*0.02)
        self.assertLess(account.impulse_N_s["external"][2], -0.9*TEST_BODY.mass_kg*9.81*0.02)

    def test_the_momentum_account_opens_when_action_and_reaction_are_not_equal(self):
        # Negative control: the force of the line on the body scaled by 1.01 while the line itself is left as it is.
        tether = PlantTether(short_line(), [0., 0., 100.], BRIDLE)
        state, nodes, speeds = hanging(TEST_BODY, tether)
        before = system_momentum(TEST_BODY, state, tether, speeds)
        original = PlantTether.loads

        def unequal(self, body, state, positions, velocities, wind):
            out, point, force, moment = original(self, body, state, positions, velocities, wind)
            return out, point, 1.01*force, 1.01*moment
        with mock.patch.object(PlantTether, "loads", unequal):
            new, _, new_speeds, account = plant_step(TEST_BODY, state, step_s=0.02, substeps=20, tether=tether, nodes=nodes,
                                                     node_velocities=speeds)
        gap = system_momentum(TEST_BODY, new, tether, new_speeds)-before-account.total_impulse_N_s()
        self.assertAlmostEqual(gap[2], 0.01*TEST_BODY.mass_kg*9.81*0.02, delta=0.05*0.01*TEST_BODY.mass_kg*9.81*0.02)

    def test_the_wind_on_the_line_can_be_a_field_and_a_failing_line_fails_the_step(self):
        tether = PlantTether(short_line(), np.zeros(3), BRIDLE)
        elevation = np.radians(40.)
        state = RigidState(0., 64.*np.array([np.cos(elevation), 0., np.sin(elevation)]), R0, np.array([-6., 4., 8.]), np.zeros(3))
        nodes, speeds = straight_start(tether, BODY, state)
        common = dict(step_s=0.02, substeps=20, tether=tether, nodes=nodes, node_velocities=speeds)
        uniform = plant_step(BODY, state, wind=WIND, **common)
        field = plant_step(BODY, state, wind=lambda points, time: np.broadcast_to(WIND, points.shape), **common)
        self.assertTrue(np.array_equal(uniform[1], field[1]))
        still = plant_step(BODY, state, **common)                               # no wind given: still air, the line still has drag
        self.assertFalse(np.array_equal(uniform[1], still[1]))
        with self.assertRaises(FloatingPointError):                             # a wind outside the finite range ends in the line's state
            plant_step(BODY, state, wind=lambda points, time: np.full(points.shape, 1e200), **common)

    def test_energy_and_the_work_account_hold_with_roll_and_yaw(self):
        # The M600 body (centre of mass 0.52 m beside the bridle pivot, full inertia tensor) on the conservative line, pushed
        # and twisted out of the plane: the mechanical energy changes by the external work, and the kinetic energy of the body
        # by the sum of the four works, the line's among them.
        line = short_line(properties=dict(bending_stiffness_Nm2=0.), longitudinal_damping_ratio=0., bending_damping_ratio=0.,
                          air_density_kg_m3=0.)
        tether = PlantTether(line, [0., 0., 100.], BRIDLE)
        hung, nodes, speeds = hanging(TEST_BODY, tether)
        state = RigidState(0., hung.position_cm_m-BRIDLE.pivot_body_m+BODY.centre_of_mass_body_m, np.eye(3), np.zeros(3), np.zeros(3))
        first = system_energy(BODY, state, tether, nodes, speeds)["total"]

        def pulse(time, state, model):
            shape = np.sin(0.5*np.pi*time)**2 if time <= 2. else 0.
            return 5000.*shape*np.array([0.6, 0.4, -0.6928]), 3000.*shape*np.array([0.3, 0.2, 1.0])
        external = absolute = works = everything = line_work = worst = gap = 0.
        vectors = []
        for _ in range(125):
            state, nodes, speeds, account = plant_step(BODY, state, step_s=0.02, substeps=40, tether=tether, nodes=nodes,
                                                       node_velocities=speeds, external=pulse)
            external += account.work_J["external"]
            absolute += abs(account.work_J["external"])
            works += account.total_work_J()
            everything += sum(abs(v) for v in account.work_J.values())
            line_work += abs(account.work_J["tether"])
            energy = system_energy(BODY, state, tether, nodes, speeds)
            worst = max(worst, abs(energy["total"]-first-external))
            gap = max(gap, abs(energy["body_translation"]+energy["body_rotation"]-works))
            vectors.append(rotation_vector(state.rotation))
        vectors = np.array(vectors)
        self.assertGreater(np.ptp(vectors[:, 0]), 0.05)                         # the body rolled
        self.assertGreater(np.ptp(vectors[:, 2]), 0.05)                         # and yawed
        self.assertGreater(line_work, 100.)                                     # the line did work on the body
        self.assertLess(worst, 1e-8*absolute)
        self.assertLess(gap, 1e-8*everything)

    def test_energy_of_the_conservative_subset_changes_by_the_work_of_the_external_force(self):
        line = short_line(properties=dict(bending_stiffness_Nm2=0.), longitudinal_damping_ratio=0., bending_damping_ratio=0.,
                          air_density_kg_m3=0.)
        tether = PlantTether(line, [0., 0., 100.], BRIDLE)
        state, nodes, speeds = hanging(TEST_BODY, tether)
        first = system_energy(TEST_BODY, state, tether, nodes, speeds)["total"]

        def pulse(time, state, model):
            return (5000.*np.sin(0.5*np.pi*time)**2*np.array([0.6, 0., -0.8]) if time <= 2. else np.zeros(3)), np.zeros(3)
        work = absolute = worst = 0.
        for _ in range(125):
            state, nodes, speeds, account = plant_step(TEST_BODY, state, step_s=0.02, substeps=40, tether=tether, nodes=nodes,
                                                       node_velocities=speeds, external=pulse)
            work += account.work_J["external"]
            absolute += abs(account.work_J["external"])
            worst = max(worst, abs(system_energy(TEST_BODY, state, tether, nodes, speeds)["total"]-first-work))
        self.assertGreater(absolute, 100.)                                      # the pulse did work
        self.assertLess(worst, 1e-6*absolute)
        self.assertLess(abs(state.position_cm_m[1]), 1e-9)                      # the motion stayed in the x-z plane

    def test_a_cancelling_callback_forces_the_reference_motion_and_the_line_matches_the_prescribed_path(self):
        radius, speed, distance = 30., 20., 64.5
        shift = np.array([np.sqrt(distance**2-radius**2), 0., 0.])

        def reference(times):
            m = circular_motion(np.asarray(times, dtype=float), radius, speed, 9., 3.)
            return RigidMotion(m.time_s, m.position_m+shift, m.rotation, m.velocity_m_s, m.angular_velocity_rad_s, m.acceleration_m_s2,
                               m.angular_acceleration_rad_s2)

        def at(time):
            motion = reference([0., time if time > 0. else 1.])
            k = 1 if time > 0. else 0
            return motion.pose(motion.time_s[k]), motion.acceleration_m_s2[k]

        def cancel(time, state, model):
            pose, acceleration = at(time)
            omega = pose.angular_velocity_rad_s
            arm = pose.rotation@BODY.centre_of_mass_body_m
            inertia = pose.rotation@BODY.inertia_body_kg_m2@pose.rotation.T
            return (BODY.mass_kg*(acceleration+np.cross(omega, np.cross(omega, arm)))-BODY.mass_kg*G-model["force_inertial_N"],
                    np.cross(omega, inertia@omega)-model["moment_cm_inertial_Nm"])
        tether = PlantTether(short_line(), np.zeros(3), BRIDLE)
        state = state_from_pose(at(0.)[0], BODY)
        nodes, speeds = straight_start(tether, BODY, state)
        first = (nodes.copy(), speeds.copy())
        h, count = 0.001, 500
        tensions = []
        for k in range(count):
            state, nodes, speeds, _ = plant_step(BODY, state, step_s=h, substeps=1, tether=tether, nodes=nodes, node_velocities=speeds,
                                                 external=cancel, wind=WIND)
            state = RigidState((k+1)*h, state.position_cm_m, state.rotation, state.velocity_cm_m_s, state.angular_velocity_body_rad_s)
            if (k+1) % 20 == 0:
                tensions.append(float(tether.loads(BODY, state, nodes, speeds, WIND)[0]["tension"][-1]))
        exact, pose = at(state.time_s)[0], state.pose(BODY)
        assert_allclose(pose.position_m, exact.position_m, rtol=0, atol=1e-9*radius)
        assert_allclose(pose.velocity_m_s, exact.velocity_m_s, rtol=0, atol=1e-9*speed)
        self.assertLess(angle(exact.rotation, state.rotation), 1e-9)
        prescribed = PrescribedTether(tether.line, tether.anchor_m, BRIDLE, substeps=1, positions=first[0], velocities=first[1])
        final, report = run_tether_alone(prescribed, reference(np.arange(count+1)*h), WIND, record_every=20)
        other = np.array([row["tension_kite_N"] for row in report])
        self.assertGreater(other.mean(), 1000.)                                 # a taut line: the comparison means something
        self.assertLess(np.abs(np.array(tensions)-other).max(), 1e-3*other.mean())
        self.assertLess(np.linalg.norm(nodes-final.positions, axis=1).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
