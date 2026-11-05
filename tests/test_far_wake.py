"""Opt-in far-wake treatments: only the newest rows convect with the induced velocity
(WakeConfig.free_convection_rows); rows older than a limit are discarded (WakeConfig.retained_rows)."""
import unittest
from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose

from openfawe.aero.methods import AeroBackend
from openfawe.aero.vortex import TreeSettings
from openfawe.aero.cores import edge_radii
from openfawe.aero.lifting_surface import LiftingSurfaceConfig, validate_lifting_surface_wake
from openfawe.aero.ll_straight import LLStraightConfig
from openfawe.aero.ll_straight import validate_configuration as validate_straight_ll
from openfawe.aero.vortex import segments_velocity
from openfawe.aero.vsm_wake import _solve as vsm_solve
from openfawe.aero.vsm_wake import advance_vsm_wake, history_vsm_induction
from openfawe.aero.wake import current_influence, pack_rings
from openfawe.aero.wing import solve_circulation
from openfawe.aero_archive import AeroWriter
from openfawe.aero.wake import WakeConfig, WakeState, node_convection_velocity, wake_network
from openfawe.aero.wing import SectionPolar, SolveSettings, Wing, wake_config_metadata
from openfawe.rotor.actuator_disc import ShedRing, ring_edges, ring_nodes
from openfawe.rotor.ring_wake import RingWake

NEWTON = SolveSettings(solver="newton", max_iterations=60)
DT, WIND = 0.05, np.array([12., 0., 0.8])


def config(limit, steps=12, **options):
    return WakeConfig(dt_s=DT, max_rows=steps, core_radius_m=0.15, solve=NEWTON, free_convection_rows=limit, **options)


def truncated(keep, steps, method="vsm_free_wake", wind=WIND, limit=None, rows=None, **options):
    """March with ``retained_rows=keep`` (None: complete wake); returns the state of every step."""
    return march(limit, steps, wind=wind, method=method, rows=steps if rows is None else rows, retained_rows=keep, **options)


def march(limit, steps, wind=WIND, method="vsm_free_wake", rows=None, **options):
    wing, polar = Wing.ellipse(8), SectionPolar.linear()
    backend = AeroBackend(method, wake=config(limit, steps if rows is None else rows, **options))
    state = backend.start(wing, polar, wind)
    states = {0: state}
    for index in range(1, steps+1):
        state = backend.advance(state, polar, wind, next_time_s=index*DT)
        states[index] = state
    return states


def shear(points, time):
    """Ambient wind growing with height: 12 m/s at z = 0, 0.5 m/s more per metre."""
    out = np.zeros_like(points)
    out[:, 0] = 12.+0.5*points[:, 2]
    out[:, 2] = 0.8
    return out


class ConfigurationTests(unittest.TestCase):
    def test_default_is_off_and_leaves_the_metadata_as_it_was(self):
        self.assertIsNone(WakeConfig().free_convection_rows)
        self.assertNotIn("free_convection_rows", wake_config_metadata(WakeConfig()))
        self.assertEqual(wake_config_metadata(config(7))["free_convection_rows"], 7)
        self.assertEqual(AeroBackend("vsm_free_wake", wake=config(7)).metadata()["wake"]["free_convection_rows"], 7)

    def test_invalid_limits_are_rejected(self):
        for bad in (0, -3, 2.5, True, "4"):
            with self.assertRaises(ValueError):
                config(bad)
        with self.assertRaises(ValueError):                       # nothing to limit when no row is free
            config(4, free_wake=False)
        self.assertIs(type(config(np.int64(5)).free_convection_rows), int)          # stored as a plain integer

    def test_only_the_methods_that_apply_the_limit_accept_it_and_they_declare_it(self):
        for method in ("vsm", "ll_straight", "vlm", "vlm_free_wake"):
            with self.assertRaises(ValueError, msg=method):
                AeroBackend(method, wake=config(4))
        for method in ("free_wake", "vsm_free_wake"):
            data = AeroBackend(method, wake=config(4)).metadata()
            self.assertIn("ambient", data["far_wake"])
            self.assertNotIn("far_wake", AeroBackend(method, wake=config(None)).metadata())


class ConvectionTests(unittest.TestCase):
    def test_a_limit_the_wake_never_reaches_changes_nothing_bitwise(self):
        for method in ("vsm_free_wake", "free_wake"):
            plain, limited = march(None, 6, method=method), march(6, 6, method=method)      # 6 point rows at the last advance
            for index in plain:
                self.assertTrue(np.array_equal(plain[index].free_rows, limited[index].free_rows), (method, index))
                self.assertTrue(np.array_equal(plain[index].solution.circulation, limited[index].solution.circulation))

    def test_free_rows_get_the_full_velocity_and_older_rows_the_ambient_velocity(self):
        limit, steps = 4, 9
        for wind in (WIND, shear):
            state = march(limit, steps, wind=wind)[steps]
            limited = node_convection_velocity(state, state.config, wind)                  # (rows, nodes, 3); row 0 = trailing edge
            unlimited_config = replace(state.config, free_convection_rows=None)
            full = node_convection_velocity(replace(state, config=unlimited_config), unlimited_config, wind)
            self.assertEqual(limited.shape, (steps+1, state.wing.node_count, 3))
            self.assertTrue(np.array_equal(limited[:limit], full[:limit]))                  # bit for bit: same per-point sums
            points = np.concatenate(([state.wing.trailing_edge], state.free_rows))[limit:].reshape(-1, 3)
            ambient = np.broadcast_to(wind, points.shape) if not callable(wind) else wind(points, state.time_s)
            self.assertTrue(np.array_equal(limited[limit:].reshape(-1, 3), ambient))
            self.assertGreater(np.abs(full[limit:].reshape(-1, 3)-ambient).max(), 1e-3)    # the induced part is really left out

    def test_old_rows_drift_rigidly_in_a_uniform_wind(self):
        limit, steps = 3, 8
        states = march(limit, steps)
        for index in range(limit+1, steps+1):
            old, new = states[index-1], states[index]
            # New free row j is the convected old free row j-1 (new row 0 is the convected old trailing edge).
            # Old point rows limit.. (old free rows limit-1..) are beyond the limit: displaced by wind * dt exactly.
            assert_allclose(new.free_rows[limit:], old.free_rows[limit-1:]+WIND*DT, rtol=0, atol=1e-14)
            # The row just inside the limit still moves with the induced velocity.
            moved = new.free_rows[limit-1]-old.free_rows[limit-2]
            self.assertGreater(np.abs(moved-WIND*DT).max(), 1e-6)
        # Rigid drift keeps the shape of the old rows: spacing between two frozen rows does not change.
        last, before = states[steps], states[steps-1]
        assert_allclose(last.free_rows[-1]-last.free_rows[-2], before.free_rows[-1]-before.free_rows[-2], rtol=0, atol=1e-13)

    def test_the_line_method_and_the_two_stage_integrator_freeze_the_same_rows(self):
        limit, steps = 3, 8
        for method, options in (("free_wake", {}), ("vsm_free_wake", dict(convection_integrator="heun")),
                                ("vsm_free_wake", dict(convection_integrator="rk4", geometry_substeps=2))):
            states = march(limit, steps, method=method, **options)
            plain = march(None, steps, method=method, **options)
            for index in range(limit+1, steps+1):
                old, new = states[index-1], states[index]
                # Uniform wind: every stage velocity of a frozen row is the wind, so the row drifts rigidly (to rounding).
                assert_allclose(new.free_rows[limit:], old.free_rows[limit-1:]+WIND*DT, rtol=0, atol=1e-13, err_msg=str((method, options)))
            self.assertGreater(np.abs(states[steps].free_rows-plain[steps].free_rows).max(), 1e-6)
            self.assertTrue(np.array_equal(states[limit].free_rows, plain[limit].free_rows))

    def test_frozen_rows_follow_a_sheared_wind_over_several_steps(self):
        limit, steps = 3, 8
        states = march(limit, steps, wind=shear)
        for index in range(limit+1, steps+1):
            old, new = states[index-1], states[index]
            frozen = old.free_rows[limit-1:]
            expected = frozen+DT*shear(frozen.reshape(-1, 3), old.time_s).reshape(frozen.shape)      # explicit Euler
            assert_allclose(new.free_rows[limit:], expected, rtol=0, atol=1e-13)
        # Rows at different heights are carried at different speeds: not a rigid drift.
        moved = states[steps].free_rows[-1]-states[steps-1].free_rows[-2]
        self.assertGreater(np.ptp(moved[:, 0]), 1e-4)

    def test_frozen_rows_keep_their_circulation_and_still_act_on_the_wing(self):
        limit, steps = 3, 9
        states = march(limit, steps, rows=steps+1)
        last = states[steps]
        # Row j of the history carries the circulation the wing had when the row was released, unchanged.
        for j in range(steps-1):
            self.assertTrue(np.array_equal(last.older_circulation[j], states[steps-1-j].solution.circulation), j)
        # Remove the circulation of the frozen rows only: the next wing solution changes.
        emptied = last.older_circulation.copy()
        emptied[limit:] = 0.
        backend_config = last.config
        wing, polar = last.wing, SectionPolar.linear()
        kept = advance_vsm_wake(last, polar, backend_config, WIND, next_time_s=(steps+1)*DT)
        without = advance_vsm_wake(replace(last, older_circulation=emptied), polar, backend_config, WIND, next_time_s=(steps+1)*DT)
        change = np.linalg.norm(kept.solution.force-without.solution.force)/np.linalg.norm(kept.solution.force)
        self.assertGreater(change, 1e-5)

    def test_loads_are_identical_until_the_limit_bites_and_close_afterwards(self):
        limit, steps = 4, 12
        plain, limited = march(None, steps), march(limit, steps)
        # Point rows at the advance to step n: n (old trailing edge + n-1 free rows); the limit bites when n > limit.
        for index in range(0, limit+1):
            self.assertTrue(np.array_equal(plain[index].solution.force, limited[index].solution.force), index)
        differences = [np.linalg.norm(plain[k].solution.force-limited[k].solution.force)/np.linalg.norm(plain[k].solution.force)
                       for k in range(limit+1, steps+1)]
        self.assertGreater(max(differences), 0.)
        # A different, declared model. The bound is an empirical regression number for this 12-step wake, not a criterion.
        self.assertLess(max(differences), 5e-3)

    def test_compiled_and_tree_backends_treat_the_free_rows_the_same_way(self):
        limit, steps = 3, 7
        for options in (dict(velocity_backend="numba"), dict(velocity_backend="tree", tree=TreeSettings(theta=0.2, order=2, leaf_size=8))):
            state = march(limit, steps, **options)[steps]
            limited = node_convection_velocity(state, state.config, WIND)
            unlimited_config = replace(state.config, free_convection_rows=None)
            full = node_convection_velocity(replace(state, config=unlimited_config), unlimited_config, WIND)
            self.assertTrue(np.array_equal(limited[:limit], full[:limit]), options["velocity_backend"])
            self.assertTrue(np.array_equal(limited[limit:], np.broadcast_to(WIND, limited[limit:].shape)))


class TruncationConfigurationTests(unittest.TestCase):
    def test_default_is_off_and_leaves_the_metadata_as_it_was(self):
        self.assertIsNone(WakeConfig().retained_rows)
        self.assertNotIn("retained_rows", wake_config_metadata(WakeConfig()))
        data = AeroBackend("vsm_free_wake", wake=config(None, retained_rows=5)).metadata()
        self.assertEqual(data["wake"]["retained_rows"], 5)
        self.assertIn("removed whole", data["far_wake_truncation"])
        self.assertNotIn("far_wake_truncation", AeroBackend("vsm_free_wake", wake=config(None)).metadata())

    def test_invalid_limits_and_methods_are_rejected(self):
        for bad in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                config(None, retained_rows=bad)
        with self.assertRaises(ValueError):                       # more rows kept than the configuration allows at all
            config(None, steps=6, retained_rows=7)
        self.assertIs(type(config(None, retained_rows=np.int64(4)).retained_rows), int)
        for method in ("vsm", "ll_straight", "vlm", "vlm_free_wake"):
            with self.assertRaises(ValueError, msg=method):
                AeroBackend(method, wake=config(None, retained_rows=4))


class TruncationTests(unittest.TestCase):
    def test_a_limit_the_wake_never_reaches_changes_nothing_bitwise(self):
        for method in ("vsm_free_wake", "free_wake"):
            plain, limited = truncated(None, 6, method=method), truncated(6, 6, method=method)
            for index in plain:
                self.assertTrue(np.array_equal(plain[index].free_rows, limited[index].free_rows), (method, index))
                self.assertTrue(np.array_equal(plain[index].solution.force, limited[index].solution.force))

    def test_the_state_holds_the_newest_rows_and_the_run_goes_on_past_max_rows(self):
        keep, steps = 4, 11
        for method in ("vsm_free_wake", "free_wake"):
            # Ageing cores, so that a core history exists and can be checked; max_rows 5: a complete wake would stop at step 5.
            states = truncated(keep, steps, method=method, rows=5, core_evolution="age")
            last = states[steps]
            self.assertEqual((last.step, len(last.free_rows), len(last.release_times_s), len(last.older_circulation)),
                             (steps, keep, keep, keep-1))
            assert_allclose(last.release_times_s, DT*np.arange(steps-1, steps-1-keep, -1), rtol=0, atol=1e-12)
            # The kept strengths are the circulations of the newest steps, unchanged; the last ring is the oldest kept one.
            for j in range(keep-1):
                self.assertTrue(np.array_equal(last.older_circulation[j], states[steps-1-j].solution.circulation), (method, j))
            for index in range(1, steps+1):
                self.assertEqual(len(states[index].free_rows), min(index, keep))
            # Core radii under the 'age' policy depend on the age only: the kept rows carry exactly the radii the
            # newest rows of a complete wake carry after the same number of steps.
            complete = truncated(None, steps, method=method, core_evolution="age")[steps]
            self.assertEqual((last.cores.span.shape[0], last.cores.stream.shape[0]), (keep, keep))
            self.assertTrue(np.array_equal(last.cores.span, complete.cores.span[:keep]), method)
            self.assertTrue(np.array_equal(last.cores.stream, complete.cores.stream[:keep]), method)
        with self.assertRaises(ValueError):                                  # without the truncation the same run stops
            truncated(None, steps, rows=5)

    def test_one_kept_row_is_a_valid_wake(self):
        for method in ("vsm_free_wake", "free_wake"):
            last = truncated(1, 6, method=method)[6]
            self.assertEqual((len(last.free_rows), last.older_circulation.shape), (1, (0, last.wing.panels)))
            nodes, edges, gamma = wake_network(last)
            self.assertEqual(len(nodes), 3*last.wing.node_count)            # bound row, trailing edge, one released row
            self.assertTrue(np.all(np.isfinite(last.solution.force)))

    def test_the_network_of_a_truncated_state_has_the_kept_rows_and_ends_with_the_oldest_kept_ring(self):
        keep, steps = 3, 9
        states = truncated(keep, steps)
        last = states[steps]
        nodes, edges, gamma = wake_network(last)
        self.assertEqual(len(nodes), (keep+2)*last.wing.node_count)         # bound row, trailing edge, three released rows
        # The spanwise filaments of the last kept row carry the wing circulation of step N-R+1: whole rings were removed.
        first = last.wing.node_count
        on_last_row = np.all(edges >= len(nodes)-first, axis=1)
        self.assertEqual(int(on_last_row.sum()), last.wing.panels)
        assert_allclose(np.sort(np.abs(gamma[on_last_row])), np.sort(np.abs(states[steps-keep+1].solution.circulation)), rtol=0, atol=0)
        velocity = node_convection_velocity(last, last.config, WIND)
        self.assertEqual(velocity.shape, (keep+1, first, 3))
        with self.assertRaises(ValueError):                                  # a state with fewer rows than the limit allows is rejected
            replace(last, free_rows=last.free_rows[:-1])
        with self.assertRaises(ValueError):                                  # and so is the complete-wake reading of the same arrays
            replace(last, config=replace(last.config, retained_rows=None))

    def test_the_vsm_solution_of_a_step_sees_exactly_the_rows_the_state_keeps(self):
        keep, steps = 3, 8
        states = truncated(keep, steps)
        previous, last = states[steps-1], states[steps]
        # The same previous content read as a complete-wake state (its own step count is 'keep'), advanced once without
        # truncation: the newest 'keep' rows are those of the truncated advance, but its solution also feels ring keep+1.
        complete_config = replace(previous.config, retained_rows=None)
        as_complete = replace(previous, config=complete_config, step=keep)
        untruncated = advance_vsm_wake(as_complete, SectionPolar.linear(), complete_config, WIND, next_time_s=previous.time_s+DT)
        self.assertTrue(np.array_equal(untruncated.free_rows[:keep], last.free_rows))        # convection is the same
        self.assertEqual(len(untruncated.free_rows), keep+1)
        self.assertGreater(np.abs(untruncated.solution.circulation-last.solution.circulation).max(), 1e-9)
        # Solving on the kept rows alone reproduces the truncated step bit for bit.
        known, known_load = history_vsm_induction(last.wing, last.free_rows, last.older_circulation, last.cores, last.config)
        again = vsm_solve(last.wing, SectionPolar.linear(), last.config, WIND, last.time_s, (0., 0., 0.), last.free_rows[:1],
                          last.cores, known, known_load, previous.solution.circulation)
        self.assertTrue(np.array_equal(again.circulation, last.solution.circulation))

    def test_the_line_method_solution_of_a_step_sees_exactly_the_rows_the_state_keeps(self):
        keep, steps = 3, 8
        states = truncated(keep, steps, method="free_wake")
        previous, last = states[steps-1], states[steps]
        wing, cfg, polar = last.wing, last.config, SectionPolar.linear()
        nodes, edges, gamma = pack_rings(last.free_rows, last.older_circulation, wing.panel_edges)
        radii = edge_radii(wing, edges+2*wing.node_count, last.cores, cfg.core_radius_m)
        known = segments_velocity(wing.cp, nodes, edges, gamma, model=cfg.core_model, core_radius=radii)
        matrix = current_influence(wing, last.free_rows[0], last.cores, cfg)
        again = solve_circulation(wing, polar, np.broadcast_to(WIND, wing.cp.shape), matrix, structural_velocity=np.zeros_like(wing.cp),
                                  known_induction=known, density=cfg.density_kg_m3, kinematic_viscosity=cfg.kinematic_viscosity_m2_s,
                                  initial=previous.solution.circulation, settings=cfg.solve)
        self.assertTrue(np.array_equal(again.circulation, last.solution.circulation))
        # One more ring (the one the truncation removed) changes the known induction.
        complete_config = replace(previous.config, retained_rows=None)
        untruncated = AeroBackend("free_wake", wake=complete_config).advance(
            replace(previous, config=complete_config, step=keep), polar, WIND, next_time_s=previous.time_s+DT)
        self.assertGreater(np.abs(untruncated.solution.circulation-last.solution.circulation).max(), 1e-9)

    def test_loads_equal_the_complete_wake_until_the_first_row_is_dropped_and_stay_close(self):
        keep, steps = 5, 12
        for options in (dict(), dict(convection_integrator="heun"), dict(convection_integrator="rk4", geometry_substeps=2)):
            plain, cut = truncated(None, steps, **options), truncated(keep, steps, **options)
            for index in range(0, keep+1):
                self.assertTrue(np.array_equal(plain[index].solution.force, cut[index].solution.force), (options, index))
            differences = [np.linalg.norm(plain[k].solution.force-cut[k].solution.force)/np.linalg.norm(plain[k].solution.force)
                           for k in range(keep+1, steps+1)]
            self.assertGreater(min(differences), 0.)
            # Five kept rows are 3 m of wake behind a 5 m span. Measured 0.5 % to 1.7 % over these steps with explicit
            # Euler; the bound is a regression guard for this case, not a criterion.
            self.assertLess(max(differences), 0.03)

    def test_a_sheared_wind_runs_with_the_truncation(self):
        last = truncated(4, 9, wind=shear)[9]
        self.assertEqual(len(last.free_rows), 4)
        self.assertTrue(np.all(np.isfinite(last.free_rows)) and np.all(np.isfinite(last.solution.force)))

    def test_truncation_and_the_convection_limit_work_together(self):
        states = truncated(6, 12, limit=3)
        last, before = states[12], states[11]
        self.assertEqual(len(last.free_rows), 6)
        # Rows 3..5 are frozen: they are the previous rows 2..4 moved by the wind.
        assert_allclose(last.free_rows[3:], before.free_rows[2:5]+WIND*DT, rtol=0, atol=1e-14)

    def test_compiled_and_tree_backends_agree_with_the_reference_on_a_truncated_run(self):
        reference = truncated(4, 9)[9]
        fast = truncated(4, 9, velocity_backend="numba")[9]
        assert_allclose(fast.free_rows, reference.free_rows, rtol=0, atol=1e-11)
        assert_allclose(fast.solution.circulation, reference.solution.circulation, rtol=0, atol=1e-10)
        tree = truncated(4, 9, velocity_backend="tree", tree=TreeSettings(theta=0.2, order=2, leaf_size=8))[9]
        assert_allclose(tree.solution.force, reference.solution.force, rtol=2e-3)

    def test_rotor_rings_are_not_truncated(self):
        keep, steps = 2, 6
        wing, polar = Wing.ellipse(8), SectionPolar.linear()
        backend = AeroBackend("vsm_free_wake", wake=config(None, steps, retained_rows=keep))
        state = backend.start(wing, polar, WIND)
        for index in range(1, steps+1):
            rings = state.rings if state.rings is not None else RingWake()
            nodes = ring_nodes([1.5, 0.4, 0.2], [-1.0, 0.0, 0.0], 0.6, 8)
            state = replace(state, rings=rings.shed(ShedRing(nodes, ring_edges(8), 2.5, state.time_s, 0.0, 0.1)))
            state = backend.advance(state, polar, WIND, next_time_s=index*DT)
        self.assertEqual((len(state.free_rows), state.rings.count), (keep, steps*8))   # wing rows capped, every ring kept

    def test_the_archive_and_the_other_methods_refuse_a_truncated_wake(self):
        last = truncated(3, 5)[5]
        with self.assertRaises(ValueError):                                  # the archive stores one row per step
            AeroWriter._state_attributes(None, last)
        limited = config(None, retained_rows=3)
        with self.assertRaises(ValueError):
            validate_lifting_surface_wake(LiftingSurfaceConfig(), limited, True)
        with self.assertRaises(ValueError):
            validate_lifting_surface_wake(LiftingSurfaceConfig(), config(3), True)
        with self.assertRaises(ValueError):
            validate_straight_ll(limited, LLStraightConfig())


class RingTests(unittest.TestCase):
    def test_ring_nodes_stay_free_when_the_wing_rows_are_limited(self):
        limit, steps = 2, 6
        wing, polar = Wing.ellipse(8), SectionPolar.linear()
        plain = AeroBackend("vsm_free_wake", wake=config(None, steps+1))
        state = plain.start(wing, polar, WIND)
        for index in range(1, steps+1):
            rings = state.rings if state.rings is not None else RingWake()
            nodes = ring_nodes([1.5, 0.4, 0.2], [-1.0, 0.0, 0.0], 0.6, 8)
            state = replace(state, rings=rings.shed(ShedRing(nodes, ring_edges(8), 2.5, state.time_s, 0.0, 0.1)))
            state = plain.advance(state, polar, WIND, next_time_s=index*DT)
        # One more step from the same state, once without and once with the limit.
        limited = AeroBackend("vsm_free_wake", wake=config(limit, steps+1))
        after_plain = plain.advance(state, polar, WIND, next_time_s=(steps+1)*DT)
        after_limited = limited.advance(replace(state, config=limited.wake), polar, WIND, next_time_s=(steps+1)*DT)
        self.assertEqual(after_limited.rings.count, steps*8)
        # Ring nodes and the free rows inside the limit: the same induced velocity from the same state, bit for bit.
        self.assertTrue(np.array_equal(after_limited.rings.nodes, after_plain.rings.nodes))
        self.assertTrue(np.array_equal(after_limited.free_rows[:limit], after_plain.free_rows[:limit]))
        # The rings did move with more than the ambient wind.
        self.assertGreater(np.abs(after_plain.rings.nodes-(state.rings.nodes+WIND*DT)).max(), 1e-4)
        # Wing rows beyond the limit drifted with the wind only.
        assert_allclose(after_limited.free_rows[limit:], state.free_rows[limit-1:]+WIND*DT, rtol=0, atol=1e-14)
        self.assertGreater(np.abs(after_plain.free_rows[limit:]-after_limited.free_rows[limit:]).max(), 1e-6)


if __name__ == "__main__":
    unittest.main()
