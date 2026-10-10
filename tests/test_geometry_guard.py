"""Analytic contact counterexamples for a stationary camber-surface guard."""

from dataclasses import replace
import json
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from openfawe.aero.aircraft import WingAssembly, load_aircraft
from openfawe.aero.geometry_guard import StaticSurfaceGuard, WakeGeometryViolation
from openfawe.aero.lifting_surface import LiftingSurfaceConfig, build_lifting_surface_mesh
from openfawe.aero.wing import Wing


def rectangle():
    leading = np.array([[0., 0., 0.], [0., 1., 0.]])
    return Wing(leading, leading+[1., 0., 0.])


def guard(wing=None, panels=2, **kwargs):
    wing = rectangle() if wing is None else wing
    mesh = build_lifting_surface_mesh(wing, LiftingSurfaceConfig(chordwise_panels=panels))
    return StaticSurfaceGuard.from_mesh(wing, mesh, **kwargs)


class GeometryGuardTests(unittest.TestCase):
    def test_downstream_birth_and_exact_initial_field_are_permitted(self):
        check = guard()
        wing = check.wing
        old = wing.trailing_edge[None]
        initial = check.check_rows(old, old, wing, allow_initial_te=True, stage='initial')
        self.assertTrue(initial.acceptable)
        self.assertGreater(initial.allowed_te_contacts, 0)
        new = old+[.2, 0., 0.]
        report = check.check_rows(old, new, wing, stage='heun_predictor', time_s=.1)
        self.assertTrue(report.acceptable)
        self.assertEqual(report.checked_node_paths, 2)
        self.assertEqual(report.checked_segments, 3)
        self.assertGreater(report.allowed_te_contacts, 0)
        json.dumps(report.to_dict(), allow_nan=False)
        with self.assertRaises(ValueError):
            check.check_rows(old, new, wing, allow_initial_te=True)

    def test_coplanar_inward_release_and_stationary_positive_time_are_rejected(self):
        check = guard()
        old = check.wing.trailing_edge[None]
        for displacement in ([-.2, 0., 0.], [0., 0., 0.]):
            with self.subTest(displacement=displacement), self.assertRaises(WakeGeometryViolation) as caught:
                check.check_rows(old, old+displacement, check.wing)
            self.assertTrue(any(c.kind == 'coplanar_contact' for c in caught.exception.report.contacts))

    def test_node_path_detects_crossing_with_clear_final_segments(self):
        check = guard()
        old = np.array([[[.4, .25, -.2], [.4, .75, -.2]]])
        new = old+[0., 0., .4]
        with self.assertRaises(WakeGeometryViolation) as caught:
            check.check_rows(old, new, check.wing, stage='rk4_k2', time_s=.2)
        paths = [c for c in caught.exception.report.contacts if c.query_kind == 'node_path']
        self.assertEqual(len(paths), 2)
        for contact in paths:
            self.assertAlmostEqual(contact.parameter_start, .5)
            self.assertAlmostEqual(contact.point_m[2], 0.)
        self.assertEqual(caught.exception.geometry_report['stage'], 'rk4_k2')
        json.dumps(caught.exception.geometry_report, allow_nan=False)

    def test_candidate_span_crossing_endpoint_and_edge_contacts(self):
        check = guard()
        for row in ([[.5, .25, -.2], [.5, .75, .2]],
                    [[.5, .25, 0.], [.5, .75, .2]],
                    [[0., .25, 0.], [0., .75, .2]]):
            candidate = np.array([row])
            with self.subTest(row=row), self.assertRaises(WakeGeometryViolation) as caught:
                check.check_rows(candidate, candidate, check.wing)
            self.assertTrue(any(c.query_kind == 'span_segment' for c in caught.exception.report.contacts))

    def test_parallel_positive_and_negative_clearance(self):
        check = guard()
        old = check.wing.trailing_edge[None]
        for height in (-.2, .2):
            new = np.array([[[.4, .25, height], [.4, .75, height]]])
            result = check.check_rows(old, new, check.wing)
            self.assertTrue(result.acceptable)

    def test_old_material_node_returning_to_te_has_no_birth_exemption(self):
        check = guard()
        te = check.wing.trailing_edge
        old = np.array([te, te+[.5, 0., .1]])
        new = np.array([te+[.2, 0., .1], te])
        with self.assertRaises(WakeGeometryViolation) as caught:
            check.check_rows(old, new, check.wing)
        self.assertTrue(any(c.query_kind == 'node_path' and c.query_id[0] == 1
                            for c in caught.exception.report.contacts))

    def test_same_position_on_other_surface_does_not_inherit_exemption(self):
        base = rectangle()
        wing = WingAssembly((base, base), ('one', 'two'))
        check = guard(wing)
        old = wing.trailing_edge[None]
        with self.assertRaises(WakeGeometryViolation) as caught:
            check.check_rows(old, old+[.2, 0., .1], wing)
        self.assertTrue(any(c.source_surface != c.target_surface for c in caught.exception.report.contacts))

    def test_main_wake_crosses_other_named_surface(self):
        main = rectangle()
        tail = Wing(main.leading_edge+[1.5, 0., 0.], main.trailing_edge+[1.5, 0., 0.])
        wing = WingAssembly((main, tail), ('main', 'tail'))
        check = guard(wing)
        old = wing.trailing_edge[None]
        new = old.copy()
        new[0, :2] += [1., 0., 0.]
        new[0, 2:] += [.2, 0., .2]
        with self.assertRaises(WakeGeometryViolation) as caught:
            check.check_rows(old, new, wing)
        self.assertTrue(any(c.source_surface == 'main' and c.target_surface == 'tail'
                            for c in caught.exception.report.contacts))

    def test_disconnected_surfaces_do_not_gain_invented_span_edges(self):
        main = rectangle()
        remote = Wing(main.leading_edge+[0., 4., 0.], main.trailing_edge+[0., 4., 0.])
        wing = WingAssembly((main, remote), ('first', 'remote'))
        check = guard(wing)
        old = wing.trailing_edge[None]
        report = check.check_rows(old, old+[.2, 0., .1], wing)
        self.assertEqual(report.checked_segments, 6)
        self.assertEqual(report.checked_node_paths, 4)

    def test_true_camber_facets_instead_of_flat_chords(self):
        base = rectangle()
        offsets = np.zeros((2, 3, 3))
        offsets[:, 1, 2] = .2
        wing = Wing(base.leading_edge, base.trailing_edge, [0., .5, 1.], offsets)
        check = guard(wing)
        old = wing.trailing_edge[None]
        new = np.array([[[.5, .25, .2], [.5, .75, .2]]])
        with self.assertRaises(WakeGeometryViolation) as caught:
            check.check_rows(old, new, wing)
        self.assertTrue(any(abs(c.point_m[2]-.2) < 1e-14 for c in caught.exception.report.contacts))
        self.assertTrue(guard(base).check_rows(old, new, base).acceptable)
        self.assertEqual(check.triangles[:, :, 2].max(), .2)

    def test_pure_inputs_and_budget_counts_all_violations(self):
        check = guard(max_events=1)
        old = check.wing.trailing_edge[None].copy()
        new = old+[-.2, 0., 0.]
        before_old, before_new = old.copy(), new.copy()
        with self.assertRaises(WakeGeometryViolation) as caught:
            check.check_rows(old, new, check.wing)
        assert_array_equal(old, before_old)
        assert_array_equal(new, before_new)
        self.assertEqual(len(caught.exception.report.contacts), 1)
        self.assertGreater(caught.exception.report.violation_count, 1)
        self.assertFalse(check.triangles.flags.writeable)
        with self.assertRaises(ValueError):
            check.triangles[0, 0, 0] += 1

    def test_motion_camber_and_source_topology_changes_are_rejected(self):
        check = guard()
        old = check.wing.trailing_edge[None]
        moved = Wing(check.wing.leading_edge+[.1, 0., 0.], check.wing.trailing_edge+[.1, 0., 0.])
        with self.assertRaises(NotImplementedError):
            check.check_rows(old, old+[.2, 0., .1], moved)
        identical = Wing(check.wing.leading_edge, check.wing.trailing_edge)
        self.assertTrue(check.check_rows(old, old+[.2, 0., .1], identical).acceptable)
        offsets = np.zeros((2, 3, 3)); offsets[:, 1, 2] = .1
        cambered = Wing(identical.leading_edge, identical.trailing_edge, [0., .5, 1.], offsets)
        with self.assertRaises(NotImplementedError):
            check.check_rows(old, old+[.2, 0., .1], cambered)

    def test_explicit_triangles_have_no_implicit_te_exemptions(self):
        original = guard()
        external = StaticSurfaceGuard.from_triangles(original.wing, original.triangles,
                                                     surface_ids=original.surface_ids)
        old = original.wing.trailing_edge[None]
        with self.assertRaises(WakeGeometryViolation):
            external.check_rows(old, old+[.2, 0., .1], original.wing)
        matched = StaticSurfaceGuard.from_triangles(original.wing, original.triangles,
                   surface_ids=original.surface_ids, te_node_ids=original.te_node_ids)
        self.assertTrue(matched.check_rows(old, old+[.2, 0., .1], original.wing).acceptable)
        self.assertEqual(matched.geometry_sha256, original.geometry_sha256)
        numpy_ids = tuple(tuple(np.int64(node) for node in nodes) for nodes in original.te_node_ids)
        portable = StaticSurfaceGuard.from_triangles(original.wing, original.triangles,
                   surface_ids=original.surface_ids, te_node_ids=numpy_ids)
        self.assertEqual(portable.geometry_sha256, original.geometry_sha256)

    def test_scale_rotation_translation_and_direction_independent_contacts(self):
        original = rectangle()
        row = np.array([[[.4, .25, -.2], [.4, .75, -.2]]])
        candidate = row+[0., 0., .4]
        angle = .63
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        for scale in (1e-6, 1., 1e6):
            shift = scale*np.array([3., -4., 2.])
            wing = Wing(scale*original.leading_edge@rotation.T+shift,
                        scale*original.trailing_edge@rotation.T+shift)
            check = guard(wing)
            old = scale*row@rotation.T+shift
            new = scale*candidate@rotation.T+shift
            for start, end in ((old, new), (new, old)):
                with self.assertRaises(WakeGeometryViolation) as caught:
                    check.check_rows(start, end, wing)
                paths = [c for c in caught.exception.report.contacts if c.query_kind == 'node_path']
                self.assertEqual(len(paths), 2)
                assert_allclose([c.parameter_start for c in paths], .5, atol=2e-14)

    def test_input_validation_and_mesh_is_physical(self):
        wing = rectangle()
        mesh = build_lifting_surface_mesh(wing)
        with self.assertRaises(ValueError):
            StaticSurfaceGuard.from_mesh(wing, replace(mesh, vertices=mesh.vortex_rows.reshape(-1, 3)))
        for triangles in (np.zeros((1, 3, 3)), np.zeros((0, 3, 3)),
                          np.full((1, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                StaticSurfaceGuard.from_triangles(wing, triangles)
        check = guard(wing)
        old = wing.trailing_edge[None]
        for bad in (np.zeros((0, 2, 3)), np.zeros((1, 1, 3)), np.full((1, 2, 3), np.inf)):
            with self.assertRaises(ValueError):
                check.check_rows(old, bad, wing)
        with self.assertRaises(ValueError):
            check.check_rows(old, old, wing, time_s=[0.])
        with self.assertRaises(ValueError):
            check.check_rows(old, old, wing, allow_initial_te=1)

    def test_original_kite_guards_preserve_source_camber_and_surfaces(self):
        root = Path(__file__).resolve().parents[1]
        for name, panels, surfaces in (
            ('cases/aircraft/v3_cad_camber_integral.json', 36, ('V3-wing',)),
            ('cases/qualification/m600_makani_reconstructed.json', 49,
             ('Wing', 'Horizontal tail', 'Vertical tail'))):
            definition = load_aircraft(root/name)
            check = guard(definition.wing, panels=16)
            self.assertEqual(len(check.triangles), 2*16*panels)
            self.assertEqual(set(check.surface_ids), set(surfaces))
            self.assertIsNotNone(definition.wing.camber_x)
            self.assertTrue(all(contact in surfaces for contact in check._node_surfaces))
            self.assertEqual(len(check.geometry_sha256), 64)
            if surfaces == ('V3-wing',):
                mesh = build_lifting_surface_mesh(definition.wing,
                         LiftingSurfaceConfig(chordwise_panels=16))
                old_te = mesh.vortex_rows[-1:]
                self.assertTrue(check.check_rows(old_te, old_te, definition.wing,
                                                allow_initial_te=True).acceptable)


if __name__ == '__main__':
    unittest.main()
