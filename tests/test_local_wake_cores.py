"""Local-chord material birth cores, independent AIC, and archive continuation."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import unittest
from uuid import uuid4

import numpy as np

from openfawe.aero.cores import (advance_cores, birth_radii, core_identity,
                                edge_radii, initial_cores, check_cores)
from openfawe.aero.lifting_surface import LiftingSurfaceConfig
from openfawe.aero.methods import AeroBackend
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.vsm import geometry, vsm_self_correction
from openfawe.aero.wake import WakeConfig, current_influence, pack_rings, wake_network
from openfawe.aero.wing import Wing, SectionPolar, wake_config_metadata
from openfawe.aero_archive import (AeroWriter, AeroReader, wake_cores_from_frame,
                                  _validate_local_core_history, _validate_wake_frame,
                                  _validate_lattice_frame)


METHODS = ('free_wake', 'vsm_free_wake', 'vlm_free_wake')


def tapered_wing():
    y = np.linspace(-2., 2., 5)
    le = np.column_stack((np.zeros(5), y, np.zeros(5)))
    chord = np.array([.6, 1., 1.2, 1., .6])
    return Wing(le, le+np.column_stack((chord, np.zeros((5, 2)))))


def backend(method, policy='constant', **kwargs):
    config = WakeConfig(dt_s=.05, core_radius_m=.02, max_rows=8,
                        wake_core_chord_fraction=.1, core_evolution=policy, **kwargs)
    return AeroBackend(method, wake=config, lifting_surface=LiftingSurfaceConfig(
        chordwise_panels=2, polar_coupling='grid_section'))


class LocalWakeCoreTests(unittest.TestCase):
    def test_birth_radius_uses_endpoint_chords_and_preserves_material_history(self):
        wing = tapered_wing()
        config = WakeConfig(wake_core_chord_fraction=.1)
        span, stream = birth_radii(wing, config)
        np.testing.assert_allclose(span, [.08, .11, .11, .08], rtol=1e-15)
        np.testing.assert_allclose(stream, [.06, .1, .12, .1, .06], rtol=1e-15)
        old = SimpleNamespace(wing=wing, free_rows=np.empty((0, 5, 3)),
                              cores=initial_cores(wing, config))
        rows = wing.trailing_edge[None]+[.2, 0., 0.]
        first = advance_cores(old, rows, config)
        np.testing.assert_array_equal(first.span[0], span)
        np.testing.assert_array_equal(first.stream[0], stream)
        # Changed current chord affects the new birth, not material already shed.
        changed = Wing(wing.leading_edge, wing.leading_edge+2*(wing.trailing_edge-wing.leading_edge))
        old = SimpleNamespace(wing=changed, free_rows=rows, cores=first)
        second = advance_cores(old, np.concatenate((changed.trailing_edge[None]+[.2, 0, 0], rows+[.2, 0, 0])), config)
        np.testing.assert_array_equal(second.span, np.vstack((2*span, span)))
        np.testing.assert_array_equal(second.stream, np.vstack((2*stream, stream)))
        with self.assertRaises(ValueError):
            first.span[0, 0] = 1.

    def test_local_age_analytic_squared_radius_and_stretch_invariant(self):
        wing = tapered_wing()
        config = WakeConfig(dt_s=.1, wake_core_chord_fraction=.1, core_evolution='age')
        span, stream = birth_radii(wing, config)
        old = SimpleNamespace(wing=wing, free_rows=np.empty((0, 5, 3)), cores=initial_cores(wing, config))
        for step in range(1, 4):
            rows = np.concatenate((wing.trailing_edge[None], old.free_rows))+[.3, 0., 0.]
            cores = advance_cores(old, rows, config)
            ages = np.arange(1, step+1)*config.dt_s
            rate = core_identity(config)[2]
            np.testing.assert_allclose(cores.span**2, span[None]**2+rate*ages[:, None], rtol=2e-15)
            np.testing.assert_allclose(cores.stream**2, stream[None]**2+rate*(ages[:, None]-.05), rtol=2e-15)
            old = SimpleNamespace(wing=wing, free_rows=rows, cores=cores)
        config = replace(config, core_evolution='age_stretch', eddy_viscosity_factor=0.)
        old = SimpleNamespace(wing=wing, free_rows=np.empty((0, 5, 3)), cores=initial_cores(wing, config))
        rows = wing.trailing_edge[None]+[.3, 0., 0.]
        rows[:, :, 1] *= 2
        cores = advance_cores(old, rows, config)
        a, b = wing.panel_edges.T
        l0 = np.linalg.norm(wing.trailing_edge[b]-wing.trailing_edge[a], axis=1)
        l1 = np.linalg.norm(rows[0, b]-rows[0, a], axis=1)
        np.testing.assert_allclose(cores.span[0]**2*l1, span**2*l0, rtol=2e-15)
        # An older stream edge doubles its length; its squared core halves.
        first_rows = wing.trailing_edge[None]+[.3, 0., 0.]
        first = advance_cores(old, first_rows, config)
        old = SimpleNamespace(wing=wing, free_rows=first_rows, cores=first)
        second_rows = np.stack((wing.trailing_edge+[.3, 0., 0.],
                                wing.trailing_edge+[.9, 0., 0.]))
        second = advance_cores(old, second_rows, config)
        np.testing.assert_allclose(second.stream[1]**2*.6, first.stream[0]**2*.3, rtol=2e-15)

    def test_default_metadata_and_scalar_age_arithmetic_are_unchanged(self):
        config = WakeConfig(core_evolution='age')
        self.assertNotIn('wake_core_chord_fraction', wake_config_metadata(config))
        self.assertIsNone(initial_cores(tapered_wing(), WakeConfig()))
        wing = tapered_wing()
        old = SimpleNamespace(wing=wing, free_rows=np.empty((0, 5, 3)), cores=initial_cores(wing, config))
        rows = wing.trailing_edge[None]+[.3, 0., 0.]
        cores = advance_cores(old, rows, config)
        rate = core_identity(config)[2]
        np.testing.assert_array_equal(cores.span[0], np.full(4, np.sqrt(config.core_radius_m**2+rate*config.dt_s)))
        np.testing.assert_array_equal(cores.stream[0], np.full(5, np.sqrt(config.core_radius_m**2+rate*config.dt_s/2)))

    def test_rejects_silent_configuration_changes_and_unsupported_backends(self):
        config = WakeConfig(wake_core_chord_fraction=.1)
        for method in ('ll_straight', 'vsm', 'vlm'):
            with self.subTest(method=method), self.assertRaises(ValueError):
                AeroBackend(method, wake=config)
        # Both settings must construct successfully before the intended backend rejection.
        linear = LiftingSurfaceConfig(wake_basis='linear_potential', wake_release='current')
        with self.assertRaisesRegex(ValueError, '^local material birth cores require a constant-basis released wake$'):
            AeroBackend('vlm_free_wake', wake=config, lifting_surface=linear)
        capped = LiftingSurfaceConfig(initial_cap='wake_core')
        with self.assertRaisesRegex(ValueError, '^local material birth cores currently require the legacy zero-core initial cap$'):
            AeroBackend('vlm_free_wake', wake=config, lifting_surface=capped)
        for value in (0., -1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                WakeConfig(wake_core_chord_fraction=value)
        with self.assertRaises(ValueError):
            WakeConfig(wake_core_chord_fraction=.1, core_model=CoreModel.NONE)
        with self.assertRaises(ValueError):
            check_cores(initial_cores(tapered_wing(), config), replace(config, wake_core_chord_fraction=.2))
        with self.assertRaises(ValueError):
            check_cores(None, config)

    def test_initial_bound_aic_and_solution_unchanged_for_all_free_backends(self):
        wing, polar, wind = tapered_wing(), SectionPolar.linear(), [5., 0., .1]
        for method in METHODS:
            with self.subTest(method=method):
                local = backend(method)
                fixed = replace(local, wake=replace(local.wake, wake_core_chord_fraction=None))
                a, b = local.start(wing, polar, wind), fixed.start(wing, polar, wind)
                np.testing.assert_array_equal(a.solution.circulation, b.solution.circulation)
                np.testing.assert_array_equal(a.solution.force, b.solution.force)

    def test_ll_shared_edge_operator_is_the_aggregated_network(self):
        wing, polar, wind = tapered_wing(), SectionPolar.linear(), [5., 0., .1]
        for policy in ('constant', 'age', 'age_stretch'):
            adapter = backend('free_wake', policy)
            state = adapter.start(wing, polar, wind)
            for _ in range(3):
                state = adapter.advance(state, polar, wind)
            nodes, edges, gamma = wake_network(state)
            radii = edge_radii(wing, edges, state.cores, adapter.wake.core_radius_m)
            full = segments_velocity(wing.cp, nodes, edges, gamma, model=adapter.wake.core_model,
                                     core_radius=radii, singularity='zero')
            aic = current_influence(wing, state.free_rows[0], state.cores, adapter.wake)
            nodes, edges, gamma = pack_rings(state.free_rows, state.older_circulation, wing.panel_edges)
            known = segments_velocity(wing.cp, nodes, edges, gamma, model=adapter.wake.core_model,
                core_radius=edge_radii(wing, edges+2*wing.node_count, state.cores, adapter.wake.core_radius_m))
            np.testing.assert_allclose(full, known+np.einsum('ijk,j->ik', aic, state.solution.circulation),
                                       rtol=1e-11, atol=1e-12)

    def test_vsm_local_shared_network_reconstructs_both_closure_and_load_fields(self):
        wing, polar, wind = tapered_wing(), SectionPolar.linear(), np.array([5., 0., .1])
        for policy in ('constant', 'age', 'age_stretch'):
            adapter = backend('vsm_free_wake', policy)
            state = adapter.start(wing, polar, wind)
            for _ in range(3):
                state = adapter.advance(state, polar, wind)
            nodes, edges, gamma = wake_network(state)
            radii = edge_radii(wing, edges, state.cores, adapter.wake.core_radius_m)
            cp, chord, normal, tangent, span = geometry(wing)
            induced = segments_velocity(cp, nodes, edges, gamma, model=adapter.wake.core_model, core_radius=radii)
            induced -= vsm_self_correction(wing, cp)*state.solution.circulation[:, None]
            velocity = wind+induced
            alpha = np.arctan2(np.sum(velocity*normal, axis=1), np.sum(velocity*tangent, axis=1))
            np.testing.assert_allclose(alpha, state.solution.polar_alpha_rad, rtol=1e-12, atol=1e-12)
            keep = ~np.all(edges < wing.node_count, axis=1)
            load = segments_velocity(wing.cp, nodes, edges[keep], gamma[keep],
                model=adapter.wake.core_model, core_radius=radii[keep])
            np.testing.assert_allclose(load, state.solution.induced_velocity, rtol=1e-12, atol=1e-12)
            balance = np.zeros(len(nodes))
            np.add.at(balance, edges[:, 0], gamma)
            np.add.at(balance, edges[:, 1], -gamma)
            np.testing.assert_allclose(balance, 0., atol=1e-12)

    def test_all_free_backends_archive_material_state_and_continue_exactly(self):
        parent = Path(__file__).resolve().parents[1]/'local'/'local-core-tests'
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex
        root.mkdir()
        try:
            wing, polar, wind = tapered_wing(), SectionPolar.linear(), [5., 0., .1]
            for method in METHODS:
                for policy in ('constant', 'age', 'age_stretch'):
                    with self.subTest(method=method, policy=policy):
                        adapter = backend(method, policy)
                        case = {'name': 'Local wake core archive verification', 'aero_method': method,
                                'panels': 4, 'aerodynamics': adapter.metadata()}
                        output = root/(method+'-'+policy)
                        writer = AeroWriter(output, case)
                        try:
                            state = adapter.start(wing, polar, wind)
                            writer.append(state)
                            for _ in range(2):
                                state = adapter.advance(state, polar, wind)
                                writer.append(state)
                            writer.finish()
                        except BaseException as error:
                            writer.fail(error)
                            raise
                        reader = AeroReader(output)
                        try:
                            frames = [reader.frame(i) for i in range(3)]
                        finally:
                            reader.close()
                        frame = frames[-1]
                        self.assertEqual(frame['wake_core_birth_policy'], 'local_chord_v1')
                        self.assertEqual(frame['core_cumulative_log_strain_status'], 'unrecorded')
                        restored = wake_cores_from_frame(frame)
                        np.testing.assert_array_equal(restored.span, state.cores.span)
                        np.testing.assert_array_equal(restored.stream, state.cores.stream)
                        resumed = replace(state, free_rows=frame['free_rows'], cores=restored)
                        a, b = adapter.advance(state, polar, wind), adapter.advance(resumed, polar, wind)
                        for left, right in ((a.cores.span, b.cores.span), (a.cores.stream, b.cores.stream),
                                            (a.free_rows, b.free_rows), (a.solution.force, b.solution.force)):
                            np.testing.assert_array_equal(left, right)
                        broken = dict(frame, wake_core_birth_chord_fraction=.2)
                        with self.assertRaises(ValueError):
                            wake_cores_from_frame(broken)
                        broken = dict(frame, span_core_radius_m=frame['span_core_radius_m']*1.01)
                        with self.assertRaises(ValueError):
                            _validate_local_core_history(broken, frames[-2])
                        broken = dict(frame, network_core_radius_m=frame['network_core_radius_m']+.001)
                        with self.assertRaises(ValueError):
                            (_validate_lattice_frame if method == 'vlm_free_wake' else _validate_wake_frame)(broken)
                        if policy == 'age_stretch':
                            first = frames[1]
                            i, j = wing.panel_edges.T
                            birth_length = np.linalg.norm(wing.trailing_edge[j]-wing.trailing_edge[i], axis=1)
                            length = np.linalg.norm(first['free_rows'][0, j]-first['free_rows'][0, i], axis=1)
                            print(json.dumps({'method': method, 'accepted_step': 1,
                                'birth_span_length_ratio': (length/birth_length).tolist(),
                                'cumulative_log_strain': 'unrecorded'}))
        finally:
            if root.resolve().parent != parent.resolve():
                raise ValueError('Refusing cleanup outside test workspace')
            shutil.rmtree(root)


if __name__ == '__main__':
    unittest.main()
