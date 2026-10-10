"""Nonzero-camber source-moment identity and exactly-once load correction."""

from dataclasses import replace
import json
from pathlib import Path
import unittest
import numpy as np

from openfawe.aero.aircraft import PanelPolars
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface)
from openfawe.aero.polar_coupling import SectionReferences, inverse_reference
from openfawe.aero.vortex import CoreModel
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar


CASES = ('straight', 'constant-prescribed', 'constant-free',
         'linear-prescribed', 'linear-free')


def evaluate_cambered_case(case, *, initial_cap='legacy_zero', core_model=CoreModel.NONE):
    """Return reproducible load identities; this is a manufactured-source test."""
    profile = json.loads((Path(__file__).parent / 'fixtures' /
        'polar_coupling_geometric_reference.json').read_text(encoding='utf-8'))
    le = np.array([[0., -20., 0.], [0., 0., 0.], [0., 20., 0.]])
    offsets = np.zeros((len(le), len(profile['x']), 3))
    offsets[:, :, 2] = profile['z']
    wing = Wing(le, le+[1., 0., 0.], profile['x'], offsets)
    linear, released = case.startswith('linear'), case != 'straight'
    settings = LiftingSurfaceConfig(8, chordwise_spacing='cosine',
        camber_normal='reconstructed_slope', wake_basis='linear_potential' if linear else 'constant',
        wake_release='current' if linear else 'previous',
        wake_quadrature='te_graded' if linear else 'gauss', wake_length_m=1e5,
        initial_cap=initial_cap)
    cfg = WakeConfig(dt_s=.02, free_wake=case.endswith('-free'),
        core_model=core_model, core_radius_m=.05)
    winds = [np.array([4., 0., .1]), np.array([4., 0., .12])]
    pure = [start_lifting_surface(wing, cfg, winds[0], settings=settings, free_wake=released)]
    if released:
        pure.append(advance_lifting_surface(pure[-1], cfg, winds[1]))
    reference = SectionReferences(wing, pure[0].mesh)
    strengths, _ = reference.response(np.zeros(wing.panels))
    coefficients = 2*strengths.sum(axis=1)
    alphas = []
    for state, wind in zip(pure, winds):
        physical = np.full(wing.panels, np.arctan2(wind[2], wind[0]))
        alphas.append(inverse_reference(coefficients,
            2*state.solution.circulation/(np.linalg.norm(wind)*reference.chord), physical)[0])
    tables = []
    for j in range(wing.panels):
        knots = np.unique(np.round(np.r_[np.linspace(-30., 30., 121),
            np.rad2deg(np.asarray(alphas)[:, j])], 12))
        angle = np.deg2rad(knots)
        cl = coefficients[j, 0]*np.cos(angle)+coefficients[j, 1]*np.sin(angle)
        cm = np.array([reference.moment(strengths, np.full(wing.panels, a))[j] for a in angle])
        tables.append(SectionPolar(knots, cl, np.zeros_like(knots), cm,
            name='Manufactured exact cambered grid reference'))
    matching = PanelPolars(tables)
    delta_cm = .03
    shifted = PanelPolars(tuple(replace(p, cm=p.cm+delta_cm) for p in tables))
    selected = replace(settings, polar_coupling='grid_section')
    coupled = [start_lifting_surface(wing, cfg, winds[0], settings=selected,
        free_wake=released, polar=matching)]
    changed = [start_lifting_surface(wing, cfg, winds[0], settings=selected,
        free_wake=released, polar=shifted)]
    if released:
        coupled.append(advance_lifting_surface(coupled[-1], cfg, winds[1], polar=matching))
        changed.append(advance_lifting_surface(changed[-1], cfg, winds[1], polar=shifted))
    records = []
    for a, b, c in zip(pure, coupled, changed):
        closure = b.solution.polar_coupling
        expected_strip = (.5*cfg.density_kg_m3*closure.speed_m_s**2*closure.area_m2*
            closure.chord_m*delta_cm)[:, None]*closure.span
        expected = expected_strip.sum(axis=0)
        max_abs = lambda values: float(np.max(np.abs(values)))
        records.append(dict(case=case, initial_cap=initial_cap, core_model=core_model.name,
            step=a.step, time_s=a.time_s,
            minimum_abs_reference_cm=float(np.min(np.abs(closure.potential_cm))),
            virtual_rotation_max_rad=max_abs(closure.delta_rad),
            matching_gamma_error_m2_s=max_abs(b.solution.lattice_circulation-a.solution.lattice_circulation),
            matching_force_error_N=max_abs(b.solution.lattice_force-a.solution.lattice_force),
            matching_moment_error_N_m=max_abs(b.solution.moment(wing)-a.solution.moment(wing)),
            matching_correction_couple_max_N_m=max_abs(closure.correction_couple_N_m),
            shifted_gamma_error_m2_s=max_abs(c.solution.lattice_circulation-b.solution.lattice_circulation),
            shifted_force_error_N=max_abs(c.solution.lattice_force-b.solution.lattice_force),
            shifted_moment_error_N_m=max_abs(c.solution.moment(wing)-b.solution.moment(wing)-expected),
            shifted_strip_couple_error_N_m=max_abs(c.solution.polar_coupling.correction_couple_N_m.sum(axis=0)-expected_strip),
            expected_delta_moment_N_m=expected.tolist(),
            reference_cm=closure.potential_cm.tolist(),
            gamma_rate_max_m2_s2=max_abs(a.solution.lattice_gamma_rate)))
    return records


class CamberedPolarLoadTests(unittest.TestCase):
    def test_regularized_initial_cap_preserves_cambered_moment_identity(self):
        for case in CASES[1:]:
            for row in evaluate_cambered_case(case, initial_cap='wake_core', core_model=CoreModel.LAMB_OSEEN):
                with self.subTest(case=case, step=row['step']):
                    self.assertGreater(row['minimum_abs_reference_cm'], .01)
                    self.assertLess(row['virtual_rotation_max_rad'], 2e-12)
                    self.assertLess(row['matching_gamma_error_m2_s'], 3e-11)
                    self.assertLess(row['matching_force_error_N'], 3e-9)
                    self.assertLess(row['matching_moment_error_N_m'], 3e-9)
                    self.assertLess(row['matching_correction_couple_max_N_m'], 3e-9)
                    self.assertLess(row['shifted_force_error_N'], 3e-9)
                    self.assertLess(row['shifted_moment_error_N_m'], 3e-9)

    def test_matching_nonzero_camber_moment_preserves_potential_loads(self):
        for case in CASES:
            for row in evaluate_cambered_case(case):
                with self.subTest(case=case, step=row['step']):
                    self.assertGreater(row['minimum_abs_reference_cm'], .01)
                    self.assertLess(row['virtual_rotation_max_rad'], 2e-12)
                    self.assertLess(row['matching_gamma_error_m2_s'], 3e-11)
                    self.assertLess(row['matching_force_error_N'], 3e-9)
                    self.assertLess(row['matching_moment_error_N_m'], 3e-9)
                    self.assertLess(row['matching_correction_couple_max_N_m'], 3e-9)

    def test_delta_source_moment_is_added_exactly_once(self):
        for case in CASES:
            for row in evaluate_cambered_case(case):
                with self.subTest(case=case, step=row['step']):
                    self.assertLess(row['shifted_gamma_error_m2_s'], 3e-11)
                    self.assertLess(row['shifted_force_error_N'], 3e-9)
                    self.assertLess(row['shifted_moment_error_N_m'], 3e-9)
                    self.assertLess(row['shifted_strip_couple_error_N_m'], 3e-9)
                    self.assertGreater(np.linalg.norm(row['expected_delta_moment_N_m']), 1.)
                    if row['step']:
                        self.assertGreater(row['gamma_rate_max_m2_s2'], .01)


if __name__ == '__main__':
    unittest.main()
