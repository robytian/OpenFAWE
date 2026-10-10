"""Convection-only bound cores: independent fields, ownership and archive binding."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import numpy as np

from openfawe.aero import lifting_surface as ls
from openfawe.aero.aircraft import WingAssembly
from openfawe.aero.methods import AeroBackend, AeroMethod
from openfawe.aero.vortex import CoreModel, segments_velocity
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing
from openfawe.aero_archive import AeroReader, AeroWriter


def assembly():
    """Two separated tapered surfaces with explicitly known node chords."""
    main = Wing([[0., -1., 0.], [0., 1., 0.]],
                [[1., -1., 0.], [2., 1., 0.]])
    tail = Wing([[2.2, -.5, .2], [2.2, .5, .2]],
                [[2.7, -.5, .2], [3.2, .5, .2]])
    return WingAssembly((main, tail), ("wing", "horizontal_tail"))


def settings(basis, mode="all_bound_chord"):
    return ls.LiftingSurfaceConfig(chordwise_panels=3, wake_basis=basis,
        wake_release="current", bound_convection_core=mode)


def bits(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                  np.asarray(expected).view(np.uint64))


def integrated_velocity(point, start, end, gamma, radius):
    """Gauss integral of the Rosenhead law; no closed segment formula."""
    x, weight = np.polynomial.legendre.leggauss(192)
    delta = end-start
    offset = point-(start+((x+1)/2)[:, None]*delta)
    value = np.cross(delta, offset)/(np.sum(offset*offset, axis=1)+radius**2)[:, None]**1.5
    return gamma/(4*np.pi)*np.sum((weight/2)[:, None]*value, axis=0)


class BoundConvectionFixedTests(unittest.TestCase):
    def test_fixed_state_aic_loads_and_probes_are_bitwise_unchanged(self):
        wing, wind = assembly(), np.array([2., 0., .12])
        for basis in ("constant", "linear_potential"):
            cfg = WakeConfig(dt_s=.01, core_radius_m=.07, max_rows=3)
            old = ls.start_lifting_surface(wing, cfg, wind,
                settings=settings(basis, "legacy_zero"), free_wake=True)
            new = ls.start_lifting_surface(wing, cfg, wind,
                settings=settings(basis), free_wake=True)
            for name in ("lattice_circulation", "lattice_force", "lattice_couple", "force"):
                bits(getattr(old.solution, name), getattr(new.solution, name))
            old = ls.advance_lifting_surface(old, cfg, wind)
            new = replace(old, settings=settings(basis))
            # Same accepted geometry/Gamma: only the declared stage-field policy differs.
            aic_old = ls.lifting_surface_influence(old.wing, old.mesh, cfg)
            aic_new = ls.lifting_surface_influence(new.wing, new.mesh, cfg)
            bits(aic_old, aic_new)
            for a, b in zip(ls.lifting_surface_network(old), ls.lifting_surface_network(new)):
                np.testing.assert_array_equal(a, b)
            bits(ls.lifting_surface_edge_radii(old), ls.lifting_surface_edge_radii(new))
            probes = np.array([[2.4, .1, .3], [3., .7, .6], [.6, .4, .3]])
            bits(ls.lifting_surface_induced_velocity(old, probes),
                 ls.lifting_surface_induced_velocity(new, probes))
            loads = []
            for state in (old, new):
                loads.append(ls._loads(state.wing, state.mesh, state.solution.lattice_circulation,
                    state.free_rows, state.wake_circulation, state.cores, cfg, wind, (0.,0.,0.),
                    state.time_s, state.solution.lattice_gamma_rate,
                    nodal_phi=state.wake_node_potential_m2_s, settings=state.settings))
            for a, b in zip(*loads):
                bits(a, b)
            self.assertGreater(np.max(np.abs(ls._convect(new, cfg, wind)-ls._convect(old, cfg, wind))), 1e-12)

    def test_non_te_tail_edge_field_matches_independent_integral(self):
        wing = assembly()
        cfg = WakeConfig(core_model=CoreModel.ROSENHEAD_MOORE, core_radius_m=.07)
        for basis in ("constant", "linear_potential"):
            state = ls.start_lifting_surface(wing, cfg, [2.,0.,.12],
                settings=settings(basis), free_wake=True)
            if basis == "linear_potential":
                network, radii, _ = ls._linear_convection_network(
                    state, cfg, state.mesh.vortex_rows[-1].copy(), 0.)
            else:
                network = ls.lifting_surface_network(state)
                radii = ls._bound_convection_radii(wing, network[1],
                    ls.lifting_surface_edge_radii(state), 4*wing.node_count, state.settings, cfg)
            nodes, edges, gamma = network
            # First-row spanwise filament of the horizontal tail: no TE endpoint.
            wanted = tuple(sorted((2, 3)))
            index = next(i for i, edge in enumerate(edges) if tuple(sorted(edge)) == wanted)
            self.assertLess(max(edges[index]), 3*wing.node_count)
            self.assertAlmostEqual(radii[index], .075, places=14)
            point = nodes[edges[index]].mean(axis=0)+[0.,0.,.03]
            start, end = nodes[edges[index]]
            expected = integrated_velocity(point, start, end, gamma[index], .075)
            actual = segments_velocity([point], nodes, edges[index:index+1], gamma[index:index+1],
                model=cfg.core_model, core_radius=radii[index:index+1])[0]
            np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-13)
            ideal = integrated_velocity(point, start, end, gamma[index], 0.)
            self.assertGreater(np.linalg.norm(actual-ideal), .1*np.linalg.norm(ideal))

    def test_local_chord_on_true_bound_edges_preserves_existing_material_radii(self):
        wing, wind = assembly(), [2.,0.,.12]
        # Values are declared geometry, not obtained from the production radius helper.
        node_chords = np.array([1., 2., .5, 1.])
        for basis in ("constant", "linear_potential"):
            cfg = WakeConfig(dt_s=.01, core_radius_m=.07, core_evolution="age_stretch", max_rows=3)
            old = ls.start_lifting_surface(wing, cfg, wind,
                settings=settings(basis, "legacy_zero"), free_wake=True)
            for _ in range(2):
                old = ls.advance_lifting_surface(old, cfg, wind)
            new = replace(old, settings=settings(basis))
            if basis == "linear_potential":
                moving = np.concatenate((old.mesh.vortex_rows[-1:], old.free_rows)).reshape(-1,3)
                network, before, _ = ls._linear_convection_network(old, cfg, moving, 0.)
                other, after, _ = ls._linear_convection_network(new, cfg, moving, 0.)
                for a, b in zip(network, other):
                    np.testing.assert_array_equal(a, b)
            else:
                network = ls.lifting_surface_network(old)
                before = ls.lifting_surface_edge_radii(old)
                after = ls._bound_convection_radii(wing, network[1], before,
                    4*wing.node_count, new.settings, cfg)
            edges = network[1]
            true_bound = np.all(edges < 4*wing.node_count, axis=1) & ~np.all(edges//wing.node_count == 3, axis=1)
            for edge, radius in zip(edges[true_bound], after[true_bound]):
                expected = .05*(node_chords[edge[0] % wing.node_count]+node_chords[edge[1] % wing.node_count])
                self.assertAlmostEqual(radius, expected, places=14)
            material = np.any(edges >= 4*wing.node_count, axis=1)
            self.assertTrue(material.any())
            bits(after[material], before[material])
            bits(ls.lifting_surface_edge_radii(new), ls.lifting_surface_edge_radii(old))

    def test_three_integrators_both_bases_and_backend_configuration_ownership(self):
        wing, wind = assembly(), [2.,0.,.12]
        for basis in ("constant", "linear_potential"):
            for integrator in ("euler", "heun", "rk4"):
                cfg = WakeConfig(dt_s=.005, max_rows=3, convection_integrator=integrator)
                backend = AeroBackend(AeroMethod.VLM_FREE_WAKE, wake=cfg, lifting_surface=settings(basis))
                state = backend.start(wing, None, wind)
                for step in (1, 2):
                    old_rows = state.free_rows.copy()
                    advanced = backend.advance(state, None, wind)
                    repeated = backend.advance(state, None, wind)
                    bits(state.free_rows, old_rows)
                    bits(advanced.free_rows, repeated.free_rows)
                    self.assertEqual(advanced.step, step)
                    self.assertTrue(np.isfinite(advanced.solution.force).all())
                    state = advanced
                changed = replace(backend, lifting_surface=replace(backend.lifting_surface,
                                                                  bound_convection_chord_fraction=.2))
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    changed.advance(state, None, wind)
                with self.assertRaisesRegex(ValueError, "settings changed"):
                    changed.induced_velocity(state, [[3.,.7,.6]])


class BoundConvectionArchiveTests(unittest.TestCase):
    def test_legacy_and_new_round_trip_both_bases(self):
        parent = Path(__file__).resolve().parents[1]/"local/bound-convection-tests"
        parent.mkdir(parents=True, exist_ok=True)
        root = parent/uuid4().hex
        root.mkdir()
        success = False
        try:
            for basis in ("constant", "linear_potential"):
                for mode in ("legacy_zero", "all_bound_chord"):
                    cfg = WakeConfig(dt_s=.005, max_rows=2, convection_integrator="heun")
                    state = ls.start_lifting_surface(assembly(), cfg, [2.,0.,.12],
                        settings=settings(basis, mode), free_wake=True)
                    path = root/f"{basis}-{mode}"
                    writer = AeroWriter(path, {"aero_method":"vlm_free_wake"})
                    try:
                        writer.append(state)
                        state = ls.advance_lifting_surface(state, cfg, [2.,0.,.12])
                        writer.append(state)
                        writer.finish()
                    except BaseException as error:
                        writer.fail(error)
                        raise
                    reader = AeroReader(path)
                    try:
                        frame = reader.frame(1)
                        bits(frame["free_rows"], state.free_rows)
                        bits(frame["lattice_panel_force_N"], state.solution.lattice_force.reshape(-1,3))
                        bits(frame["network_core_radius_m"], ls.lifting_surface_edge_radii(state))
                        declared = json.loads(frame["lattice_settings_json"])
                        if mode == "all_bound_chord":
                            self.assertEqual(frame["bound_convection_core"], mode)
                            self.assertEqual(reader.manifest["convection_bound_core_chord_fraction"], .1)
                            self.assertEqual(declared["bound_convection_chord_fraction"], .1)
                        else:
                            self.assertNotIn("bound_convection_core", declared)
                            self.assertNotIn("bound_convection_chord_fraction", declared)
                    finally:
                        reader.close()
                    if mode == "all_bound_chord":
                        tampered = root/f"{basis}-changed-bound-fraction"
                        shutil.copytree(path, tampered)
                        manifest_path = tampered/"manifest.json"
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        manifest["convection_bound_core_chord_fraction"] = .2
                        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                        with self.assertRaisesRegex(ValueError, "convection core"):
                            AeroReader(tampered)
            success = True
        finally:
            if success:
                if root.resolve().parent != parent.resolve():
                    raise ValueError("Unsafe test cleanup")
                shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
