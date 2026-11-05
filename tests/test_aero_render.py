"""Lit 3D rendering (aero_render) and the fuselage part of the ParaView export.

The rendering tests need pyvista and an off-screen VTK context; they are
skipped where those are unavailable. The export test needs only NumPy.
"""
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import numpy as np

from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero_archive import AeroReader, AeroWriter
from openfawe.aero_vtk import export_paraview

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None


def small_run(root, method="vsm_free_wake", steps=3):
    y = np.linspace(-1., 1., 5)
    le = np.column_stack((np.zeros(5), y, np.zeros(5)))
    wing = Wing(le, le + [1., 0., 0.])
    polar, wind = SectionPolar.linear(), [3., 0., .1]
    backend = AeroBackend(method, wake=WakeConfig(dt_s=.05, max_rows=steps + 1))
    writer = AeroWriter(root, {"aero_method": method, "model": "test-m600"})
    try:
        state = backend.start(wing, polar, wind)
        writer.append(state)
        for index in range(1, steps + 1):
            state = backend.advance(state, polar, wind, next_time_s=index * .05)
            writer.append(state)
        writer.finish()
    except BaseException as error:
        writer.fail(error)
        raise
    return root


def fuselage_file(path):
    # A closed 8 x 6 tube of quads along x, like the OF-0006-body20260930 meshes.
    stations = np.linspace(-1.5, 1.5, 8)
    angles = np.linspace(0., 2 * np.pi, 6, endpoint=False)
    nodes = np.array([[x, .2 * np.cos(a), .2 * np.sin(a)] for x in stations for a in angles])
    quads = [[i * 6 + j, i * 6 + (j + 1) % 6, (i + 1) * 6 + (j + 1) % 6, (i + 1) * 6 + j]
             for i in range(7) for j in range(6)]
    np.savez(path, nodes=nodes, quads=np.asarray(quads, dtype=np.int64))
    return path


class ExportWithFuselage(unittest.TestCase):
    def test_pvd_has_three_parts_and_static_fuselage(self):
        parent = Path(__file__).resolve().parents[1] / "local" / "render-tests"
        root = parent / uuid4().hex
        root.mkdir(parents=True)
        try:
            run = small_run(root / "run")
            mesh = fuselage_file(root / "fuselage.npz")
            pvd = export_paraview(run, root / "vtk", fuselage_mesh=mesh)
            tree = ET.parse(pvd)
            sets = tree.getroot().findall("./Collection/DataSet")
            parts = sorted({d.get("part") for d in sets})
            self.assertEqual(parts, ["0", "1", "2"])
            fuselage_sets = [d for d in sets if d.get("part") == "2"]
            self.assertEqual({d.get("file") for d in fuselage_sets}, {"fuselage.vtp"})
            self.assertEqual(len(fuselage_sets), len([d for d in sets if d.get("part") == "0"]))
            piece = ET.parse(root / "vtk" / "fuselage.vtp").getroot().find("./PolyData/Piece")
            self.assertEqual(piece.get("NumberOfPolys"), "42")
            names = [a.get("Name") for a in piece.find("FieldData")]
            self.assertIn("fuselage_0_display_only_1_in_computation", names)
            without = export_paraview(run, root / "vtk-plain")
            parts = sorted({d.get("part") for d in ET.parse(without).getroot().findall("./Collection/DataSet")})
            self.assertEqual(parts, ["0", "1"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


@unittest.skipUnless(HAS_PYVISTA, "pyvista not installed")
class OffScreenRender(unittest.TestCase):
    def test_render_state_and_replay_scene(self):
        import pyvista as pv
        pv.OFF_SCREEN = True
        from openfawe import aero_render as ar
        parent = Path(__file__).resolve().parents[1] / "local" / "render-tests"
        root = parent / uuid4().hex
        root.mkdir(parents=True)
        try:
            run = small_run(root / "run")
            mesh = fuselage_file(root / "fuselage.npz")
            reader = AeroReader(run)
            try:
                frame = reader.frame(reader.count - 1)
                from openfawe.aero_vtk import frame_network
                nodes, edges, gamma = frame_network(frame)
                try:
                    polys = ar.render_state(root / "state.png", nodes=nodes, edges=edges, gamma=gamma,
                                            leading_edge=frame["leading_edge"], trailing_edge=frame["trailing_edge"],
                                            panel_edges=np.column_stack([np.arange(4), np.arange(1, 5)]),
                                            fuselage_path=mesh, view="m600", title="test", subtitle="α = 0°",
                                            size=(400, 300))
                except RuntimeError as error:  # no usable off-screen context on this machine
                    self.skipTest(f"no off-screen VTK context: {error}")
                self.assertTrue((root / "state.png").stat().st_size > 1000)
                self.assertEqual(polys["wake"].n_cells, len(edges))
                self.assertEqual(polys["fuselage"].n_cells, 42)
                plotter = pv.Plotter(off_screen=True, window_size=[400, 300])
                scene = ar.ReplayScene(plotter, reader, fuselage_path=mesh)
                for index in range(reader.count):
                    wake = scene.show_frame(index)
                    self.assertEqual(wake.n_cells, len(frame_network(reader.frame(index))[1]))
                scene.set_fuselage_visible(False)
                plotter.screenshot(str(root / "replay.png"))
                plotter.close()
            finally:
                reader.close()
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
