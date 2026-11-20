"""OF-0013 addendum 20 (G-0): the frozen program's CUDA setup and the spec's numba-cuda tree (no CUDA needed)."""
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path

from openfawe_studio import frozen_cuda

RELEASE = Path(__file__).resolve().parents[1] / "release"
_spec = importlib.util.spec_from_file_location("cuda_bundle", RELEASE / "cuda_bundle.py")
cuda_bundle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cuda_bundle)


class FrozenSetup(unittest.TestCase):
    def test_bundled_cuda_home_and_dll_folders(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            for parts in (("nvidia", "cuda_nvcc", "nvvm", "bin"), ("nvidia", "cuda_nvcc", "nvvm", "libdevice"),
                          ("nvidia", "cuda_runtime", "bin"), ("nvidia", "nvjitlink", "bin")):
                base.joinpath(*parts).mkdir(parents=True)
            environ = {"PATH": "C:\\Windows", "CUDA_PATH": "C:\\CUDA\\v11", "CUDA_HOME": "elsewhere"}
            added = []
            done = frozen_cuda.setup(base, environ, add_dll_directory=added.append)
            nvidia = base / "nvidia"
            self.assertEqual(environ["CUDA_HOME"], str(nvidia / "cuda_nvcc"))
            self.assertEqual(environ["CUDA_PATH"], "C:\\CUDA\\v11")          # the user's own is left alone
            folders = [str(nvidia / "cuda_runtime" / "bin"), str(nvidia / "nvjitlink" / "bin"),
                       str(nvidia / "cuda_nvcc" / "nvvm" / "bin")]                # cuda_nvrtc/bin is not there
            self.assertEqual(environ["PATH"], os.pathsep.join(folders + ["C:\\Windows"]))
            self.assertEqual(added, folders)
            self.assertEqual(done["dll_folders"], folders)

    def test_wheel_dll_folders(self):                                   # supplement 1
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            (base / "nvidia" / "cuda_nvcc" / "nvvm" / "bin").mkdir(parents=True)
            (base / "numba_cuda.libs").mkdir()
            environ = {}
            added = []
            frozen_cuda.setup(base, environ, add_dll_directory=added.append)
            self.assertEqual(added, [str(base / "nvidia" / "cuda_nvcc" / "nvvm" / "bin"), str(base / "numba_cuda.libs")])

    def test_nothing_without_bundled_cuda(self):
        with tempfile.TemporaryDirectory() as d:
            environ = {"PATH": "x"}
            self.assertEqual(frozen_cuda.setup(Path(d), environ, add_dll_directory=lambda f: None), {})
            self.assertEqual(environ, {"PATH": "x"})


class SpecTree(unittest.TestCase):
    def test_tree_goes_to_numba_cuda_without_tests(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "numba_cuda" / "numba" / "cuda"
            for rel in ("__init__.py", "cudadrv/driver.py", "cext/_dispatcher.cp312-win_amd64.pyd", "include/12/a.h",
                        "tests/test_x.py", "__pycache__/x.pyc", "cudadrv/__pycache__/y.pyc"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            spec = types.SimpleNamespace(submodule_search_locations=[str(Path(d) / "numba_cuda")], origin=None)
            self.assertEqual(cuda_bundle.numba_cuda_root(spec), root)
            datas = cuda_bundle.tree_datas(root)
            got = sorted((Path(src).relative_to(root).as_posix(), Path(dest).as_posix()) for src, dest in datas)
            self.assertEqual(got, [("__init__.py", "numba/cuda"), ("cext/_dispatcher.cp312-win_amd64.pyd", "numba/cuda/cext"),
                                   ("cudadrv/driver.py", "numba/cuda/cudadrv"), ("include/12/a.h", "numba/cuda/include/12")])

    def test_libs_binaries(self):                                       # supplement 1
        with tempfile.TemporaryDirectory() as d:
            site = Path(d)
            (site / "numba_cuda.libs").mkdir()
            (site / "numba_cuda.libs" / "msvcp140-abc.dll").write_bytes(b"x")
            (site / "numba_cuda.libs" / "readme.txt").write_bytes(b"x")
            got = cuda_bundle.libs_binaries([site / "numba_cuda", site / "cuda_bindings"])
            self.assertEqual(got, [(str(site / "numba_cuda.libs" / "msvcp140-abc.dll"), "numba_cuda.libs")])

    def test_names(self):
        self.assertTrue(cuda_bundle.is_numba_cuda_name("numba.cuda"))
        self.assertTrue(cuda_bundle.is_numba_cuda_name("numba.cuda.cudadrv.driver"))
        self.assertFalse(cuda_bundle.is_numba_cuda_name("numba.cudax"))
        self.assertFalse(cuda_bundle.is_numba_cuda_name("numba_cuda.numba.cuda"))


if __name__ == "__main__":
    unittest.main()


class Supplement2(unittest.TestCase):
    """OF-0013 addendum 20 supplement 2 (G-0'): the entry order, the GPU part's verdicts, the child probe, tracebacks."""

    def test_setup_before_freeze_support(self):
        text = (RELEASE / "studio_entry.py").read_text(encoding="utf-8")
        body = text.split('if __name__ == "__main__":', 1)[1]
        self.assertLess(body.index("frozen_cuda.setup()"), body.index("mp.freeze_support()"))

    def _part(self, driver, fail):
        from unittest import mock
        from openfawe_studio import selftest
        fake = types.ModuleType("openfawe.aero.vortex_cuda")
        fake.available = lambda: False
        fake.probe_error = lambda: "probe said no"
        patches = [mock.patch.object(selftest, "nvidia_driver", return_value=driver),
                   mock.patch.object(selftest, "_numba_cuda_origin", return_value=("RecursionError: x", ["line"]))]
        if fail:
            patches.append(mock.patch.dict("sys.modules", {"openfawe.aero.vortex_cuda": None}))   # import fails
        else:
            patches.append(mock.patch.dict("sys.modules", {"openfawe.aero.vortex_cuda": fake}))
        for p in patches:
            p.start()
        try:
            return selftest.part_gpu()
        finally:
            for p in reversed(patches):
                p.stop()

    def test_driver_and_no_gpu_fails(self):
        for fail in (True, False):
            row = self._part(driver=True, fail=fail)
            self.assertIs(row["passed"], False)
            self.assertEqual(row["numba_cuda_traceback"], ["line"])
            self.assertTrue(row["driver"])

    def test_no_driver_is_skipped(self):
        for fail in (True, False):
            row = self._part(driver=False, fail=fail)
            self.assertIsNone(row["passed"])

    def test_child_probe_keys(self):
        from openfawe_studio import selftest
        row = selftest.child_probe()
        self.assertTrue({"cuda_home", "available", "probe"} <= set(row))
        self.assertIsInstance(row["available"], bool)

    def test_short_traceback(self):
        from unittest import mock
        from openfawe_studio import selftest

        def deep(n):
            return deep(n + 1)
        try:
            deep(0)
        except RecursionError as error:
            caught = error
        lines = selftest.short_traceback(caught)            # Python folds the repeated frames itself
        self.assertTrue(lines[-1].startswith("RecursionError"))
        long = [f"line {i}\n" for i in range(500)]
        with mock.patch.object(selftest.traceback, "format_exception", return_value=long):
            lines = selftest.short_traceback(caught)
        self.assertEqual(len(lines), selftest.TRACEBACK_HEAD + selftest.TRACEBACK_TAIL + 1)
        self.assertEqual(lines[0], "line 0")
        self.assertEqual(lines[-1], "line 499")
        self.assertIn("380 lines left out", lines[selftest.TRACEBACK_HEAD])


class Addendum22(unittest.TestCase):
    """OF-0013 addendum 22: the collected NVIDIA files (REL-05) and the staged cases (REL-13)."""

    def test_runtime_only(self):
        entries = [("C:/sp/nvidia/cuda_nvcc/bin/ptxas.exe", "nvidia/cuda_nvcc/bin"),
                   ("C:/sp/nvidia/cuda_runtime/lib/x64/cudart_static.lib", "nvidia/cuda_runtime/lib/x64"),
                   ("C:/sp/nvidia/cuda_cccl/include/cub/cub.cuh", "nvidia/cuda_cccl/include/cub"),
                   ("C:/sp/nvidia/cuda_cccl/lib/cmake/cub/cub-config.cmake", "nvidia/cuda_cccl/lib/cmake/cub"),
                   ("C:/sp/nvidia/cuda_runtime/bin/cudart64_12.dll", "nvidia/cuda_runtime/bin"),
                   ("C:/sp/nvidia/cuda_nvcc/nvvm/bin/nvvm64_40_0.dll", "nvidia/cuda_nvcc/nvvm/bin"),
                   ("C:/sp/nvidia/cuda_nvcc/nvvm/libdevice/libdevice.10.bc", "nvidia/cuda_nvcc/nvvm/libdevice"),
                   ("C:/sp/nvidia/cuda_runtime/include/cuda_fp16.h", "nvidia/cuda_runtime/include")]
        kept = [d for _, d in cuda_bundle.runtime_only(entries)]
        self.assertEqual(kept, ["nvidia/cuda_runtime/bin", "nvidia/cuda_nvcc/nvvm/bin", "nvidia/cuda_nvcc/nvvm/libdevice",
                                "nvidia/cuda_runtime/include"])

    def test_cases_without_floating(self):
        spec = importlib.util.spec_from_file_location("stage_bundle", RELEASE / "stage_bundle.py")
        stage_bundle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stage_bundle)
        top = stage_bundle.REPO / "cases"
        self.assertIn("floating", stage_bundle._cases_ignore(top, ["floating", "ground", "__pycache__"]))
        self.assertNotIn("floating", stage_bundle._cases_ignore(top / "ground", ["floating", "x.json"]))
        self.assertIn("__pycache__", stage_bundle._cases_ignore(top / "ground", ["__pycache__"]))


class Supplement6(unittest.TestCase):
    """OF-0013 addendum 20 supplement 6: the CUDA distributions' METADATA in the bundle (G-0)."""

    def site(self, root):
        for folder, name, version in (("cuda_bindings-12.9.4.dist-info", "cuda-bindings", "12.9.4"),
                                      ("nvidia_cuda_runtime_cu12-12.9.79.dist-info", "nvidia-cuda-runtime-cu12", "12.9.79"),
                                      ("requests-2.32.0.dist-info", "requests", "2.32.0")):
            (root / folder).mkdir(parents=True)
            (root / folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8")
            (root / folder / "RECORD").write_text("C:/Users/someone/x.py,,\n", encoding="utf-8")
        return root

    def test_only_the_cuda_distributions_and_only_metadata(self):
        import importlib.metadata as md
        import shutil
        with tempfile.TemporaryDirectory() as d:
            site = self.site(Path(d) / "site")
            datas = cuda_bundle.metadata_datas(md.distributions(path=[str(site)]))
            self.assertEqual([(Path(s).name, dest) for s, dest in datas],
                             [("METADATA", "cuda_bindings-12.9.4.dist-info"), ("METADATA", "nvidia_cuda_runtime_cu12-12.9.79.dist-info")])
            bundle = Path(d) / "bundle"                                    # as PyInstaller places datas: file into folder
            for source, dest in datas:
                (bundle / dest).mkdir(parents=True)
                shutil.copy2(source, bundle / dest / Path(source).name)
            found = {x.metadata["Name"]: x.version for x in md.distributions(path=[str(bundle)])}
            self.assertEqual(found, {"cuda-bindings": "12.9.4", "nvidia-cuda-runtime-cu12": "12.9.79"})   # the lookup works
            self.assertFalse(list(bundle.rglob("RECORD")))                 # nothing that can hold the build PC's paths

    def test_spec_collects_the_metadata_in_the_gpu_branch(self):
        text = (RELEASE / "openfawe_studio.spec").read_text(encoding="utf-8")
        branch = text[text.index('if importable("numba_cuda"):'):text.index("a = Analysis(")]
        self.assertEqual(branch.count("datas += cuda_bundle.metadata_datas()"), 1)


class Supplement3(unittest.TestCase):
    """OF-0013 addendum 20 supplement 3: Qt for Python and the 'cuda' namespace collected as plain files."""

    def test_module_collection_mode(self):
        import ast
        tree = ast.parse((RELEASE / "openfawe_studio.spec").read_text(encoding="utf-8"))
        modes = [k for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Analysis"
                 for k in node.keywords if k.arg == "module_collection_mode"]
        self.assertEqual(len(modes), 1)
        self.assertEqual(ast.literal_eval(modes[0].value), {"PySide6": "py", "shiboken6": "py", "cuda": "py"})
        noarchive = [k for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Analysis"
                     for k in node.keywords if k.arg == "noarchive"]
        self.assertTrue(ast.literal_eval(noarchive[0].value))              # supplement 4

    def test_import_state_does_not_iterate_namespace_paths(self):
        from unittest import mock
        from openfawe_studio import selftest

        class Exploding(list):
            def __iter__(self):
                raise AssertionError("iterated")
        fake = types.ModuleType("cuda")
        fake.__path__ = type("_NamespacePath", (), {"_name": "cuda", "_path": ["a", "b"], "__iter__": Exploding.__iter__})()
        with mock.patch.dict("sys.modules", {"cuda": fake}):
            state = selftest.import_state()
        self.assertEqual(state["modules"]["cuda"]["path"], dict(type="_NamespacePath", name="cuda", path=["a", "b"]))
        self.assertTrue({"modules", "sys_path", "path_hooks"} <= set(state))


class Supplement5(unittest.TestCase):
    """OF-0013 addendum 20 supplement 5: the 'cuda' namespace copied as installed; cuda_core.libs registered."""

    def test_cuda_namespace_tree(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "cuda"
            for rel in ("core/__init__.py", "core/cu12/__init__.py", "core/cu12/_memory.cp312-win_amd64.pyd",
                        "core/cu12/_memory.pxd", "core/tests/test_a.py", "bindings/driver.cp312-win_amd64.pyd",
                        "pathfinder/__init__.py", "core/__pycache__/x.pyc"):
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            datas = cuda_bundle.tree_datas(root, dest=("cuda",), tests_anywhere=True)
            got = sorted((Path(src).relative_to(root).as_posix(), Path(dest).as_posix()) for src, dest in datas)
            self.assertEqual(got, [("bindings/driver.cp312-win_amd64.pyd", "cuda/bindings"), ("core/__init__.py", "cuda/core"),
                                   ("core/cu12/__init__.py", "cuda/core/cu12"),
                                   ("core/cu12/_memory.cp312-win_amd64.pyd", "cuda/core/cu12"),
                                   ("core/cu12/_memory.pxd", "cuda/core/cu12"), ("pathfinder/__init__.py", "cuda/pathfinder")])
        self.assertTrue(cuda_bundle.is_cuda_name("cuda") and cuda_bundle.is_cuda_name("cuda.core.cu12"))
        self.assertFalse(cuda_bundle.is_cuda_name("cudax") or cuda_bundle.is_cuda_name("numba.cuda"))

    def test_spec_excludes_cuda_and_libs(self):
        text = (RELEASE / "openfawe_studio.spec").read_text(encoding="utf-8")
        self.assertIn('excludes += ["numba.cuda", "cuda"]', text)
        self.assertNotIn('collect_submodules("cuda")', text)
        self.assertEqual(frozen_cuda.WHEEL_LIBS, ("numba_cuda.libs", "cuda_bindings.libs", "cuda_core.libs"))
