"""Reviewed configuration ownership and contiguous material-core archive contracts."""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4

import numpy as np

from openfawe.aero.lifting_surface import LiftingSurfaceConfig
from openfawe.aero.methods import AeroBackend
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero_archive import (AeroReader, AeroWriter, _validate_local_core_history,
                                  _validate_lattice_frame)


class WakeArchiveContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = Path(__file__).resolve().parents[1]/'local'/'archive-contract-tests'
        cls.root = cls.parent/uuid4().hex
        cls.root.mkdir(parents=True)
        y = np.linspace(-1., 1., 4)
        le = np.column_stack((np.zeros(4), y, np.zeros(4)))
        wing = Wing(le, le+[1., 0., 0.])
        polar, wind = SectionPolar.linear(), [3., 0., .1]
        cls.frames, cls.states = {}, {}
        for method in ('free_wake', 'vsm_free_wake', 'vlm_free_wake'):
            settings = LiftingSurfaceConfig(chordwise_panels=2, polar_coupling='grid_section')
            adapter = AeroBackend(method, wake=WakeConfig(dt_s=.05, max_rows=4,
                wake_core_chord_fraction=.1), lifting_surface=settings)
            writer = AeroWriter(cls.root/method, {'aero_method': method})
            states = [adapter.start(wing, polar, wind, time_s=10.)]
            try:
                writer.append(states[0])
                for _ in range(2):
                    states.append(adapter.advance(states[-1], polar, wind))
                    writer.append(states[-1])
                writer.finish()
            except BaseException as error:
                writer.fail(error)
                raise
            reader = AeroReader(cls.root/method)
            try:
                cls.frames[method] = [reader.frame(i) for i in range(3)]
            finally:
                reader.close()
            cls.states[method] = states
        # This supported mode deliberately has no local birth-core marker.
        adapter = AeroBackend('vlm_free_wake', wake=WakeConfig(dt_s=.05, max_rows=4),
            lifting_surface=LiftingSurfaceConfig(chordwise_panels=2,
                polar_coupling='grid_section', bound_convection_core='all_bound_chord'))
        writer = AeroWriter(cls.root/'bound-fixed', {'aero_method': 'vlm_free_wake'})
        try:
            state = adapter.start(wing, polar, wind)
            writer.append(state)
            writer.finish()
        except BaseException as error:
            writer.fail(error)
            raise
        reader = AeroReader(cls.root/'bound-fixed')
        try:
            cls.bound_frame = reader.frame(0)
        finally:
            reader.close()

    @classmethod
    def tearDownClass(cls):
        if cls.root.resolve().parent != cls.parent.resolve():
            raise ValueError('Refusing cleanup outside test workspace')
        shutil.rmtree(cls.root)

    def test_non_uvlm_methods_reject_attached_convection_core_settings(self):
        for method in ('ll_straight', 'vsm', 'free_wake', 'vsm_free_wake', 'vlm'):
            for options in ({'bound_convection_core': 'all_bound_chord'},
                            {'bound_convection_core': 'te_incident_birth',
                             'wake_basis': 'linear_potential', 'wake_release': 'current'}):
                with self.subTest(method=method, options=options):
                    settings = LiftingSurfaceConfig(**options)
                    with self.assertRaisesRegex(ValueError, '^attached convection core settings require vlm_free_wake$'):
                        AeroBackend(method, lifting_surface=settings)
            # Historical shared calls can still provide a plain chordwise mesh setting.
            self.assertEqual(AeroBackend(method, lifting_surface=LiftingSurfaceConfig(chordwise_panels=2)).method, method)

    def test_fixed_core_lattice_archive_rejects_invalid_convection_combinations(self):
        _validate_lattice_frame(self.bound_frame)
        for update in ({'free_wake': False}, {'core_model': 0}, {'core_model': 4}):
            with self.subTest(update=update):
                config = json.loads(self.bound_frame['wake_config_json'])
                config.update(update)
                changed = dict(self.bound_frame, wake_config_json=json.dumps(config, sort_keys=True))
                with self.assertRaisesRegex(ValueError, '^all_bound_chord requires free convection and a positive additive core$'):
                    _validate_lattice_frame(changed)

    def test_local_core_history_rejects_nonconsecutive_steps_explicitly(self):
        for method, frames in self.frames.items():
            with self.subTest(method=method):
                _validate_local_core_history(frames[2], frames[1])
                with self.assertRaisesRegex(ValueError, '^local core archive requires consecutive steps from step zero; sparse output and mid-run starts are unsupported$'):
                    _validate_local_core_history(frames[2], dict(frames[1], step=0))
                with self.assertRaisesRegex(ValueError, '^local core archive requires consecutive steps from step zero; sparse output and mid-run starts are unsupported$'):
                    _validate_local_core_history(frames[1], None)
                with self.assertRaisesRegex(ValueError, '^local core archive requires consecutive steps from step zero; sparse output and mid-run starts are unsupported$'):
                    _validate_local_core_history(frames[0], frames[0])
                with self.assertRaisesRegex(ValueError, '^local core archive requires consecutive steps from step zero; sparse output and mid-run starts are unsupported$'):
                    _validate_local_core_history(frames[1], frames[0], archive_index=2)

    def test_local_core_history_rejects_bad_dt_and_time_before_core_comparison(self):
        for method, frames in self.frames.items():
            for update in ({'time_s': frames[1]['time_s']+.025},
                           {'time_s': float('nan')},
                           {'time_s': frames[0]['time_s']}):
                with self.subTest(method=method, update=update):
                    with self.assertRaisesRegex(ValueError, '^local core archive time interval differs from configured dt$'):
                        _validate_local_core_history(dict(frames[1], **update), frames[0])
            for old, current in ((dict(frames[0], dt_s=.1), frames[1]),
                                 (frames[0], dict(frames[1], dt_s=.1))):
                with self.subTest(method=method), self.assertRaisesRegex(ValueError, '^local core archive dt metadata differs from configuration$'):
                    _validate_local_core_history(current, old)

    def test_local_core_writer_rejects_midrun_start_and_sparse_append_without_commit(self):
        message = '^local core archive requires consecutive steps from step zero; sparse output and mid-run starts are unsupported$'
        for method, states in self.states.items():
            with self.subTest(method=method):
                writer = AeroWriter(self.root/('rejected-'+method), {'aero_method': method})
                try:
                    with self.assertRaisesRegex(ValueError, message):
                        writer.append(states[1])
                    self.assertEqual(writer.count, 0)
                    self.assertEqual(len(writer.file['frames']), 0)
                    writer.append(states[0])
                    with self.assertRaisesRegex(ValueError, message):
                        writer.append(states[2])
                    self.assertEqual(writer.count, 1)
                    self.assertEqual(len(writer.file['frames']), 1)
                    writer.append(states[1])
                    # A state with unchanged material cores but altered time is invalid.
                    with self.assertRaisesRegex(ValueError, '^local core archive time interval differs from configured dt$'):
                        writer.append(replace(states[2], time_s=states[2].time_s+.025))
                    self.assertEqual(writer.count, 2)
                    self.assertEqual(len(writer.file['frames']), 2)
                    writer.append(states[2])
                    writer.finish()
                except BaseException as error:
                    writer.fail(error)
                    raise


if __name__ == '__main__':
    unittest.main()
