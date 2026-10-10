"""Explicit initial derivative: analytic oracle, load locality and strict replay."""
from contextlib import closing
from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from uuid import uuid4
import h5py
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from openfawe.aero.initial_rate import InitialGammaRate
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, start_lifting_surface,
    advance_lifting_surface, build_lifting_surface_mesh, _solve)
from openfawe.aero.wake import WakeConfig
from openfawe.aero.vortex import CoreModel
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero.methods import AeroBackend
from openfawe.aero_archive import AeroWriter, AeroReader, INITIAL_RATE_LATTICE_SCHEMA
from openfawe.aero_vtk import export_paraview
from openfawe.run_catalog import inspect_run
from openfawe.provenance import sha256_file

ROOT=Path(__file__).resolve().parents[1]

def wing():
    le=np.array([[0.,-2.,0.],[0.,0.,0.],[0.,2.,0.]])
    return Wing(le,le+[1.,0.,0.])

def settings():
    return LiftingSurfaceConfig(3,wake_basis='linear_potential',wake_release='current',
        wake_quadrature='te_graded',initial_cap='wake_core',gamma_derivative='bdf2_consistent')

def initial(values,**kw):
    return InitialGammaRate(values,0.,'declared_kinematic',json.dumps({'source':'manufactured test'}),**kw)

class ConsistentStartTests(unittest.TestCase):
    def test_missing_input_domain_and_invalid_provenance(self):
        w=wing(); cfg=WakeConfig(); chosen=settings()
        with self.assertRaises(ValueError):
            start_lifting_surface(w,cfg,[4.,0.,.1],settings=chosen,free_wake=True)
        with self.assertRaises(ValueError):
            replace(chosen,wake_basis='constant',wake_quadrature='gauss')
        with self.assertRaises(ValueError):
            replace(chosen,initial_cap='legacy_zero')
        for value in (initial(np.zeros((2,2))), replace(initial(np.zeros((3,2))),reference_time_s=1.)):
            with self.assertRaises(ValueError):
                start_lifting_surface(w,cfg,[4.,0.,.1],settings=chosen,free_wake=True,initial_gamma_rate=value)
        with self.assertRaises(ValueError):
            start_lifting_surface(w,cfg,[4.,0.,.1],settings=replace(chosen,gamma_derivative='bdf2'),
                free_wake=True,initial_gamma_rate=initial(np.zeros((3,2))))
        for source,evidence,unc in [('unknown',{'source':'test'},None),
                ('declared_kinematic',{},None),('estimated_right_limit',{'source':'test','epsilon_s':[.1,.01]},None),
                ('estimated_right_limit',{'source':'test','epsilon_s':[.01,.1]},np.zeros((3,2)))]:
            with self.assertRaises(ValueError):
                InitialGammaRate(np.zeros((3,2)),0.,source,json.dumps(evidence),unc)
        value=initial(np.zeros((3,2)))
        with self.assertRaises(ValueError): value.rate_m2_s2.setflags(write=True)

    def test_initial_rate_reference_binds_to_oldest_release(self):
        cfg=WakeConfig(dt_s=.02,free_wake=False)
        state=start_lifting_surface(wing(),cfg,[4.,0.,0.],settings=settings(),
            free_wake=True,time_s=2.,initial_gamma_rate=replace(initial(np.zeros((3,2))),reference_time_s=2.))
        for _ in range(2):
            state=advance_lifting_surface(state,cfg,[4.,0.,0.])
            self.assertEqual(state.release_times_s[-1],2.)
            with self.assertRaisesRegex(ValueError,'reference time'):
                replace(state,initial_gamma_rate=replace(state.initial_gamma_rate,reference_time_s=1.))

    def test_nonfinite_initial_rate_and_uncertainty_are_rejected(self):
        for invalid in (np.nan,np.inf,-np.inf):
            for field in ('rate_m2_s2','uncertainty_m2_s2'):
                values=np.zeros((3,2));values[0,0]=invalid
                with self.subTest(field=field,value=invalid),self.assertRaises(ValueError):
                    replace(initial(np.zeros((3,2))),**{field:values})

    def test_quadratic_analytic_first_step_rate(self):
        w=wing(); cfg=WakeConfig(dt_s=.03); chosen=settings(); mesh=build_lifting_surface_mesh(w,chosen)
        # Exact Gamma(t)=a*(t-h)^2+b*(t-h); Gamma(h)=0 from zero normal flow.
        # The independently differentiated value at t=h is b for every cell.
        h=cfg.dt_s; a=np.arange(1.,7.).reshape(3,2); b=.3+a*.1
        solution,_=_solve(w,mesh,chosen,cfg,[4.,0.,0.],[0.,0.,0.],h,
            np.empty((0,3,3)),np.empty((0,2)),None,previous=a*h*h-b*h,
            initial_gamma_rate=initial(b-2*a*h))
        assert_array_equal(solution.lattice_circulation,np.zeros((3,2)))
        assert_allclose(solution.lattice_gamma_rate,b,rtol=3e-15,atol=3e-15)
        pressure=cfg.density_kg_m3*b.ravel()[:,None]*mesh.area[:,None]*mesh.area_normals
        assert_allclose(solution.force,pressure.sum(axis=0),atol=1e-14)
        assert_allclose(solution.moment(w),np.cross(mesh.centroids,pressure).sum(axis=0),atol=1e-14)

    def test_d0_perturbation_changes_only_first_two_pressure_loads(self):
        w=wing(); chosen=settings(); delta=np.arange(1.,7.).reshape(3,2)*.02
        for free in (False,True):
            cfg=WakeConfig(dt_s=.02,free_wake=free,core_radius_m=.05)
            backend=AeroBackend('vlm_free_wake',wake=cfg,lifting_surface=chosen)
            a=backend.start(w,None,[4.,0.,.1],initial_gamma_rate=initial(np.zeros((3,2))))
            b=backend.start(w,None,[4.,0.,.1],initial_gamma_rate=initial(delta))
            legacy=start_lifting_surface(w,cfg,[4.,0.,.1],settings=replace(chosen,gamma_derivative='bdf2'),free_wake=True)
            for k in range(4):
                for name in ('lattice_circulation','lattice_induced_velocity'):
                    assert_array_equal(getattr(a.solution,name),getattr(b.solution,name))
                    assert_array_equal(getattr(a.solution,name),getattr(legacy.solution,name))
                assert_array_equal(a.free_rows,b.free_rows)
                assert_array_equal(a.wake_node_potential_m2_s,b.wake_node_potential_m2_s)
                sign=1 if k==0 else -1 if k==1 else 0
                expected=sign*cfg.density_kg_m3*delta.ravel()[:,None]*a.mesh.area[:,None]*a.mesh.area_normals
                assert_allclose(b.solution.force-a.solution.force,expected.sum(axis=0),atol=2e-13,rtol=2e-13)
                assert_allclose(b.solution.moment(w)-a.solution.moment(w),
                    np.cross(a.mesh.centroids,expected).sum(axis=0),atol=2e-13,rtol=2e-13)
                if k>=2:
                    assert_array_equal(a.solution.lattice_gamma_rate,legacy.solution.lattice_gamma_rate)
                    assert_array_equal(a.solution.force,legacy.solution.force)
                if k<3:
                    a,b,legacy=[advance_lifting_surface(s,cfg,[4.,0.,.1]) for s in (a,b,legacy)]

    def test_archive_consumers_and_refreshed_hash_corruption(self):
        parent=ROOT/'local/consistent-start-tests'; parent.mkdir(parents=True,exist_ok=True)
        root=parent/uuid4().hex; root.mkdir(); success=False
        try:
            for core in (CoreModel.NONE,CoreModel.LAMB_OSEEN):
                path=root/core.name; cfg=WakeConfig(dt_s=.02,core_model=core,free_wake=False)
                supplied=InitialGammaRate(np.zeros((3,2)),0.,'estimated_right_limit',
                    json.dumps({'source':'manufactured zero-normal-flow limit','epsilon_s':[.001,.0005]}),np.zeros((3,2)))
                state=start_lifting_surface(wing(),cfg,[4.,0.,0.],settings=settings(),free_wake=True,initial_gamma_rate=supplied)
                writer=AeroWriter(path,{'aero_method':'vlm_free_wake'})
                try:
                    for k in range(3):
                        if k: state=advance_lifting_surface(state,cfg,[4.,0.,0.])
                        writer.append(state)
                    writer.finish()
                except BaseException as e: writer.fail(e); raise
                with closing(AeroReader(path)) as reader:
                    self.assertEqual(reader.manifest['schema'],INITIAL_RATE_LATTICE_SCHEMA)
                    self.assertEqual(reader.frame(0)['gamma_rate_stencil'],'provided_initial_rate')
                    self.assertEqual(reader.frame(1)['gamma_rate_stencil'],'consistent_bdf2_start')
                    self.assertFalse(reader.frame(1)['initial_rate_resolution_valid'].any())
                    self.assertEqual(reader.frame(2)['gamma_rate_stencil'],'bdf2')
                self.assertIn('explicit initial Gamma rate: estimated_right_limit',inspect_run(path).core)
                export_paraview(path,root/('vtk-'+core.name))
                self.assertIn('explicit_initial_gamma_rate', (root/('vtk-'+core.name)/'wing-000000.vtp').read_text())
                import xml.etree.ElementTree as ET
                tree=ET.parse(root/('vtk-'+core.name)/'wing-000000.vtp')
                fields={a.attrib['Name']:a.text.strip() for a in tree.findall('.//FieldData/DataArray')}
                self.assertEqual(fields['initial_gamma_rate_provenance_0_declared_1_estimated'],'1')
                self.assertEqual(float(fields['initial_gamma_rate_reference_time_s']),0.)
                for mode in ('downgrade','missing-contract','changed-d0','changed-evidence','missing-uncertainty','bad-first-rate','bad-ratio'):
                    bad=root/(core.name+'-'+mode);shutil.copytree(path,bad)
                    manifest=json.loads((bad/'manifest.json').read_text())
                    with h5py.File(bad/'wake.h5','r+') as file:
                        g=file['frames/000001']
                        if mode=='downgrade':
                            manifest['schema']='openfawe.aero.lattice.initialcap.v7';file.attrs['schema']=manifest['schema']
                        elif mode=='missing-contract': manifest.pop('initial_gamma_rate_contract')
                        elif mode=='changed-d0': g['initial_gamma_rate_m2_s2'][0]=.1
                        elif mode=='changed-evidence': g.attrs['initial_gamma_rate_evidence_json']='{}'
                        elif mode=='missing-uncertainty': del g['initial_gamma_rate_uncertainty_m2_s2']
                        elif mode=='bad-first-rate': g['lattice_gamma_rate_m2_s2'][0]=.1
                        else: g['initial_rate_resolution_ratio'][0]=1.
                    manifest['wake_sha256']=sha256_file(bad/'wake.h5')
                    (bad/'manifest.json').write_text(json.dumps(manifest))
                    with self.assertRaises(ValueError): AeroReader(bad)
            success=True
        finally:
            if success:
                if root.resolve().parent!=parent.resolve() or not parent.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Refusing cleanup outside test workspace')
                shutil.rmtree(root)

    def test_nonzero_initial_rate_combined_polar_core_and_free_contracts(self):
        parent=ROOT/'local/consistent-start-tests'; parent.mkdir(parents=True,exist_ok=True)
        root=parent/uuid4().hex; root.mkdir(); success=False
        try:
            angles=np.linspace(-25.,25.,101)
            polar=SectionPolar(angles,5*np.sin(np.deg2rad(angles)),angles*0+.02,angles*0-.03)
            for coupled in (False,True):
                for free in (False,True):
                    cfg=WakeConfig(dt_s=.01,free_wake=free,core_radius_m=.05,core_evolution='age_stretch')
                    chosen=replace(settings(),polar_coupling='grid_section' if coupled else 'none')
                    supplied=initial(np.arange(1.,7.).reshape(3,2)*.01)
                    state=start_lifting_surface(wing(),cfg,[4.,0.,.1],settings=chosen,free_wake=True,
                        polar=polar,initial_gamma_rate=supplied)
                    path=root/f'{coupled}-{free}'; writer=AeroWriter(path,{'aero_method':'vlm_free_wake'})
                    try:
                        for k in range(3):
                            if k: state=advance_lifting_surface(state,cfg,[4.,0.,.1],polar=polar)
                            writer.append(state)
                        writer.finish()
                    except BaseException as error: writer.fail(error); raise
                    with closing(AeroReader(path)) as reader:
                        assert_array_equal(reader.frame(0)['lattice_gamma_rate_m2_s2'],supplied.rate_m2_s2.ravel())
                        self.assertEqual(reader.frame(2)['lattice_time_status'],'explicit_history_checked_v3')
                        assert_array_equal(reader.frame(2)['panel_force'].sum(axis=0),state.solution.force)
                    for key in (['wake_core_mapping']+(['linear_wake_convection_contract'] if free else [])+
                                (['polar_coupling_model'] if coupled else [])):
                        bad=root/f'bad-{coupled}-{free}-{key}';shutil.copytree(path,bad)
                        manifest=json.loads((bad/'manifest.json').read_text());manifest.pop(key)
                        (bad/'manifest.json').write_text(json.dumps(manifest))
                        with self.assertRaises(ValueError): AeroReader(bad)
            success=True
        finally:
            if success:
                if root.resolve().parent!=parent.resolve() or not parent.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError('Refusing cleanup outside test workspace')
                shutil.rmtree(root)
