"""Explicit camber geometry, independent integrals and archival integrity."""
from pathlib import Path
import json
import shutil
import unittest
from uuid import uuid4
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from openfawe.aero.aircraft import WingAssembly, load_aircraft
from openfawe.aero.camber import (mean_line, node_camber_normals, normalize_contour,
                                 section_offsets, thin_airfoil_coefficients)
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig, build_lifting_surface_mesh,
                                           start_lifting_surface, _loads)
from openfawe.aero.methods import AeroBackend
from openfawe.aero.reference import refine_aircraft
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing, SectionPolar
from openfawe.aero_archive import AeroReader, AeroWriter
from openfawe.aero_vtk import export_paraview
from openfawe.provenance import sha256_file


def example():
    le=np.array([[0.,-2.,0.],[0.,0.,0.],[0.,2.,0.]])
    flat=Wing(le,le+[1.,0.,0.]);x=np.array([0.,.5,1.])
    offsets=section_offsets(flat,np.tile([0.,.02,0.],(3,1)))
    return flat,Wing(le,le+[1.,0.,0.],x,offsets)


class CamberTests(unittest.TestCase):
    def test_reconstructed_normal_has_analytic_camber_response(self):
        span=4000.;m=1e-4;x=np.linspace(0.,1.,4097)
        le=np.array([[0.,-span/2,0.],[0.,span/2,0.]])
        flat=Wing(le,le+[1.,0.,0.])
        wing=Wing(le,flat.trailing_edge,x,
                  section_offsets(flat,np.tile(4*m*x*(1-x),(2,1))))
        config=WakeConfig(core_model=0,core_radius_m=.01,density_kg_m3=1.)
        with self.assertRaisesRegex(ValueError,'camber normal'):
            LiftingSurfaceConfig(camber_normal='unknown')
        with self.assertRaisesRegex(ValueError,'at least two chordwise panels'):
            LiftingSurfaceConfig(chordwise_panels=1,camber_normal='reconstructed_slope')
        for nc,limit in ((8,.10),(32,.03)):
            settings=LiftingSurfaceConfig(chordwise_panels=nc,
                chordwise_spacing='cosine',wake_length_m=1e6,
                camber_normal='reconstructed_slope')
            mesh=build_lifting_surface_mesh(wing,settings)
            derivative=4*m*(1-2*mesh.collocation_points[:,0])
            expected=np.column_stack((-derivative,np.zeros_like(derivative),
                                       np.ones_like(derivative)))
            expected/=np.linalg.norm(expected,axis=1)[:,None]
            np.testing.assert_allclose(mesh.normals,expected,atol=1e-8,rtol=0)
            values=[]
            for alpha in (-1e-4,0.,1e-4):
                wind=np.array([np.cos(alpha),0.,np.sin(alpha)])
                state=start_lifting_surface(wing,config,wind,settings=settings)
                lift=state.solution.force@np.array([-np.sin(alpha),0.,np.cos(alpha)])
                cl=lift/(.5*span)
                cm=state.solution.moment(wing,[.25,0.,0.])[1]/(.5*span)
                values.append((cl,cm))
            slope=(values[2][0]-values[0][0])/2e-4
            alpha0=-values[1][0]/slope
            self.assertLess(abs(alpha0/(-2*m)-1),limit)
            self.assertLess(abs(values[1][0]/(4*np.pi*m)-1),limit)
            self.assertLess(abs(values[1][1]/(-np.pi*m)-1),limit)
        uniform_cm_errors=[]
        for nc in (8,16):
            settings=LiftingSurfaceConfig(chordwise_panels=nc,
                chordwise_spacing='uniform',wake_length_m=1e6,
                camber_normal='reconstructed_slope')
            solution=start_lifting_surface(wing,config,[1.,0.,0.],settings=settings).solution
            cm=solution.moment(wing,[.25,0.,0.])[1]/(.5*span)
            uniform_cm_errors.append(abs(cm/(-np.pi*m)-1))
        self.assertLess(uniform_cm_errors[1]/uniform_cm_errors[0],.26)
        default=build_lifting_surface_mesh(wing,LiftingSurfaceConfig(chordwise_panels=8))
        explicit=build_lifting_surface_mesh(wing,LiftingSurfaceConfig(
            chordwise_panels=8,camber_normal='cell_secant'))
        np.testing.assert_array_equal(default.normals,explicit.normals)
        for name in ('vertices','vortex_rows','collocation_points','normals','area'):
            old=build_lifting_surface_mesh(flat,LiftingSurfaceConfig(chordwise_panels=8))
            new=build_lifting_surface_mesh(flat,LiftingSurfaceConfig(
                chordwise_panels=8,camber_normal='reconstructed_slope'))
            np.testing.assert_array_equal(getattr(old,name),getattr(new,name))

    def test_reconstructed_normal_piecewise_and_reflexed_analytic_response(self):
        span=4000.;m=1e-4;x=np.linspace(0.,1.,4097)
        le=np.array([[0.,-span/2,0.],[0.,span/2,0.]])
        flat=Wing(le,le+[1.,0.,0.])
        nodes,weights=np.polynomial.legendre.leggauss(1024)
        theta=(nodes+1)*np.pi/2
        reference_x=(1-np.cos(theta))/2
        for kind in ('naca4','reflexed'):
            if kind=='naca4':
                p=.4
                line=lambda value: np.where(value<p,m/p**2*(2*p*value-value**2),
                    m/(1-p)**2*((1-2*p)+2*p*value-value**2))
                slope=lambda value: np.where(value<p,2*m/p**2*(p-value),
                    2*m/(1-p)**2*(p-value))
            else:
                line=lambda value: 4*m*value*(1-value)*(1-2*value)
                slope=lambda value: 4*m*(1-6*value+6*value**2)
            derivative=slope(reference_x)
            integral=lambda value: np.pi/2*np.dot(weights,value)
            exact_alpha=integral(derivative*(1-np.cos(theta)))/np.pi
            exact_cm=.5*integral(derivative*(np.cos(2*theta)-np.cos(theta)))
            wing=Wing(le,flat.trailing_edge,x,
                      section_offsets(flat,np.tile(line(x),(2,1))))
            errors=[]
            for nc in (8,16):
                settings=LiftingSurfaceConfig(chordwise_panels=nc,
                    chordwise_spacing='cosine',wake_length_m=1e6,
                    camber_normal='reconstructed_slope')
                config=WakeConfig(core_model=0,core_radius_m=.01,density_kg_m3=1.)
                values=[]
                for alpha in (-1e-4,0.,1e-4):
                    wind=np.array([np.cos(alpha),0.,np.sin(alpha)])
                    state=start_lifting_surface(wing,config,wind,settings=settings)
                    lift=state.solution.force@np.array([-np.sin(alpha),0.,np.cos(alpha)])
                    cm=state.solution.moment(wing,[.25,0.,0.])[1]/(.5*span)
                    values.append((lift/(.5*span),cm))
                lift_slope=(values[2][0]-values[0][0])/2e-4
                alpha0=-values[1][0]/lift_slope
                errors.append((abs(alpha0/exact_alpha-1),
                               abs(values[1][1]/exact_cm-1)))
            alpha_limit,cm_limit=((.007,.015) if kind=='naca4' else (.075,.071))
            self.assertLess(errors[0][0],alpha_limit)
            self.assertLess(errors[0][1],cm_limit)
            for metric in (0,1):
                self.assertLess(errors[1][metric]/errors[0][metric],2**-1.7)

    def test_reconstructed_normal_taper_twist_dihedral_and_rotation(self):
        le=np.array([[0.,-2.,0.],[.08,0.,.25],[.2,2.,.5]])
        te=le+np.array([[1.2,0.,.1],[.9,0.,-.05],[.7,0.,.08]])
        flat=Wing(le,te);x=np.linspace(0.,1.,4097)
        amplitudes=np.array([.015,.03,.005])
        lines=4*amplitudes[:,None]*x[None,:]*(1-x[None,:])
        wing=Wing(le,te,x,section_offsets(flat,lines))
        settings=LiftingSurfaceConfig(chordwise_panels=4,
            chordwise_spacing='cosine',camber_normal='reconstructed_slope')
        mesh=build_lifting_surface_mesh(wing,settings)
        fractions=.5*(1-np.cos(np.pi*np.arange(5)/4))
        node_normals=node_camber_normals(flat)
        chord=te-le;length=np.linalg.norm(chord,axis=1)
        expected=[]
        for i in range(4):
            xp=fractions[i]+.75*(fractions[i+1]-fractions[i])
            tangent=chord+(4*amplitudes*(1-2*xp)*length)[:,None]*node_normals
            surface_left=le+fractions[i]*chord+(
                4*amplitudes*fractions[i]*(1-fractions[i])*length)[:,None]*node_normals
            surface_right=le+fractions[i+1]*chord+(
                4*amplitudes*fractions[i+1]*(1-fractions[i+1])*length)[:,None]*node_normals
            for a,b in wing.panel_edges:
                span=(.25*(surface_left[b]-surface_left[a])+
                      .75*(surface_right[b]-surface_right[a]))
                normal=np.cross(.5*(tangent[a]+tangent[b]),span)
                expected.append(normal/np.linalg.norm(normal))
        np.testing.assert_allclose(mesh.normals,expected,atol=5e-8,rtol=0)
        az=.37;ay=.21
        rz=np.array([[np.cos(az),-np.sin(az),0.],
                     [np.sin(az),np.cos(az),0.],[0.,0.,1.]])
        ry=np.array([[np.cos(ay),0.,np.sin(ay)],
                     [0.,1.,0.],[-np.sin(ay),0.,np.cos(ay)]])
        q=ry@rz
        moved=build_lifting_surface_mesh(wing.transformed(q,[.5,-1.,2.]),settings)
        np.testing.assert_allclose(moved.normals,mesh.normals@q.T,atol=5e-11,rtol=0)

    def test_reconstructed_normal_archive_replays_exact_mesh(self):
        parent=Path(__file__).resolve().parents[1]/'local/camber-tests'
        parent.mkdir(parents=True,exist_ok=True)
        root=parent/uuid4().hex
        root.mkdir()
        try:
            _,wing=example()
            settings=LiftingSurfaceConfig(chordwise_panels=4,
                camber_normal='reconstructed_slope')
            backend=AeroBackend('vlm',wake=WakeConfig(core_radius_m=.03),
                                lifting_surface=settings)
            writer=AeroWriter(root/'run',dict(aero_method='vlm',
                aerodynamics=backend.metadata()))
            state=backend.start(wing,SectionPolar.linear(),[5.,0.,.1])
            writer.append(state)
            writer.finish()
            reader=AeroReader(root/'run')
            try:
                frame=reader.frame(0)
                self.assertEqual(json.loads(frame['lattice_settings_json'])['camber_normal'],
                                 'reconstructed_slope')
                np.testing.assert_array_equal(frame['lattice_normals'],state.mesh.normals)
            finally:
                reader.close()
        finally:
            if root.resolve().parent!=parent.resolve():
                raise ValueError('Unsafe test cleanup')
            shutil.rmtree(root)

    def test_thin_airfoil_integrals_and_profile_datum(self):
        x=.5*(1.-np.cos(np.linspace(0.,np.pi,4097)));m=.01
        alpha,cm=thin_airfoil_coefficients(x,4*m*x*(1-x))
        self.assertLess(abs(alpha/(-2*m)-1),1e-4)
        self.assertLess(abs(cm/(-np.pi*m)-1),1e-4)
        root=Path(__file__).resolve().parents[1]
        contour=np.loadtxt(root/'docs/tasks/OF-0006-models20260922/evidence/source/makani-f8.dat',skiprows=1)
        normalized,_=normalize_contour(contour,datum='normalized_source_chord')
        np.testing.assert_allclose(mean_line(normalized,np.linspace(.01,.99,99)),0.,atol=2e-6)
        opened=np.array([[1.,0.],[.5,.1],[0.,0.],[.5,-.05],[1.,-.001]])
        with self.assertRaisesRegex(ValueError,'nonzero mean trailing'):
            normalize_contour(opened)
        corrected,metadata=normalize_contour(opened,datum='source_chord_te_linear_closure')
        self.assertEqual(metadata['mean_line_linear_correction_slope'],.0005)
        self.assertEqual(mean_line(corrected,np.array([0.,1.])).tolist(),[0.,0.])

    def test_geometry_ownership_validation_and_assembly(self):
        flat,curved=example()
        for field in ('camber_x','camber_offset_m'):
            self.assertFalse(getattr(curved,field).flags.writeable)
        for x,z in [(None,curved.camber_offset_m),([0.,.5,.5],curved.camber_offset_m),
                    (curved.camber_x,np.ones_like(curved.camber_offset_m))]:
            with self.assertRaises(ValueError):Wing(flat.leading_edge,flat.trailing_edge,x,z)
        assembly=WingAssembly((curved,flat.transformed(np.eye(3),[2.,5.,0.])),('curved','flat'))
        self.assertTrue(np.all(assembly.camber_offset_m[3:]==0))
        self.assertFalse(any(a<3<=b for a,b in assembly.panel_edges))
        for name in ('cp','bound','area','normal','tangent','chord_cp'):
            np.testing.assert_array_equal(getattr(flat,name),getattr(curved,name))
        other=Wing(flat.leading_edge,flat.trailing_edge,[0.,1.],np.zeros((3,2,3)))
        with self.assertRaisesRegex(ValueError,'sampling grids'):
            WingAssembly((curved,other),('a','b'))

    def test_curved_mesh_covariance_and_physical_load_change(self):
        flat,curved=example();settings=LiftingSurfaceConfig(chordwise_panels=4)
        theta=.37;rotation=np.array([[np.cos(theta),-np.sin(theta),0.],
                                    [np.sin(theta),np.cos(theta),0.],[0.,0.,1.]])
        translation=np.array([1.5,-.3,2.])
        mesh=build_lifting_surface_mesh(curved,settings)
        moved=curved.transformed(rotation,translation)
        transformed=build_lifting_surface_mesh(moved,settings)
        np.testing.assert_allclose(transformed.vertices,mesh.vertices@rotation.T+translation,atol=1e-12,rtol=2e-11)
        np.testing.assert_allclose(transformed.normals,mesh.normals@rotation.T,atol=1e-12,rtol=2e-11)
        self.assertGreater(np.max(mesh.vertices[:,2]),0)
        backend=AeroBackend('vlm',wake=WakeConfig(core_radius_m=.03),lifting_surface=settings)
        wind=np.array([5.,0.,0.]);polar=SectionPolar.linear()
        a=backend.start(curved,polar,wind);b=backend.start(moved,polar,wind@rotation.T)
        np.testing.assert_allclose(b.solution.force,a.solution.force@rotation.T,rtol=2e-11,atol=1e-12)
        f=backend.start(flat,polar,wind)
        self.assertGreater(a.solution.force[2],f.solution.force[2]+.1)

    def test_manufactured_rate_uses_curved_cell_area_and_moment(self):
        _,wing=example();config=WakeConfig(core_radius_m=.03)
        backend=AeroBackend('vlm_free_wake',wake=config,
            lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
        state=backend.start(wing,SectionPolar.linear(),[3.,0.,.1]);gamma=state.solution.lattice_circulation
        args=(wing,state.mesh,gamma,state.free_rows,state.wake_circulation,state.cores,
              config,[3.,0.,.1],[0.,0.,0.],0.)
        f0,m0=_loads(*args,np.zeros_like(gamma));f1,m1=_loads(*args,np.full_like(gamma,2.))
        p=state.mesh.vertices[state.mesh.cells]
        # Independent planar quadrilateral integral; each cell is planar here.
        area_vector=.5*np.cross(p[:,3]-p[:,1],p[:,2]-p[:,0])
        expected=2*config.density_kg_m3*area_vector
        np.testing.assert_allclose((f1-f0).reshape(-1,3),expected,rtol=2e-11,atol=1e-12)
        moment=np.cross(p.mean(axis=1)-state.mesh.collocation_points,expected)
        np.testing.assert_allclose((m1-m0).reshape(-1,3),moment,rtol=2e-11,atol=1e-12)
        self.assertGreater(abs(expected[0,0]),.01)

    def test_curved_archive_corruption_and_vtk(self):
        parent=Path(__file__).resolve().parents[1]/'local/camber-tests';parent.mkdir(parents=True,exist_ok=True)
        root=parent/uuid4().hex;root.mkdir()
        try:
            _,wing=example();config=WakeConfig(dt_s=.05,core_radius_m=.02)
            for method in ('vlm','vlm_free_wake','ll_straight','free_wake','vsm','vsm_free_wake'):
                backend=AeroBackend(method,wake=config,lifting_surface=LiftingSurfaceConfig(chordwise_panels=2))
                writer=AeroWriter(root/method,dict(aero_method=method,name='Camber archive',aerodynamics=backend.metadata()))
                state=backend.start(wing,SectionPolar.linear(),[5.,0.,.1]);writer.append(state)
                state=backend.advance(state,SectionPolar.linear(),[5.,0.,.1],next_time_s=.05)
                writer.append(state);writer.finish()
                reader=AeroReader(root/method)
                try:
                    frame=reader.frame(1);self.assertEqual(frame['surface_geometry_model'],'mean_surface_offsets_v1')
                    self.assertEqual(frame['camber_used_by_solver'],method in ('vlm','vlm_free_wake'))
                    self.assertEqual(frame['camber_usage_status'],'recorded')
                    np.testing.assert_array_equal(frame['camber_offsets_m'],wing.camber_offset_m)
                finally:reader.close()
            export_paraview(root/'vlm_free_wake',root/'vtk')
            tree=ET.parse(root/'vtk/wing-000001.vtp')
            flag=tree.find('.//FieldData/DataArray[@Name="camber_geometry_0_ruled_1_explicit"]')
            self.assertEqual(float(flag.text),1.)
            for field in ('camber_offsets_m','surface_geometry_model','camber_used_by_solver'):
                bad=root/field;shutil.copytree(root/'vlm_free_wake',bad)
                with h5py.File(bad/'wake.h5','r+') as f:
                    g=f['frames/000001']
                    if field=='camber_offsets_m':g[field][0,1,2]+=.01
                    elif field=='camber_used_by_solver':g.attrs[field]=False
                    else:del g.attrs[field]
                m=json.loads((bad/'manifest.json').read_text(encoding='utf-8'))
                m['wake_sha256']=sha256_file(bad/'wake.h5');(bad/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'camber|lattice mesh'):AeroReader(bad)
            legacy=root/'legacy-camber';shutil.copytree(root/'vlm_free_wake',legacy)
            with h5py.File(legacy/'wake.h5','r+') as f:
                for g in f['frames'].values():del g.attrs['camber_used_by_solver']
            m=json.loads((legacy/'manifest.json').read_text(encoding='utf-8'))
            m['wake_sha256']=sha256_file(legacy/'wake.h5')
            (legacy/'manifest.json').write_text(json.dumps(m),encoding='utf-8')
            reader=AeroReader(legacy)
            try:
                self.assertTrue(reader.frame(1)['camber_used_by_solver'])
                self.assertEqual(reader.frame(1)['camber_usage_status'],'inferred_from_method_and_input')
            finally:reader.close()
        finally:
            if root.resolve().parent!=parent.resolve():raise ValueError('Unsafe test cleanup')
            shutil.rmtree(root)

    def test_derived_inputs_preserve_original_v3_polars_and_symmetry(self):
        root=Path(__file__).resolve().parents[1]
        old=load_aircraft(root/'cases/aircraft/v3_cad_attached.json')
        new=load_aircraft(root/'cases/aircraft/v3_cad_camber.json')
        self.assertIs(refine_aircraft(new,1),new)
        refined=refine_aircraft(new,2)
        np.testing.assert_array_equal(refined.wing.camber_x,new.wing.camber_x)
        np.testing.assert_array_equal(refined.wing.camber_offset_m[::2],new.wing.camber_offset_m)
        np.testing.assert_allclose(refined.wing.camber_offset_m[1::2],
            .5*(new.wing.camber_offset_m[:-1]+new.wing.camber_offset_m[1:]),rtol=0,atol=1e-15)
        np.testing.assert_array_equal(refined.wing.leading_edge[::2],new.wing.leading_edge)
        np.testing.assert_array_equal(refined.wing.trailing_edge[::2],new.wing.trailing_edge)
        np.testing.assert_array_equal(old.wing.cp,new.wing.cp)
        for a,b in zip(old.polar.polars,new.polar.polars):
            for key in ('alpha_deg','cl','cd','cm'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        offsets=new.wing.camber_offset_m
        np.testing.assert_allclose(offsets,offsets[::-1]*[1.,-1.,1.],rtol=0,atol=5e-13)
        integral=load_aircraft(root/'cases/aircraft/v3_cad_camber_integral.json')
        np.testing.assert_array_equal(integral.wing.leading_edge,new.wing.leading_edge)
        np.testing.assert_array_equal(integral.wing.trailing_edge,new.wing.trailing_edge)
        self.assertEqual(len(integral.wing.camber_x),4093)
        for a,b in zip(integral.polar.polars,new.polar.polars):
            for key in ('alpha_deg','cl','cd','cm'):
                np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        audit=json.loads((root/'docs/tasks/OF-0006-latticegap20260923/evidence/v3-integral-case-audit.json').read_text(encoding='utf-8'))
        self.assertEqual(len(audit['original_failures']),9)
        self.assertTrue(all(row['passed'] for row in audit['profiles']))
        self.assertLessEqual(max(abs(row['revised_delta_alpha0_rad']) for row in audit['profiles']),.001)
        self.assertLessEqual(max(abs(row['revised_delta_cm']) for row in audit['profiles']),.001)
        m=load_aircraft(root/'cases/qualification/m600_makani_reconstructed.json')
        self.assertEqual(m.wing.names,('Wing','Horizontal tail','Vertical tail'))
        self.assertEqual(m.wing.panels,49)
        m_refined=refine_aircraft(m,2)
        self.assertEqual(m_refined.wing.panels,98)
        for source,target in zip(m.wing.wings,m_refined.wing.wings):
            np.testing.assert_array_equal(target.camber_x,source.camber_x)
            np.testing.assert_array_equal(target.camber_offset_m[::2],source.camber_offset_m)
            np.testing.assert_allclose(target.camber_offset_m[1::2],
                .5*(source.camber_offset_m[:-1]+source.camber_offset_m[1:]),rtol=0,atol=1e-15)
        flat=load_aircraft(root/'cases/qualification/m600_makani_reconstructed_flat.json')
        self.assertIsNone(flat.wing.camber_x)
        for key in ('leading_edge','trailing_edge','panel_edges'):
            np.testing.assert_array_equal(getattr(flat.wing,key),getattr(m.wing,key))
        for a,b in zip(flat.polar.polars,m.polar.polars):
            for key in ('alpha_deg','cl','cd','cm'):np.testing.assert_array_equal(getattr(a,key),getattr(b,key))


if __name__=='__main__':unittest.main()
