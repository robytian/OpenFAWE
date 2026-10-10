"""Independent whole-body traction and displaced initial stage limits."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from openfawe.aero.lifting_surface import (LiftingSurfaceConfig,start_lifting_surface,
    lifting_surface_network,lifting_surface_edge_radii,_linear_convection_network)
from openfawe.aero.vortex import CoreModel,segments_velocity
from openfawe.aero.wake import WakeConfig
from openfawe.aero.wing import Wing

class InitialCapLoadStageTests(unittest.TestCase):
    def state(self,basis):
        le=np.array([[0.,-2.,-.1],[0.,0.,0.],[0.,2.,.2]])
        w=Wing(le,le+[1.,0.,0.])
        setting=LiftingSurfaceConfig(5,chordwise_spacing='cosine',initial_cap='wake_core',
            wake_basis=basis,wake_release='current' if basis=='linear_potential' else 'previous',
            wake_quadrature='te_graded' if basis=='linear_potential' else 'gauss')
        return start_lifting_surface(w,WakeConfig(dt_s=.01,core_radius_m=.05,core_model=CoreModel.LAMB_OSEEN),
            [4.,.1,.2],settings=setting,free_wake=True)

    def test_whole_initial_force_and_moment_from_directed_cell_tractions(self):
        # Integrate each oriented cell separately. Shared tractions cancel in
        # the whole-body sum; no production load allocation/reduction is used.
        for basis in ('constant','linear_potential'):
            s=self.state(basis); nodes,edges,strength=lifting_surface_network(s)
            radii=lifting_surface_edge_radii(s); total=np.zeros(3); moment=np.zeros(3)
            rows=s.mesh.vortex_rows; nc=s.settings.chordwise_panels
            for i in range(nc):
                for j,(a,b) in enumerate(s.wing.panel_edges):
                    corners=[rows[i,a],rows[i,b],rows[i+1,b],rows[i+1,a]]
                    for edge_id in range(4):
                        if i==nc-1 and edge_id==2: continue  # Kutta traction cut.
                        first,last=corners[edge_id],corners[(edge_id+1)%4]
                        point=.5*(first+last)
                        exclude=[]
                        for k,(p,q) in enumerate(edges):
                            if ((np.array_equal(nodes[p],first) and np.array_equal(nodes[q],last)) or
                                (np.array_equal(nodes[q],first) and np.array_equal(nodes[p],last))):
                                exclude.append((0,k))
                        u=np.array([4.,.1,.2])+segments_velocity(point[None],nodes,edges,strength,
                            model=s.config.core_model,core_radius=radii,
                            exclusions=np.asarray(exclude,dtype=int).reshape(-1,2))[0]
                        force=s.config.density_kg_m3*s.solution.lattice_circulation[i,j]*np.cross(u,last-first)
                        total+=force; moment+=np.cross(point,force)
            assert_allclose(s.solution.force,total,atol=5e-12,rtol=5e-12)
            assert_allclose(s.solution.moment(s.wing),moment,atol=5e-12,rtol=5e-12)

    def test_displaced_initial_stage_field_tends_to_birth_cap_field(self):
        for basis in ('constant','linear_potential'):
            s=self.state(basis); network=lifting_surface_network(s); radii=lifting_surface_edge_radii(s)
            points=s.mesh.collocation_points
            initial=segments_velocity(points,*network,model=s.config.core_model,core_radius=radii)
            errors=[]
            for eps in (1e-5,1e-7,1e-9):
                if basis=='linear_potential':
                    net,radius,_=_linear_convection_network(s,s.config,s.wing.trailing_edge+eps*np.array([1.,0.,.1]),.5)
                else:
                    nodes,edges,gamma=network; nodes=nodes.copy()
                    nodes[-s.wing.node_count:]+=eps*np.array([1.,0.,.1]);net=(nodes,edges,gamma);radius=radii
                current=segments_velocity(points,*net,model=s.config.core_model,core_radius=radius)
                errors.append(float(np.abs(current-initial).max()))
            self.assertLess(errors[-1],.02*errors[-2])
            self.assertLess(errors[-1],1e-6)
