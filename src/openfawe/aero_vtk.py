"""Small ParaView VTP/PVD exporter for verified complete aerodynamic archives."""

from pathlib import Path
import json
import xml.etree.ElementTree as ET
import numpy as np

from .aero.wake import pack_rings
from .aero_archive import AeroReader


def frame_network(frame):
    if 'network_nodes_m' in frame:
        # The archive owns the exact finite network used by the solver. Never
        # reconstruct chordwise rings from the strip-summary circulation.
        return (frame['network_nodes_m'], frame['network_edges'],
                frame['network_circulation_m2_s'])
    le,te=frame['leading_edge'],frame['trailing_edge']
    panel_edges=frame.get('panel_edges',np.column_stack([np.arange(len(le)-1),np.arange(1,len(le))]))
    if frame.get('aero_method')=='vsm':
        # Display only: solver uses analytic semi-infinite rays, never this length.
        width=len(le)
        points=np.vstack([le+.25*(te-le),te,
                          te+frame.get('vsm_display_wake_length_m',20.)*frame['prescribed_wake_direction']])
        edges=[];strengths=[]
        for (a,b),gamma in zip(panel_edges,frame['circulation']):
            edges.extend([[a,b],[b,width+b],[width+a,a],
                          [width+b,2*width+b],[2*width+a,width+a]])
            strengths.extend([gamma]*5)
        return points,np.asarray(edges,dtype=int),np.asarray(strengths)
    rows=np.concatenate(([le+.25*(te-le),te],frame['free_rows']),axis=0)
    gamma=frame['circulation']
    strengths=np.vstack([gamma,gamma,frame['older_circulation']]) if frame['step'] else gamma[None,:]
    return pack_rings(rows,strengths,panel_edges)


def _data(parent,name,values,components=1,integer=False):
    element=ET.SubElement(parent,'DataArray',type='Int64' if integer else 'Float64',
                          Name=name,NumberOfComponents=str(components),format='ascii')
    array=np.asarray(values).ravel()
    element.text=' '.join(str(int(v)) if integer else format(float(v),'.17g') for v in array)


def _vtp(path,points,cells,kind,cell_data,method='free_wake',field_data=None):
    root=ET.Element('VTKFile',type='PolyData',version='0.1',byte_order='LittleEndian')
    piece=ET.SubElement(ET.SubElement(root,'PolyData'),'Piece',NumberOfPoints=str(len(points)),
                        NumberOfVerts='0',NumberOfLines=str(len(cells) if kind=='Lines' else 0),
                        NumberOfStrips='0',NumberOfPolys=str(len(cells) if kind=='Polys' else 0))
    field=ET.SubElement(piece,'FieldData')
    method_ids={'free_wake':0,'vsm':1,'vsm_free_wake':2,'vlm':3,'vlm_free_wake':4,'ll_straight':5}
    if method not in method_ids:
        raise ValueError('unknown aerodynamic method for VTK export')
    _data(field,'aero_method_id_0_ll_free_wake_1_vsm_straight_2_vsm_free_wake_3_vlm_straight_4_uvlm_free_wake_5_ll_straight',
          [method_ids[method]],integer=True)
    if method_ids[method] < 5:
        _data(field,'aero_method_id_0_ll_free_wake_1_vsm_straight_2_vsm_free_wake_3_vlm_straight_4_uvlm_free_wake',
              [method_ids[method]],integer=True)
    if method_ids[method] < 3:
        _data(field,'aero_method_id_0_ll_free_wake_1_vsm_straight_2_vsm_free_wake',[method_ids[method]],integer=True)
    # Preserve the historical two-method field only where its name is truthful.
    if method_ids[method] < 2:
        _data(field,'aero_method_id_0_free_wake_1_vsm',[method_ids[method]],integer=True)
    for name, values in (field_data or {}).items():
        _data(field,name,values)
    _data(ET.SubElement(piece,'Points'),'Points',points,3)
    shape=ET.SubElement(piece,kind)
    _data(shape,'connectivity',cells,integer=True)
    _data(shape,'offsets',np.arange(1,len(cells)+1)*cells.shape[1],integer=True)
    data=ET.SubElement(piece,'CellData')
    for name,values in cell_data.items():
        array=np.asarray(values)
        if len(array)!=len(cells):
            raise ValueError(f'VTK cell data length differs from connectivity: {name}')
        _data(data,name,array,array.shape[1] if array.ndim==2 else 1,
              integer=np.issubdtype(array.dtype,np.integer))
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def _fuselage_vtp(path,mesh_path,method):
    """Static quad mesh of a fuselage (OF-0006-body20260930 .npz: nodes, quads)."""
    with np.load(mesh_path) as data:
        nodes,quads=np.asarray(data['nodes'],dtype=float),np.asarray(data['quads'],dtype=int)
    _vtp(path,nodes,quads,'Polys',{'fuselage_panel_index':np.arange(len(quads))},method,
         {'fuselage_0_display_only_1_in_computation':[0]})


def export_paraview(run_directory,output_directory,*,fuselage_mesh=None):
    """Write wing/wake VTP per frame and one wake.pvd; optional static fuselage part.

    The fuselage is display geometry unless the archive records a body
    coupling; its field data says so.
    """
    reader=AeroReader(run_directory)
    output=Path(output_directory)
    try:
        output.mkdir(parents=True,exist_ok=False)
        root=ET.Element('VTKFile',type='Collection',version='0.1',byte_order='LittleEndian')
        collection=ET.SubElement(root,'Collection')
        fuselage_name=None
        if fuselage_mesh is not None:
            fuselage_name='fuselage.vtp'
            _fuselage_vtp(output/fuselage_name,fuselage_mesh,reader.method)
        for i in range(reader.count):
            frame=reader.frame(i);le,te=frame['leading_edge'],frame['trailing_edge'];width=len(le)
            points=np.vstack([le,te])
            span_edges=frame.get('panel_edges',np.column_stack([np.arange(width-1),np.arange(1,width)]))
            panels=np.array([[a,b,b+width,a+width] for a,b in span_edges])
            wing_name=f'wing-{i:06d}.vtp';wake_name=f'wake-{i:06d}.vtp'
            fields={}
            if 'solver_trace_json' in frame:
                trace=json.loads(frame['solver_trace_json'])
                fields['solver_policy_1_newton_first_picard_v1']=[1]
                fields['solver_path_0_newton_1_picard']=[int(trace['fallback_used'])]
                fields['solver_picard_relaxation']=[trace['relaxation']]
                fields['solver_attempt_count']=[len(trace['attempts'])]
                fields['solver_delta_seed_linf_m2_s']=[trace['delta_seed_linf_m2_s']]
            if reader.method in ('vlm','vlm_free_wake'):
                points,panels=frame['lattice_nodes_m'],frame['lattice_cells']
                loads={'circulation_m2_s':frame['lattice_circulation_m2_s'],
                       'panel_force_N':frame['lattice_panel_force_N'],
                       'strip_index':frame['lattice_strip_index'],
                       'chord_index':frame['lattice_chord_index'],
                       'surface_normal':frame['lattice_normals'],
                       'integrated_area_normal':frame['lattice_area_normals'],
                       'area_m2':frame['lattice_area_m2'],
                       'circulation_rate_m2_s2':frame['lattice_gamma_rate_m2_s2'],
                       'panel_couple_N_m':frame['lattice_panel_couple_N_m']}
                if 'lattice_boundary_residual_m_s' in frame:
                    key='virtual_boundary_residual_m_s' if frame['section_polar_used'] else 'boundary_residual_m_s'
                    loads[key]=frame['lattice_boundary_residual_m_s']
                fields={'boundary_residual_max_m_s':[frame['boundary_residual_m_s']],
                        'boundary_residual_limit_m_s':[frame['boundary_residual_limit_m_s']],
                        'chordwise_spacing_0_uniform_1_cosine':[int(frame['chordwise_spacing'] == 'cosine')],
                        'matrix_condition_number':[frame['matrix_condition_number']],
                        'condition_roundoff_indicator':[frame['matrix_condition_number']*np.finfo(float).eps],
                        'section_polar_used_0_false_1_true':[int(frame['section_polar_used'])]}
                if frame['section_polar_used']:
                    fields['polar_correction_0_none_1_grid_section']=[1]
                    fields['boundary_0_geometric_1_virtual']=[1]
                    index=frame['lattice_strip_index']
                    loads.update(virtual_boundary_normal=frame['polar_virtual_normals'],
                        polar_cl_residual=frame['polar_cl_residual'][index],
                        virtual_incidence_rad=frame['polar_delta_rad'][index],
                        effective_polar_alpha_rad=frame['polar_effective_alpha_rad'][index],
                        geometric_boundary_residual_m_s=frame['polar_geometric_residual_m_s'],
                        polar_force_correction_N=frame['polar_correction_force_N'].reshape(-1,3),
                        polar_couple_correction_N_m=frame['polar_correction_couple_N_m'].reshape(-1,3))
                for name in ('minimum_chordwise_cell_length_m','prescribed_step_to_chord_ratio'):
                    if name in frame:
                        fields[name] = [frame[name]]
                fields['wake_release_0_previous_1_current_2_trapezoidal'] = [
                    {'previous':0, 'current':1, 'trapezoidal':2}[frame.get('wake_release', 'previous')]]
                fields['gamma_derivative_0_BE_1_BDF2'] = [int(frame.get('gamma_derivative') in ('bdf2', 'bdf2_consistent'))]
                fields['time_history_checked_0_legacy_1_explicit'] = [
                    int(frame['lattice_time_status'] in ('explicit_history_checked_v1', 'explicit_history_checked_v2', 'explicit_history_checked_v3'))]
                if frame.get('initial_gamma_rate_contract') == 'provided_d0_bdf2_start_v1':
                    fields['explicit_initial_gamma_rate'] = [1]
                    fields['initial_gamma_rate_provenance_0_declared_1_estimated'] = [
                        {'declared_kinematic':0, 'estimated_right_limit':1}[frame['initial_gamma_rate_provenance']]]
                    fields['initial_gamma_rate_reference_time_s'] = [frame['initial_gamma_rate_reference_time_s']]
                    loads['initial_gamma_rate_m2_s2'] = frame['initial_gamma_rate_m2_s2']
                    if frame['initial_gamma_rate_uncertainty_m2_s2'].size:
                        loads['initial_gamma_rate_uncertainty_m2_s2'] = frame['initial_gamma_rate_uncertainty_m2_s2']
                    if 'initial_rate_resolution_ratio' in frame:
                        loads['initial_rate_resolution_ratio'] = frame['initial_rate_resolution_ratio']
                        loads['initial_rate_resolution_valid'] = frame['initial_rate_resolution_valid'].astype(int)
                fields['wake_basis_0_constant_1_linear_potential'] = [int(frame.get('wake_basis') == 'linear_potential')]
                if frame.get('bound_convection_core') == 'te_incident_birth':
                    fields['bound_convection_core_0_legacy_1_te_incident_birth'] = [1]
                    fields['convection_bound_core_radius_m'] = [frame['convection_bound_core_radius_m']]
                    fields['saved_velocity_field_0_load_probe_1_convection'] = [0]
                if frame.get('initial_cap_contract') == 'closed_te_birth_core_v1':
                    fields['initial_cap_0_legacy_1_birth_core'] = [1]
                if frame.get('wake_basis') == 'linear_potential':
                    fields['wake_quadrature_order'] = [frame['wake_quadrature_order']]
                    fields['wake_quadrature_0_gauss_1_te_graded'] = [int(frame['wake_quadrature'] == 'te_graded')]
                fields['camber_geometry_0_ruled_1_explicit'] = [
                    int(frame['surface_geometry_model'] == 'mean_surface_offsets_v1')]
            if reader.method in ('vsm', 'vsm_free_wake'):
                sampling_id = {'legacy_closure_reuse':0,'separate_load_input_v1':1,'unrecorded':2}[frame['vsm_load_ambient']]
                fields['vsm_load_ambient_0_legacy_closure_reuse_1_separate_input_v1_2_unrecorded'] = [sampling_id]
                if sampling_id < 2:
                    fields['vsm_load_ambient_0_legacy_closure_reuse_1_separate_input_v1'] = [sampling_id]
            if reader.method not in ('vlm', 'vlm_free_wake'):
                loads={'Cl':frame['cl'],'circulation_m2_s':frame['circulation'],'panel_force_N':frame['panel_force']}
            fields.update({
                'force_model_0_quasi_steady_polar_1_bound_edge_joukowski':
                    [int(frame['force_model'].startswith('bound_edge_joukowski'))],
                'unsteady_term_0_none_1_rho_dgamma_dt_area_normal':
                    [int(frame['unsteady_term'] == 'rho_dgamma_dt_area_normal')]})
            _vtp(output/wing_name,points,panels,'Polys',loads,reader.method,fields)
            nodes,edges,gamma=frame_network(frame)
            cell_data={'circulation_m2_s':gamma}
            if 'network_core_radius_m' in frame:
                cell_data['core_radius_m']=frame['network_core_radius_m']
            _vtp(output/wake_name,nodes,edges,'Lines',cell_data,reader.method,fields)
            parts=[wing_name,wake_name]+([fuselage_name] if fuselage_name else [])
            for part,name in enumerate(parts):
                ET.SubElement(collection,'DataSet',timestep=format(float(frame['time_s']),'.17g'),group='',part=str(part),file=name)
        # Collection is written last; a partial export has no complete entry point.
        ET.ElementTree(root).write(output/'wake.pvd',encoding='utf-8',xml_declaration=True)
    finally:
        reader.close()
    return output/'wake.pvd'
