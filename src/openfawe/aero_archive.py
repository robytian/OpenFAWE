"""Separate aerodynamic HDF5 archive; never uses synthetic power/line channels."""

from pathlib import Path
from dataclasses import asdict
import json
from uuid import uuid4
import numpy as np
import h5py

from .provenance import source_identity,environment,sha256_file,write_json,utc_now
from .aero.methods import AeroMethod,VsmState
from .aero.vsm import core_kernel_identity
from .aero.wing import wake_config_metadata
from .aero.wake import WakeState, wake_network
from .aero.cores import edge_radii
from .aero.ll_straight import LLStraightState, straight_network, network_radii

SCHEMA="openfawe.aero.v2"
LEGACY_SCHEMA="openfawe.aero.wake.v1"
LATTICE_SCHEMA="openfawe.aero.lattice.v3"
POLAR_LATTICE_SCHEMA="openfawe.aero.lattice.polar.v4"
POTENTIAL_LATTICE_SCHEMA="openfawe.aero.lattice.potential.v5"
POTENTIAL_POLAR_LATTICE_SCHEMA="openfawe.aero.lattice.potential.polar.v6"
INITIAL_CAP_LATTICE_SCHEMA="openfawe.aero.lattice.initialcap.v7"
INITIAL_RATE_LATTICE_SCHEMA="openfawe.aero.lattice.initialrate.v8"
POTENTIAL_SCHEMAS=(POTENTIAL_LATTICE_SCHEMA,POTENTIAL_POLAR_LATTICE_SCHEMA)
POLAR_SCHEMAS=(POLAR_LATTICE_SCHEMA,POTENTIAL_POLAR_LATTICE_SCHEMA)
LATTICE_SCHEMAS=(LATTICE_SCHEMA,POLAR_LATTICE_SCHEMA,*POTENTIAL_SCHEMAS,INITIAL_CAP_LATTICE_SCHEMA,
                 INITIAL_RATE_LATTICE_SCHEMA)
SUPPORTED_SCHEMAS=(SCHEMA,LEGACY_SCHEMA,*LATTICE_SCHEMAS)
LATTICE_METHODS=('vlm','vlm_free_wake')
METHOD_LABELS = {
    'vsm': 'OpenFAWE VSM\u2013Straight Wake',
    'free_wake': 'OpenFAWE LL\u2013Free Vortex Wake',
    'vsm_free_wake': 'OpenFAWE VSM\u2013Free Vortex Wake',
    'vlm': 'OpenFAWE VLM\u2013Straight Wake',
    'vlm_free_wake': 'OpenFAWE UVLM\u2013Free Vortex Wake',
    'll_straight': 'OpenFAWE LL\u2013Straight Wake',
}


def _declared_vsm_jacobian(case):
    """A recorded solve dictionary with no derivative field declares legacy FD."""
    solve = (case.get('aerodynamics') or {}).get('solve')
    return solve.get('jacobian', 'finite_difference') if isinstance(solve, dict) else None


def _reject_non_vsm_jacobian(method, case, frame, root, manifest):
    if method == 'vsm':
        return
    trace = json.loads(frame['solver_trace_json']) if 'solver_trace_json' in frame else {}
    if (any('vsm_jacobian_policy' in data for data in (frame, root, manifest))
            or trace.get('jacobian', 'finite_difference') != 'finite_difference'
            or _declared_vsm_jacobian(case) not in (None, 'finite_difference')):
        raise ValueError('non-VSM archive carries a VSM Jacobian policy')


def method_label(method):
    """Stable public names; method IDs remain compatible with old archives."""
    return METHOD_LABELS.get(method, f'OpenFAWE unknown method ({method})')


def field_semantics(method, *, separate_vsm_load=False, polar_coupled=False):
    """Describe existing fields without interpreting table extent as validity."""
    lattice = method in LATTICE_METHODS
    vsm = method in ('vsm', 'vsm_free_wake')
    semantics = {
        'alpha_rad_definition': ('Prescribed relative-flow angle at strip quarter chord' if lattice else
            'Load-direction angle at quarter chord; polar closure uses polar_alpha_rad' if vsm else
            'Effective section angle at quarter chord, used for polar closure and loads'),
        'polar_alpha_rad_definition': ('Closure angle at three-quarter chord' if vsm else
            'Not stored; section polars unused' if lattice else 'Same closure angle as alpha_rad'),
        'reynolds_definition': ('Strip chord times prescribed relative speed / kinematic viscosity' if lattice else
            'Section chord times closure velocity normal-to-span magnitude / kinematic viscosity' if vsm else
            'Section chord times full relative velocity magnitude / kinematic viscosity'),
        'polar_reynolds_policy': ('Not applicable; section polars unused' if lattice else
            'Fixed table per section; no Reynolds interpolation; table provenance stays in case input'),
        'polar_validity_status': ('not_applicable' if lattice else 'unknown'),
        'polar_clipped_definition': ('Not applicable' if lattice else
            'Outside tabulated angle endpoints only; does not establish physical validity'),
        'section_polar_used': not lattice,
        'force_model': 'bound_edge_joukowski' if lattice else 'quasi_steady_sectional_polar',
        'unsteady_term': 'rho_dgamma_dt_area_normal' if method == 'vlm_free_wake' else 'none',
    }
    if lattice and polar_coupled:
        semantics.update(section_polar_used=True,
            polar_alpha_rad_definition='polar_effective_alpha_rad: inverse grid-consistent section reference',
            polar_reynolds_policy='Fixed source table per strip; no Reynolds interpolation',
            polar_validity_status='unknown', polar_clipped_definition='Clipping forbidden by coupled closure',
            force_model='bound_edge_joukowski_plus_source_polar_corrections',
            boundary_condition='Virtual normals: geometric residual retained separately')
    if vsm:
        if separate_vsm_load is None:
            semantics.update(vsm_load_ambient='unrecorded',
                load_point_ambient_sampling='Unrecorded: ambient-array presence does not establish the sampling convention',
                alpha_rad_definition='Load-direction angle; ambient sampling convention unrecorded; polar closure uses polar_alpha_rad')
            return semantics
        semantics['vsm_load_ambient'] = ('separate_load_input_v1' if separate_vsm_load else 'legacy_closure_reuse')
        semantics['load_point_ambient_sampling'] = (
            'Spatial callables sampled separately at quarter chord; legacy panel arrays reused as supplied'
            if separate_vsm_load else 'Legacy closure-point ambient reused with quarter-chord induction')
        if not separate_vsm_load:
            semantics['alpha_rad_definition'] = 'Load-direction angle from closure-point ambient plus quarter-chord induction (legacy reuse)'
    return semantics


def replay_config(case, frame):
    """Actual state metadata takes precedence; missing legacy values stay unknown."""
    saved = frame.get('wake_config_json')
    if saved is not None:
        return json.loads(saved)
    return dict(case.get('wake') or (case.get('aerodynamics') or {}).get('wake') or {})


def _camber_fields(wing, method):
    """Separate supplied geometry provenance from geometry used by the solver."""
    x = getattr(wing, 'camber_x', None)
    if x is None:
        return {'surface_geometry_model':'ruled_chord_v1', 'camber_used_by_solver':False}, {}
    return ({'surface_geometry_model':'mean_surface_offsets_v1',
             'camber_used_by_solver':method in LATTICE_METHODS},
            {'camber_x_over_c':x, 'camber_offsets_m':wing.camber_offset_m})


def _validate_camber_frame(frame):
    from .aero.wing import camber_data
    present = ('camber_x_over_c' in frame, 'camber_offsets_m' in frame)
    marker = frame.get('surface_geometry_model', 'ruled_chord_v1')
    expected = 'mean_surface_offsets_v1' if all(present) else 'ruled_chord_v1'
    if present[0] != present[1] or marker != expected:
        raise ValueError('inconsistent archived camber geometry marker or arrays')
    x, offsets = camber_data(frame.get('camber_x_over_c'), frame.get('camber_offsets_m'),
                             len(frame['leading_edge']))
    frame['surface_geometry_model'] = expected
    # The original marker describes the supplied geometry. Historical camber
    # frames did not say whether their backend actually used those offsets.
    used = x is not None and frame.get('aero_method') in LATTICE_METHODS
    recorded = 'camber_used_by_solver' in frame
    if recorded and (not isinstance(frame['camber_used_by_solver'], (bool,np.bool_))
                     or bool(frame['camber_used_by_solver']) != used):
        raise ValueError('camber usage marker contradicts the aerodynamic method')
    frame['camber_used_by_solver'] = used
    frame['camber_usage_status'] = 'recorded' if recorded else 'inferred_from_method_and_input'
    return x, offsets


def _validate_common_frame(frame):
    _validate_camber_frame(frame)
    for name, value in frame.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise ValueError(f'nonfinite archive field: {name}')
    if ('closure_ambient_velocity' in frame and
            frame['closure_ambient_velocity'].shape != frame['relative_velocity'].shape):
        raise ValueError('closure ambient velocity shape differs from panel velocities')
    for name in ('time_s', 'iterations'):
        if not np.isfinite(frame[name]):
            raise ValueError(f'nonfinite archive scalar: {name}')
    if frame['iterations'] < 1 or int(frame['iterations']) != frame['iterations']:
        raise ValueError('positive integer solver iterations required')
    if frame['aero_method'] not in LATTICE_METHODS:
        residual, limit = frame['residual_m2_s'], frame['residual_limit_m2_s']
        if not np.isfinite([residual, limit]).all() or not 0 <= residual <= limit or limit <= 0:
            raise ValueError('circulation residual did not pass original tolerance')
        if 'solver_trace_json' in frame:
            from .aero.nonlinear import validate_solver_trace
            validate_solver_trace(frame['solver_trace_json'], residual, limit, len(frame['circulation']))
    for name in ('kinematic_mismatch_m_s', 'condition_roundoff_indicator',
                 'prescribed_step_to_chord_ratio', 'minimum_chordwise_cell_length_m'):
        if name in frame and (not np.isfinite(frame[name]) or frame[name] < 0):
            raise ValueError(f'invalid archive diagnostic: {name}')


def wake_cores_from_frame(frame):
    """Restore material core state; absent legacy cumulative strain is unknown."""
    from types import SimpleNamespace
    from .aero.cores import WakeCores, check_cores
    config = SimpleNamespace(**json.loads(frame.get('wake_config_json', '{}')))
    fraction = getattr(config, 'wake_core_chord_fraction', None)
    marker = frame.get('wake_core_birth_policy')
    if fraction is not None:
        if (marker != 'local_chord_v1'
                or frame.get('wake_core_birth_chord_fraction') != fraction
                or frame.get('core_cumulative_log_strain_status') != 'unrecorded'):
            raise ValueError('local wake core archive lacks its consistent birth contract')
    elif marker is not None or 'wake_core_birth_chord_fraction' in frame:
        raise ValueError('local birth core metadata contradicts archived configuration')
    policy = frame.get('core_evolution', 'constant')
    if policy == 'constant' and fraction is None:
        return None
    cores = WakeCores(frame['span_core_radius_m'], frame['stream_core_radius_m'],
        policy, frame['initial_core_radius_m'], frame['core_diffusion_rate_m2_s'],
        chord_fraction=fraction)
    if getattr(config, 'core_evolution', None) is not None:
        check_cores(cores, config)
    step = int(frame['step'])
    if (cores.span.shape != (step, len(frame['circulation']))
            or cores.stream.shape != (step, len(frame['leading_edge']))):
        raise ValueError('archived physical core history shape mismatch')
    return cores


def _local_core_attributes(config):
    fraction = getattr(config, 'wake_core_chord_fraction', None)
    return ({} if fraction is None else dict(wake_core_birth_policy='local_chord_v1',
        wake_core_birth_chord_fraction=fraction, core_cumulative_log_strain_status='unrecorded'))


def _validate_wake_frame(frame):
    """Validate old line-wake archives as well as explicit-network extensions."""
    from .aero.wake import pack_rings
    le, te = frame['leading_edge'], frame['trailing_edge']
    n, step = len(frame['circulation']), int(frame['step'])
    if (frame['free_rows'].shape != (step, len(le), 3)
            or frame['release_times_s'].shape != (step,)
            or frame['older_circulation'].shape != (max(step-1, 0), n)):
        raise ValueError('archived wake history shape mismatch')
    if (np.any(frame['release_times_s'] >= frame['time_s'])
            or np.any(np.diff(frame['release_times_s']) >= 0)):
        raise ValueError('invalid archived wake release times')
    rows = np.concatenate(([le+.25*(te-le), te], frame['free_rows']))
    gamma = frame['circulation']
    strengths = np.vstack([gamma, gamma, frame['older_circulation']]) if step else gamma[None]
    panel_edges = frame.get('panel_edges', np.column_stack((np.arange(len(le)-1), np.arange(1, len(le)))))
    network = pack_rings(rows, strengths, panel_edges)
    names = ('network_nodes_m', 'network_edges', 'network_circulation_m2_s')
    if any(name in frame for name in names):
        if not all(name in frame for name in names):
            raise ValueError('incomplete archived vortex network')
        for name, expected in zip(names, network):
            if not np.array_equal(frame[name], expected):
                raise ValueError('archived vortex network differs from saved wake')
    if 'network_core_radius_m' in frame:
        radii = frame['network_core_radius_m']
        if radii.shape != (len(network[1]),) or np.any(radii < 0):
            raise ValueError('invalid wake network core radii')
        # Reconstruct material-radius routing, not merely its length.
        from types import SimpleNamespace
        cores = wake_cores_from_frame(frame)
        initial = frame.get('initial_core_radius_m')
        if initial is not None:
            wing = SimpleNamespace(node_count=len(le), panel_edges=panel_edges)
            expected = edge_radii(wing, network[1], cores, initial)
            if frame.get('core_model') == 0:
                expected = np.zeros_like(expected)
            if not np.array_equal(radii, expected):
                raise ValueError('archived core radii differ from material edge routing')


def _validate_straight_ll_frame(frame):
    from .aero.wake import pack_rings
    if any(not np.isfinite(a).all() for a in frame.values() if isinstance(a, np.ndarray)):
        raise ValueError('nonfinite straight LL archive')
    le, te = frame['leading_edge'], frame['trailing_edge']
    far = frame['prescribed_wake_row_m']
    if far.shape != te.shape or len(frame['free_rows']) or len(frame['release_times_s']) or len(frame['older_circulation']):
        raise ValueError('straight LL archive cannot contain released history')
    if not np.allclose(np.linalg.norm(far-te, axis=1), frame['prescribed_wake_length_m'], rtol=1e-12, atol=1e-12):
        raise ValueError('prescribed LL wake length differs from saved geometry')
    rows = np.array([le+.25*(te-le), te, far])
    network = pack_rings(rows, np.repeat(frame['circulation'][None, :], 2, axis=0), frame['panel_edges'])
    for name, expected in zip(('network_nodes_m', 'network_edges', 'network_circulation_m2_s'), network):
        if not np.array_equal(frame[name], expected):
            raise ValueError('straight LL archive network differs from saved state')
    radii = frame['network_core_radius_m']
    if radii.shape != (len(network[1]),) or np.any(radii < 0):
        raise ValueError('invalid straight LL network core radii')
    residual, limit = frame['residual_m2_s'], frame['residual_limit_m2_s']
    if not np.isfinite([residual, limit]).all() or not 0 <= residual <= limit or limit <= 0:
        raise ValueError('straight LL circulation residual did not pass')


def aero_source_identity():
    identity=source_identity()
    package=Path(__file__).resolve().parent
    # The original scaffold fingerprint covers top-level modules. Explicitly
    # include the numerical subpackage for aerodynamic scientific provenance.
    identity['files_sha256'].update({"openfawe/"+p.relative_to(package).as_posix():sha256_file(p)
                                    for p in sorted((package/'aero').glob('*.py'))})
    return identity


def _bound_convection_metadata(settings, config):
    """Explicit model binding; archived filament radii remain load/probe radii."""
    if settings.bound_convection_core == 'all_bound_chord':
        return dict(bound_convection_core='all_bound_chord',
            convection_bound_core_chord_fraction=settings.bound_convection_chord_fraction,
            bound_core_model='all_bound_chord_convection_v1; mean endpoint chord; load and probe radii unchanged')
    if settings.bound_convection_core == 'legacy_zero':
        return {}
    model = ('ideal bound edges; zero-history closing TE uses recorded birth core'
             if settings.initial_cap == 'wake_core'
             else 'ideal finite segments; wake cores use recorded WakeConfig')
    return dict(bound_convection_core='te_incident_birth',
                convection_bound_core_radius_m=config.core_radius_m,
                bound_core_model=model+'; TE-incident bound edges use birth core for convection only')


def _validate_lattice_frame(frame):
    """Reject inconsistent new-schema topology, load routing and SI residuals."""
    for name,array in frame.items():
        if isinstance(array,np.ndarray) and not np.isfinite(array).all():
            raise ValueError(f'nonfinite archive field: {name}')
    strips=len(frame['circulation'])
    nodes=np.asarray(frame['lattice_nodes_m'])
    cells=np.asarray(frame['lattice_cells'])
    count=len(cells)
    if nodes.ndim!=2 or nodes.shape[1]!=3 or not count or cells.shape!=(count,4):
        raise ValueError('invalid lifting-surface lattice shape')
    from types import SimpleNamespace
    from .aero.lifting_surface import (LiftingSurfaceConfig, build_lifting_surface_mesh,
                                       validate_lifting_surface_wake)
    wing = SimpleNamespace(leading_edge=frame['leading_edge'], trailing_edge=frame['trailing_edge'],
        panel_edges=frame['panel_edges'], node_count=len(frame['leading_edge']), panels=strips,
        camber_x=frame.get('camber_x_over_c'), camber_offset_m=frame.get('camber_offsets_m'))
    settings = LiftingSurfaceConfig(**json.loads(frame['lattice_settings_json']))
    linear = settings.wake_basis == 'linear_potential'
    wake_config = SimpleNamespace(**json.loads(frame['wake_config_json']))
    validate_lifting_surface_wake(settings, wake_config, frame['aero_method'] == 'vlm_free_wake')
    convection_metadata = _bound_convection_metadata(settings, wake_config)
    for name, value in convection_metadata.items():
        if frame.get(name) != value:
            raise ValueError('bound convection core metadata differs from settings/configuration')
    if not convection_metadata and any(name in frame for name in (
            'bound_convection_core', 'convection_bound_core_radius_m')):
        raise ValueError('bound convection core evidence stored for a legacy field')
    capped = settings.initial_cap == 'wake_core'
    if capped:
        if frame.get('initial_cap_contract') != 'closed_te_birth_core_v1':
            raise ValueError('explicit initial cap lacks its mandatory core contract')
    elif frame.get('initial_cap_contract') is not None:
        raise ValueError('initial cap evidence stored for a legacy initial state')
    if linear:
        from .aero.lifting_surface import linear_wake_convection_contract
        if (wake_config.free_wake and
                frame.get('linear_wake_convection_contract') != linear_wake_convection_contract(settings)):
            raise ValueError('linear free convection requires its attached frozen-node contract')
        if not wake_config.free_wake and frame.get('linear_wake_convection_contract') is not None:
            raise ValueError('convection contract stored for prescribed material geometry')
    mesh = build_lifting_surface_mesh(wing, settings)
    if settings.steady_mirror_projection:
        from .aero.lifting_surface import _steady_mirror_permutation
        permutation = _steady_mirror_permutation(
            wing, mesh, frame['prescribed_wake_row_m'], (0., 0., 0.), (0., 0., 0.))
        gamma = frame['lattice_circulation_m2_s']
        if not np.array_equal(gamma, gamma[permutation]):
            raise ValueError('archived steady mirror circulation is not exactly paired')
    for name, expected in {
        'lattice_nodes_m':mesh.vertices, 'lattice_cells':mesh.cells,
        'lattice_vortex_rows_m':mesh.vortex_rows, 'lattice_control_points_m':mesh.collocation_points,
        'lattice_force_reference_points_m':mesh.collocation_points,
        'lattice_normals':mesh.normals, 'lattice_area_normals':mesh.area_normals,
        'lattice_area_m2':mesh.area, 'lattice_centroids_m':mesh.centroids,
        'lattice_strip_index':mesh.strip_index, 'lattice_chord_index':mesh.chord_index}.items():
        if name not in frame or not np.array_equal(frame[name], expected):
            raise ValueError(f'archived lattice mesh differs from geometry/settings: {name}')
    for name,connectivity,size in [('lattice_cells',cells,len(nodes)),
                                  ('network_edges',np.asarray(frame['network_edges']),len(frame['network_nodes_m']))]:
        width=4 if name=='lattice_cells' else 2
        if connectivity.ndim!=2 or connectivity.shape[1]!=width or not np.issubdtype(connectivity.dtype,np.integer):
            raise ValueError(f'integer connectivity required: {name}')
        if np.any(connectivity<0) or np.any(connectivity>=size):
            raise ValueError(f'connectivity outside saved nodes: {name}')
    for name in ('lattice_control_points_m','lattice_force_reference_points_m','lattice_centroids_m',
                 'lattice_normals','lattice_area_normals','lattice_panel_force_N',
                 'lattice_panel_couple_N_m','lattice_relative_velocity_m_s',
                 'lattice_induced_velocity_m_s','lattice_structural_velocity_m_s'):
        if np.shape(frame[name])!=(count,3):
            raise ValueError(f'invalid lifting-surface field shape: {name}')
    for name in ('lattice_circulation_m2_s','lattice_gamma_rate_m2_s2',
                 'lattice_strip_index','lattice_chord_index','lattice_area_m2',
                 'lattice_boundary_residual_m_s'):
        if np.shape(frame[name])!=(count,):
            raise ValueError(f'invalid lifting-surface field shape: {name}')
    strip=np.asarray(frame['lattice_strip_index'])
    chord=np.asarray(frame['lattice_chord_index'])
    if (not np.issubdtype(strip.dtype,np.integer) or not np.issubdtype(chord.dtype,np.integer)
            or np.any(strip<0) or np.any(strip>=strips) or np.any(chord<0)):
        raise ValueError('invalid lattice strip/chord mapping')
    summed=np.zeros((strips,3))
    np.add.at(summed,strip,frame['lattice_panel_force_N'])
    if not np.allclose(summed,frame['panel_force'],rtol=2e-13,atol=2e-13):
        raise ValueError('lattice forces do not sum to saved strip forces')
    moment=np.zeros((strips,3))
    arm=frame['lattice_force_reference_points_m']-frame['control_points'][strip]
    np.add.at(moment,strip,frame['lattice_panel_couple_N_m']+
              np.cross(arm,frame['lattice_panel_force_N']))
    if not np.allclose(moment,frame['panel_couple'],rtol=2e-13,atol=2e-13):
        raise ValueError('lattice moments do not reduce to saved strip couples')
    if np.shape(frame['network_nodes_m'])[1:]!=(3,):
        raise ValueError('invalid saved vortex node shape')
    for name in ('network_circulation_m2_s','network_core_radius_m'):
        if np.shape(frame[name])!=(len(frame['network_edges']),):
            raise ValueError(f'invalid saved network field shape: {name}')
    # Rebuild the closed ring graph from the saved lattice and material history.
    # A valid shape or refreshed file hash cannot establish state consistency.
    from .aero.wake import pack_rings
    lattice_gamma = frame['lattice_circulation_m2_s'].reshape(settings.chordwise_panels, strips)
    if not np.array_equal(lattice_gamma[-1], frame['circulation']):
        raise ValueError('archived lattice network strip circulation is inconsistent')
    if frame['aero_method'] == 'vlm':
        wake_rows = frame['prescribed_wake_row_m'][None]
        wake_gamma = frame['circulation'][None]
    else:
        wake_rows = frame['free_rows']
        wake_gamma = frame['wake_circulation_m2_s']
    if wake_rows.shape != (len(wake_gamma), wing.node_count, 3) or wake_gamma.shape[1:] != (strips,):
        raise ValueError('invalid archived lattice network history shape')
    if linear:
        phi = frame.get('wake_node_potential_m2_s')
        if (phi is None or phi.shape != (len(wake_rows)+1, strips) or
                not np.array_equal(phi[0], frame['circulation']) or
                not np.array_equal(wake_gamma, .5*(phi[:-1]+phi[1:]))):
            raise ValueError('archived nodal potential TE trace or cell means are inconsistent')
        from .aero.wake_potential import potential_network, potential_edge_radii
        network = potential_network(wing, mesh, lattice_gamma, wake_rows, phi,
                                    settings.wake_quadrature_order, settings.wake_quadrature)
        if int(wake_config.core_model) == 0:
            expected_radius = np.zeros(len(network[1]))
        else:
            if frame.get('wake_core_mapping') != 'material_edge_q_v1':
                raise ValueError('regularized linear potential requires material core mapping contract')
            from .aero.cores import WakeCores, core_identity
            cores = None
            if wake_config.core_evolution != 'constant':
                identity = core_identity(wake_config)
                if (frame.get('core_evolution'), frame.get('initial_core_radius_m'),
                        frame.get('core_diffusion_rate_m2_s')) != identity:
                    raise ValueError('linear potential core parameters differ from configuration')
                cores = WakeCores(frame['span_core_radius_m'], frame['stream_core_radius_m'], *identity)
                if cores.span.shape != (len(wake_rows), strips) or cores.stream.shape != (len(wake_rows), wing.node_count):
                    raise ValueError('linear potential physical core history shape mismatch')
            expected_radius = potential_edge_radii(wing, mesh, wake_rows, network[1], cores,
                wake_config.core_radius_m, settings.wake_quadrature_order, settings.wake_quadrature,
                initial_cap=capped and not len(wake_rows))
        if not np.array_equal(frame['network_core_radius_m'], expected_radius):
            raise ValueError('linear potential network cores differ from physical material mapping')
    else:
        if 'wake_node_potential_m2_s' in frame:
            raise ValueError('nodal potential stored without its declared wake basis')
        network = pack_rings(np.concatenate((mesh.vortex_rows, wake_rows)),
                             np.vstack((lattice_gamma, wake_gamma)), wing.panel_edges)
        if capped or getattr(wake_config, 'wake_core_chord_fraction', None) is not None:
            from .aero.lifting_surface import _radii
            cores = wake_cores_from_frame(frame)
            expected_radius = _radii(wing, network[1], settings.chordwise_panels, cores,
                wake_config.core_radius_m, initial_cap=capped and not len(wake_rows) and int(wake_config.core_model) != 0)
            if not np.array_equal(frame['network_core_radius_m'], expected_radius):
                raise ValueError('initial cap network cores differ from declared history')
    for name, expected in zip(('network_nodes_m', 'network_edges', 'network_circulation_m2_s'), network):
        if not np.array_equal(frame[name], expected):
            raise ValueError(f'archived lattice network differs from saved state: {name}')
    if np.any(frame['network_core_radius_m']<0):
        raise ValueError('negative saved vortex core radius')
    if (frame.get('section_polar_used')!=(settings.polar_coupling != 'none') or 'polar_clipped' in frame
            or 'residual_m2_s' in frame or 'residual_limit_m2_s' in frame):
        raise ValueError('lifting-surface polar use or residual units are mislabelled')
    if settings.polar_coupling != 'none':
        if frame.get('polar_coupling_contract') not in (None, 'grid_section_v2'):
            raise ValueError('unknown archived polar coupling contract')
        from .aero.polar_coupling import validate_archive
        validate_archive(frame, wing, mesh, settings,
                         json.loads(frame['wake_config_json'])['density_kg_m3'])
    elif any(name.startswith('polar_') and name not in (
            'polar_alpha_rad_definition','polar_reynolds_policy','polar_validity_status',
            'polar_clipped_definition') for name in frame):
        raise ValueError('coupling evidence stored for an inviscid lattice')
    residual=frame['boundary_residual_m_s'];limit=frame['boundary_residual_limit_m_s']
    if not np.isfinite([residual,limit]).all() or not 0<=residual<=limit or limit<=0:
        raise ValueError('lifting-surface no-penetration residual did not pass')
    if not np.isclose(np.max(np.abs(frame['lattice_boundary_residual_m_s'])),residual,rtol=1e-12,atol=1e-15):
        raise ValueError('boundary residual maximum differs from saved field')
    if settings.steady_mirror_projection:
        if frame['aero_method'] != 'vlm':
            raise ValueError('steady mirror projection is only valid in straight VLM archives')
        before = frame.get('mirror_preprojection_max_abs_m2_s')
        raw_residual = frame.get('unprojected_boundary_residual_m_s')
        limit_gamma = settings.mirror_projection_atol_m2_s + settings.mirror_projection_rtol*np.max(
            np.abs(frame['lattice_circulation_m2_s']))
        if (before is None or raw_residual is None or
                not np.isfinite([before, raw_residual]).all() or
                not 0 <= before <= limit_gamma or raw_residual < 0):
            raise ValueError('invalid steady mirror projection diagnostics')
    elif ('mirror_preprojection_max_abs_m2_s' in frame or
          'unprojected_boundary_residual_m_s' in frame):
        raise ValueError('mirror projection diagnostics without an enabled option')
    for name, expected in _lattice_resolution_fields(frame).items():
        if name in frame and isinstance(expected, float) and not np.isclose(frame[name],expected,rtol=2e-13,atol=1e-14):
            raise ValueError(f'archived lattice resolution diagnostic is inconsistent: {name}')


def _lattice_time_metadata(settings, step, method):
    transient = method == 'vlm_free_wake'
    stencil = ('none' if not transient else 'zero_initial' if step == 0 else
               'backward_euler_startup' if step == 1 and settings.gamma_derivative == 'bdf2' else
               settings.gamma_derivative)
    operator = ('bound_plus_prescribed_wake' if not transient else
                'bound_plus_release' if step and settings.wake_release != 'previous' else 'bound_only')
    result = dict(lattice_time_contract='uvlm_time_v1', wake_release=settings.wake_release,
        gamma_derivative=settings.gamma_derivative, gamma_rate_stencil=stencil,
        condition_operator=operator,
        trailing_edge_jump_definition='current TE Gamma minus newest material row Gamma; scheme identity, not an accuracy metric')
    if settings.wake_basis == 'linear_potential':
        result.update(lattice_time_contract='uvlm_time_v2',
            wake_basis='linear_potential', wake_quadrature=settings.wake_quadrature,
            wake_quadrature_order=settings.wake_quadrature_order,
            wake_potential_contract='material_nodes_v1',
            wake_cell_circulation_definition='Derived adjacent nodal arithmetic means for display; not the induction field',
            trailing_edge_jump_definition='Current TE Gamma minus current material-node potential trace; zero after a material interval exists')
    if settings.gamma_derivative == 'bdf2_consistent':
        result.update(lattice_time_contract='uvlm_time_v3', gamma_rate_stencil=(
            'provided_initial_rate' if step == 0 else 'consistent_bdf2_start' if step == 1 else 'bdf2'))
    return result


def _lattice_previous_frame(groups, index):
    if index == 0:
        return None
    group = groups[f'{index-1:06d}']
    result = dict(group.attrs)
    for name in ('lattice_circulation_m2_s', 'previous_lattice_circulation_m2_s',
                 'wake_circulation_m2_s', 'wake_node_potential_m2_s',
                 'release_times_s', 'polar_delta_rad', 'leading_edge', 'trailing_edge', 'panel_edges', 'circulation',
                 'free_rows', 'span_core_radius_m', 'stream_core_radius_m'):
        if name in group:
            result[name] = group[name][:]
    for name in ('initial_gamma_rate_m2_s2', 'initial_gamma_rate_uncertainty_m2_s2'):
        if name in group:
            result[name] = group[name][:]
    return result


_LOCAL_CORE_SEQUENCE_ERROR = ('local core archive requires consecutive steps from step zero; '
                              'sparse output and mid-run starts are unsupported')


def _validate_local_core_history(frame, previous, *, archive_index=None):
    """Validate every accepted step from zero, then recompute material cores.

    A new local-core archive cannot begin from a resumed state or omit physical
    steps. Its complete adjacent geometry and configured dt define core birth,
    age and stretch; this is not an arbitrary sampled-state restart format.
    """
    from types import SimpleNamespace
    from .aero.cores import advance_cores
    from .aero.wake import count
    config = SimpleNamespace(**json.loads(frame.get('wake_config_json', '{}')))
    if getattr(config, 'wake_core_chord_fraction', None) is None:
        return
    step = count(frame['step'], 'archived local core step')
    if ((archive_index is not None and step != archive_index)
            or (previous is None and step != 0)
            or (previous is not None and (step == 0
                or count(previous['step'], 'previous local core step') != step-1))):
        raise ValueError(_LOCAL_CORE_SEQUENCE_ERROR)
    if previous is not None and previous.get('wake_config_json') != frame.get('wake_config_json'):
        raise ValueError('missing or inconsistent adjacent local core configuration')
    dt = float(config.dt_s)
    if (not np.isfinite(dt) or dt <= 0 or frame.get('dt_s') != dt
            or (previous is not None and previous.get('dt_s') != dt)):
        raise ValueError('local core archive dt metadata differs from configuration')
    time = float(frame['time_s'])
    old_time = time if previous is None else float(previous['time_s'])
    if (not np.isfinite([time, old_time]).all()
            or (previous is not None and (time <= old_time or not np.isclose(time-old_time, dt,
                rtol=0, atol=8*np.finfo(float).eps*max(1., abs(time), abs(old_time)))))):
        raise ValueError('local core archive time interval differs from configured dt')
    cores = wake_cores_from_frame(frame)
    if step == 0:
        return
    wing = SimpleNamespace(leading_edge=previous['leading_edge'], trailing_edge=previous['trailing_edge'],
        panel_edges=previous['panel_edges'], panels=len(previous['panel_edges']),
        node_count=len(previous['leading_edge']))
    old = SimpleNamespace(wing=wing, free_rows=previous['free_rows'], cores=wake_cores_from_frame(previous))
    expected = advance_cores(old, frame['free_rows'], config)
    if not (np.array_equal(cores.span, expected.span) and np.array_equal(cores.stream, expected.stream)):
        raise ValueError('local material core history differs from accepted adjacent evolution')


def _validate_linear_core_history(frame, previous):
    """Check accepted physical core evolution independently of auxiliary routing."""
    from types import SimpleNamespace
    from .aero.cores import WakeCores, core_identity, advance_cores
    config = SimpleNamespace(**json.loads(frame['wake_config_json']))
    if config.core_evolution == 'constant' or int(frame['step']) == 0:
        return
    if previous is None:
        raise ValueError('missing adjacent physical core history')
    wing = SimpleNamespace(trailing_edge=previous['trailing_edge'], panel_edges=previous['panel_edges'],
                           panels=len(previous['panel_edges']), node_count=len(previous['trailing_edge']))
    old = SimpleNamespace(wing=wing, free_rows=previous['free_rows'],
        cores=WakeCores(previous['span_core_radius_m'], previous['stream_core_radius_m'], *core_identity(config)))
    expected = advance_cores(old, frame['free_rows'], config)
    for name, value in (('span_core_radius_m', expected.span), ('stream_core_radius_m', expected.stream)):
        if not np.allclose(frame[name], value, rtol=1e-12, atol=1e-14):
            raise ValueError('linear potential physical core history differs from adjacent evolution')


def _validate_lattice_time(frame, previous):
    """Validate explicit time histories; old archives retain their old bytes."""
    from .aero.lifting_surface import LiftingSurfaceConfig
    raw = json.loads(frame['lattice_settings_json'])
    if 'lattice_time_contract' not in frame:
        if 'wake_release' in raw or 'gamma_derivative' in raw:
            raise ValueError('new lattice time settings lack the time contract')
        frame['lattice_time_status'] = 'legacy_previous_BE; history not independently verified'
        return
    settings = LiftingSurfaceConfig(**raw)
    step, method = int(frame['step']), frame['aero_method']
    if settings.wake_basis == 'linear_potential':
        _validate_linear_core_history(frame, previous)
    for name, expected in _lattice_time_metadata(settings, step, method).items():
        if frame.get(name) != expected:
            raise ValueError(f'inconsistent lattice time metadata: {name}')
    gamma = frame['lattice_circulation_m2_s']
    initial_rate = None
    if settings.gamma_derivative == 'bdf2_consistent':
        from .aero.initial_rate import from_archive, startup_ratio
        initial_rate = from_archive(frame, (settings.chordwise_panels, len(frame['circulation'])))
        if step == 0 and initial_rate.reference_time_s != frame['time_s']:
            raise ValueError('initial Gamma rate reference time differs from first frame')
        if previous is not None:
            old_rate = from_archive(previous, initial_rate.rate_m2_s2.shape)
            if initial_rate.to_dict() != old_rate.to_dict():
                raise ValueError('initial Gamma rate input changed inside archive')
        if step == 1:
            if previous is None:
                raise ValueError('missing consistent-start previous frame')
            ratio, valid = startup_ratio(initial_rate, gamma, previous['lattice_circulation_m2_s'],
                                         json.loads(frame['wake_config_json'])['dt_s'])
            if (not np.array_equal(frame.get('initial_rate_resolution_ratio'), ratio) or
                    not np.array_equal(frame.get('initial_rate_resolution_valid'), valid)):
                raise ValueError('initial Gamma rate resolution diagnostic differs from inputs')
        elif 'initial_rate_resolution_ratio' in frame or 'initial_rate_resolution_valid' in frame:
            raise ValueError('initial rate resolution ratio belongs only to first step')
    if settings.polar_coupling != 'none':
        if previous is not None and previous.get('polar_source_sha256') != frame['polar_source_sha256']:
            raise ValueError('coupled source polar identity changed inside archive')
        trace = json.loads(bytes(frame['polar_trace_json']).decode('utf-8'))
        if (frame.get('polar_coupling_contract') == 'grid_section_v2'
                and 'initial_virtual_incidence_rad' not in trace[0]):
            raise ValueError('coupled v2 archive requires the actual initial seed')
        if 'initial_virtual_incidence_rad' in trace[0]:
            seed = (np.full(len(frame['circulation']), np.deg2rad(settings.polar_initial_virtual_incidence_deg))
                    if method == 'vlm' or step == 0 else previous['polar_delta_rad'])
            if not np.array_equal(trace[0]['initial_virtual_incidence_rad'], seed):
                raise ValueError('coupled initial seed differs from declared continuation')
    if method == 'vlm':
        if settings.wake_release != 'previous' or settings.gamma_derivative != 'backward_euler':
            raise ValueError('transient time settings in straight VLM archive')
        rate = np.zeros_like(gamma)
    else:
        history = frame.get('previous_lattice_circulation_m2_s')
        if history is None or history.shape != ((0,) if step == 0 else gamma.shape):
            raise ValueError('invalid archived previous lattice circulation history')
        wake = frame['wake_circulation_m2_s']
        if len(wake) != step or len(frame['release_times_s']) != step:
            raise ValueError('archived lattice time/wake row count disagrees')
        if step == 0:
            rate = np.zeros_like(gamma) if initial_rate is None else initial_rate.rate_m2_s2.ravel()
        else:
            if previous is None or previous['step'] != step-1:
                raise ValueError('missing adjacent lattice time history')
            if (previous['lattice_settings_json'] != frame['lattice_settings_json'] or
                    previous['wake_config_json'] != frame['wake_config_json']):
                raise ValueError('lattice time configuration changed inside archive')
            if not np.array_equal(history, previous['lattice_circulation_m2_s']):
                raise ValueError('previous lattice circulation differs from adjacent frame')
            dt = json.loads(frame['wake_config_json'])['dt_s']
            if not np.isclose(frame['time_s']-previous['time_s'], dt, rtol=0,
                              atol=8*np.finfo(float).eps*max(1., abs(frame['time_s']))):
                raise ValueError('lattice time interval differs from configured dt')
            if frame['release_times_s'][0] != previous['time_s']:
                raise ValueError('lattice release time differs from adjacent frame')
            rate = (gamma-history)/dt
            if initial_rate is not None and step == 1:
                rate = 2*(gamma-history)/dt-initial_rate.rate_m2_s2.ravel()
            if settings.gamma_derivative in ('bdf2', 'bdf2_consistent') and step >= 2:
                older = previous.get('previous_lattice_circulation_m2_s')
                if older is None or older.shape != gamma.shape:
                    raise ValueError('missing BDF2 two-back lattice history')
                rate = (3*gamma-4*history+older)/(2*dt)
            if settings.wake_basis == 'linear_potential':
                if not np.array_equal(frame['wake_node_potential_m2_s'][1:],
                                      previous['wake_node_potential_m2_s']):
                    raise ValueError('lattice accepted nodal potential history is inconsistent')
                if not np.array_equal(frame['release_times_s'],
                                      np.r_[previous['time_s'], previous['release_times_s']]):
                    raise ValueError('lattice nodal potential release-time history is inconsistent')
            else:
                theta = {'previous':0., 'current':1., 'trapezoidal':.5}[settings.wake_release]
                strips = len(frame['circulation'])
                released = ((1-theta)*history[-strips:]+theta*gamma[-strips:])
                if (not np.array_equal(wake[0], released) or
                        not np.array_equal(wake[1:], previous['wake_circulation_m2_s'])):
                    raise ValueError('lattice released circulation history is inconsistent')
    if not np.array_equal(rate, frame['lattice_gamma_rate_m2_s2']):
        raise ValueError('lattice Gamma rate differs from its time stencil/history')
    newest = (frame['wake_node_potential_m2_s'][0] if settings.wake_basis == 'linear_potential' and step else
              frame['circulation'] if method == 'vlm' else
              frame['wake_circulation_m2_s'][0] if step else np.zeros_like(frame['circulation']))
    if not np.array_equal(frame['trailing_edge_jump_m2_s'], frame['circulation']-newest):
        raise ValueError('lattice trailing-edge jump differs from saved time history')
    frame['lattice_time_status'] = ('explicit_history_checked_v2' if settings.wake_basis == 'linear_potential'
                                    else 'explicit_history_checked_v1')
    if initial_rate is not None:
        frame['lattice_time_status'] = 'explicit_history_checked_v3'


def _lattice_resolution_fields(frame):
    """Nominal prescribed-flow step per local geometric chord cell; no gate."""
    cells = frame['lattice_nodes_m'][frame['lattice_cells']]
    chord = .5*(cells[:, 3]+cells[:, 2]-cells[:, 0]-cells[:, 1])
    lengths = np.linalg.norm(chord, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('invalid lattice chordwise cell length')
    values = {'minimum_chordwise_cell_length_m':float(np.min(lengths))}
    if 'dt_s' not in frame:
        return values  # Old frames need not have an actual saved time-step policy.
    dt = float(frame['dt_s'])
    relative = frame['lattice_relative_velocity_m_s']-frame['lattice_induced_velocity_m_s']
    ratio = dt*np.linalg.norm(relative, axis=1)/lengths
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(ratio).all():
        raise ValueError('invalid lattice prescribed-flow resolution diagnostic')
    values.update(prescribed_step_to_chord_ratio=float(np.max(ratio)),
        prescribed_step_to_chord_definition='max over cells of nominal dt * prescribed |wind-body velocity| at CP / local center-chord cell length; not actual wake convection or a stability/accuracy guarantee')
    return values


class AeroWriter:
    """Archive every accepted step from zero; local cores require the configured dt.

    Own a new run directory; incomplete/failed runs cannot open as replay.
    Sparse output and starting a new local-core archive at a resumed step are
    unsupported because physical birth/stretch validation needs adjacent states.
    """
    def __init__(self,directory,case):
        case=json.loads(json.dumps(case,sort_keys=True,allow_nan=False))
        self.method=AeroMethod(case.setdefault('aero_method','free_wake')).value
        self.schema=LATTICE_SCHEMA if self.method in LATTICE_METHODS else SCHEMA
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=False)
        self.source=aero_source_identity()
        self.case=case
        self.manifest={'schema':self.schema,'status':'running','created_utc':utc_now(),
                       'aero_method':self.method,
                       'qualification':'Prescribed rigid lifting surfaces; selected aerodynamic component. No flight/tether/winch validation.',
                       'frame':'right-handed Cartesian metres: x downstream, y span, z up',
                       'source':self.source,'environment':environment(),'frames':0}
        if self.method in LATTICE_METHODS:
            self.manifest['geometry_replay_policy'] = (
                'exact_float_rebuild_v1: geometry and vortex graph must reconstruct exactly; '
                'use the writing numerical environment if NumPy/libm/CPU rounding differs; '
                'cross-environment portability is not guaranteed')
        write_json(self.directory/'case.json',case)
        self.manifest['case_sha256']=sha256_file(self.directory/'case.json')
        write_json(self.directory/'manifest.json',self.manifest)
        self.file=h5py.File(self.directory/'wake.h5','x')
        self.file.attrs['schema']=self.schema
        self.file.attrs['aero_method']=self.method
        self.file.attrs['position_unit']='m';self.file.attrs['time_unit']='s'
        self.file.attrs['force_unit']='N';self.file.attrs['circulation_unit']='m^2/s'
        self.file.attrs['velocity_unit']='m/s';self.file.attrs['moment_unit']='N m'
        self.file.attrs['angle_unit']='rad';self.file.attrs['coefficient_unit']='1'
        for name,value in field_semantics(self.method, separate_vsm_load=True).items():
            if self.method in ('vsm','vsm_free_wake') and name in (
                    'vsm_load_ambient', 'load_point_ambient_sampling', 'alpha_rad_definition'):
                continue  # Only a state can establish the sampling convention.
            self.file.attrs[name]=value
        if self.method in LATTICE_METHODS:
            self.file.attrs['boundary_residual_unit']='m/s'
            self.file.attrs['circulation_rate_unit']='m^2/s^2'
            self.file.attrs['section_polar_used']=False
        self.file.create_group('frames')
        self.count=0;self.last_time=None

    def _state_attributes(self, state):
        config = state.config
        attributes = dict(wake_config_json=json.dumps(wake_config_metadata(config), sort_keys=True, allow_nan=False),
            dt_s=config.dt_s, core_model=int(config.core_model),
            initial_core_radius_m=config.core_radius_m,
            core_evolution=config.core_evolution,
            stage_bound_interpolation='linear_endpoint_geometry; exact for translation, approximate for rotation')
        attributes.update(_local_core_attributes(config))
        attributes.update(field_semantics(self.method, separate_vsm_load=True))
        if self.method in ('vsm', 'vsm_free_wake'):
            attributes.update(field_semantics(self.method,
                separate_vsm_load=state.solution.load_ambient_sampling == 'separate_load_input_v1'))
        if self.count:
            previous = self.file['frames'][f'{self.count-1:06d}']
            if self.method in LATTICE_METHODS:
                points, body, key = state.mesh.collocation_points, state.lattice_structural_velocity, 'lattice_control_points_m'
            elif self.method in ('vsm', 'vsm_free_wake'):
                points, body, key = state.solution.collocation_points, state.structural_velocity, 'collocation_points'
            else:
                points, body, key = state.wing.cp, state.structural_velocity, 'control_points'
            secant = (points-previous[key][:])/(state.time_s-self.last_time)
            attributes['kinematic_mismatch_m_s'] = float(np.max(np.linalg.norm(secant-body, axis=1)))
            attributes['kinematic_mismatch_definition'] = 'Endpoint closure-point velocity minus finite-step geometry secant; diagnostic, not a rigid-rotation error threshold'
        return attributes

    def append(self,state):
        if self.method in LATTICE_METHODS:
            self._append_lattice(state)
            return
        from .aero.vsm_wake import VsmWakeState
        expected={'vsm':VsmState, 'free_wake':WakeState, 'vsm_free_wake':VsmWakeState,
                  'll_straight':LLStraightState}[self.method]
        if not isinstance(state,expected) or (self.method=='free_wake' and isinstance(state,VsmWakeState)):
            raise ValueError('state and archived aerodynamic method differ')
        if state.config.wake_core_chord_fraction is not None and state.step != self.count:
            raise ValueError(_LOCAL_CORE_SEQUENCE_ERROR)
        if state.step != self.count or (self.last_time is not None and state.time_s <= self.last_time):
            raise ValueError("archive requires every accepted state once in increasing physical time")
        attributes={'time_s':state.time_s,'step':state.step,
                    'iterations':state.solution.iterations,
                    'residual_m2_s':state.solution.residual,
                    'residual_limit_m2_s':state.solution.residual_limit,
                    'aero_method':self.method,
                    'vsm_display_wake_length_m':self.case.get('vsm_display_wake_length_m') or 20.}
        attributes.update(self._state_attributes(state))
        if self.method == 'vsm':
            derivative = state.settings.jacobian
            declared_derivative = _declared_vsm_jacobian(self.case)
            if declared_derivative is not None and declared_derivative != derivative:
                raise ValueError('case and state VSM Jacobian policies differ')
            if self.count and self.manifest.get('vsm_jacobian_policy','finite_difference') != derivative:
                raise ValueError('VSM Jacobian policy changed in the archived run')
            if state.solution.solver_trace_json is not None:
                trace = json.loads(state.solution.solver_trace_json)
                if trace.get('jacobian','finite_difference') != derivative:
                    raise ValueError('VSM solver trace contradicts its Jacobian policy')
            elif derivative != 'finite_difference' and np.any(state.solution.circulation):
                raise ValueError('active VSM Newton result lacks a solver trace')
            fields = {'vsm_core_policy':state.core_policy,
                      'vsm_core_kernel_identity':core_kernel_identity(state.core_policy)}
            if derivative != 'finite_difference':
                fields['vsm_jacobian_policy'] = derivative
            declared = (self.case.get('aerodynamics') or {}).get('core_policy')
            if declared is not None and declared != state.core_policy:
                raise ValueError('case and state VSM core policies differ')
            if self.count:
                for key,value in fields.items():
                    if self.manifest.get(key) != value:
                        raise ValueError('VSM core policy changed in the archived run')
            else:
                self.manifest.update(fields)
                for key,value in fields.items():
                    self.file.attrs[key] = value
            attributes.update(fields)
        if state.solution.solver_trace_json is not None:
            attributes['solver_trace_json'] = state.solution.solver_trace_json
        _reject_non_vsm_jacobian(self.method, self.case, attributes, self.file.attrs, self.manifest)
        arrays={'leading_edge':state.wing.leading_edge,'trailing_edge':state.wing.trailing_edge,
                'panel_edges':state.wing.panel_edges,
                'control_points':state.wing.cp,'structural_velocity':state.structural_velocity,'free_rows':state.free_rows,
                'release_times_s':state.release_times_s,'older_circulation':state.older_circulation,
                'circulation':state.solution.circulation,'alpha_rad':state.solution.alpha_rad,
                'reynolds':state.solution.reynolds,'cl':state.solution.cl,'cd':state.solution.cd,
                'cm':state.solution.cm,'relative_velocity':state.solution.relative_velocity,
                'induced_velocity':state.solution.induced_velocity,'panel_force':state.solution.panel_force,
                'panel_couple':state.solution.panel_couple}
        camber_attributes, camber_arrays = _camber_fields(state.wing, self.method)
        attributes.update(camber_attributes); arrays.update(camber_arrays)
        if self.method in ('vsm','vsm_free_wake'):
            arrays.update(polar_alpha_rad=state.solution.polar_alpha_rad,
                          collocation_points=state.solution.collocation_points,
                          closure_ambient_velocity=state.solution.closure_ambient_velocity)
        if self.method=='vsm':
            arrays['prescribed_wake_direction']=state.solution.wake_direction
        elif self.method=='ll_straight':
            nodes, edges, strengths = straight_network(state)
            arrays.update(prescribed_wake_row_m=state.prescribed_wake_row,
                network_nodes_m=nodes, network_edges=edges,
                network_circulation_m2_s=strengths, network_core_radius_m=network_radii(state))
            attributes['prescribed_wake_length_m']=state.settings.wake_length_m
            attributes['ll_straight_settings_json']=json.dumps(asdict(state.settings),sort_keys=True)
            attributes['core_evolution']='constant'
            attributes['section_polar_used']=True
        elif state.cores is not None:
            attributes['core_evolution']=state.cores.policy
            attributes['initial_core_radius_m']=state.cores.initial_radius
            attributes['core_diffusion_rate_m2_s']=state.cores.diffusion_rate
            arrays.update(span_core_radius_m=state.cores.span, stream_core_radius_m=state.cores.stream,
                          network_core_radius_m=edge_radii(state.wing, wake_network(state)[1],
                                                           state.cores, state.cores.initial_radius))
        if self.method in ('free_wake', 'vsm_free_wake'):
            nodes, edges, strengths = wake_network(state)
            radii = edge_radii(state.wing, edges, state.cores, state.config.core_radius_m)
            if int(state.config.core_model) == 0:
                radii = np.zeros_like(radii)
            arrays.update(network_nodes_m=nodes, network_edges=edges,
                network_circulation_m2_s=strengths, network_core_radius_m=radii)
        if 'alpha_deg' in self.case.get('polar',{}):
            alpha=arrays.get('polar_alpha_rad',state.solution.alpha_rad)
            limits=np.deg2rad(self.case['polar']['alpha_deg'])
            arrays['polar_clipped']=(alpha<limits[0]) | (alpha>limits[-1])
        elif 'panel_polar_bounds_deg' in self.case:
            alpha=arrays.get('polar_alpha_rad',state.solution.alpha_rad)
            limits=np.deg2rad(self.case['panel_polar_bounds_deg'])
            arrays['polar_clipped']=(alpha<limits[:,0]) | (alpha>limits[:,1])
        for name,array in arrays.items():
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite archive field: {name}")
        _validate_common_frame(dict(arrays, **attributes))
        if self.method in ('free_wake', 'vsm_free_wake'):
            _validate_wake_frame(dict(arrays, **attributes))
            _validate_local_core_history(dict(arrays, **attributes),
                _lattice_previous_frame(self.file['frames'], self.count), archive_index=self.count)
        if self.method=='ll_straight':
            _validate_straight_ll_frame(dict(arrays,**attributes))
        group=self.file['frames'].create_group(f'{self.count:06d}')
        for name,value in attributes.items():
            group.attrs[name]=value
        for name,array in arrays.items():
            group.create_dataset(name,data=array,compression='gzip' if array.size else None)
        self.file.flush();self.count+=1;self.last_time=state.time_s

    def _append_lattice(self,state):
        from .aero.lifting_surface import VlmState,VlmWakeState,lifting_surface_network,lifting_surface_edge_radii
        expected=VlmState if self.method=='vlm' else VlmWakeState
        if type(state) is not expected:
            raise ValueError('state and archived aerodynamic method differ')
        if state.config.wake_core_chord_fraction is not None and state.step != self.count:
            raise ValueError(_LOCAL_CORE_SEQUENCE_ERROR)
        if state.step!=self.count or (self.last_time is not None and state.time_s<=self.last_time):
            raise ValueError('archive requires every accepted state once in increasing physical time')
        solution,mesh=state.solution,state.mesh
        nodes,edges,gamma=lifting_surface_network(state)
        saved_settings = asdict(state.settings)
        if state.settings.bound_convection_core != 'all_bound_chord':
            saved_settings.pop('bound_convection_chord_fraction')
        if state.settings.bound_convection_core == 'legacy_zero':
            saved_settings.pop('bound_convection_core')
        if state.settings.initial_cap == 'legacy_zero':
            saved_settings.pop('initial_cap')
        attributes={'time_s':state.time_s,'step':state.step,'iterations':solution.iterations,
                    'aero_method':self.method,'section_polar_used':False,
                    'boundary_residual_m_s':solution.boundary_residual_m_s,
                    'boundary_residual_limit_m_s':solution.boundary_residual_limit_m_s,
                    'matrix_condition_number':solution.matrix_condition_number,
                    'chordwise_spacing':state.settings.chordwise_spacing,
                    'lattice_settings_json':json.dumps(saved_settings,sort_keys=True,allow_nan=False),
                    'wake_config_json':json.dumps(wake_config_metadata(state.config),sort_keys=True,allow_nan=False),
                    'force_reference':'Lattice resultants and couples about each collocation point',
                    'bound_core_model':'ideal finite segments; wake cores use recorded WakeConfig'}
        attributes.update(self._state_attributes(state))
        if state.settings.steady_mirror_projection:
            attributes['mirror_preprojection_max_abs_m2_s'] = solution.mirror_preprojection_max_abs_m2_s
            attributes['unprojected_boundary_residual_m_s'] = solution.unprojected_boundary_residual_m_s
        attributes['condition_roundoff_indicator'] = solution.matrix_condition_number*np.finfo(float).eps
        attributes.update(_lattice_time_metadata(state.settings, state.step, self.method))
        attributes['condition_limit_status'] = ('not_configured' if state.settings.max_condition_number is None else 'passed')
        arrays={'leading_edge':state.wing.leading_edge,'trailing_edge':state.wing.trailing_edge,
                'panel_edges':state.wing.panel_edges,'control_points':state.wing.cp,
                'structural_velocity':state.structural_velocity,'free_rows':state.free_rows,
                'release_times_s':state.release_times_s,'older_circulation':state.older_circulation,
                'circulation':solution.circulation,'alpha_rad':solution.alpha_rad,
                'reynolds':solution.reynolds,'cl':solution.cl,'cd':solution.cd,'cm':solution.cm,
                'relative_velocity':solution.relative_velocity,'induced_velocity':solution.induced_velocity,
                'panel_force':solution.panel_force,'panel_couple':solution.panel_couple,
                'lattice_nodes_m':mesh.vertices,'lattice_cells':mesh.cells,
                'lattice_control_points_m':mesh.collocation_points,'lattice_normals':mesh.normals,
                'lattice_force_reference_points_m':mesh.collocation_points,
                'lattice_centroids_m':mesh.centroids,'lattice_area_normals':mesh.area_normals,
                'lattice_area_m2':mesh.area,'lattice_strip_index':mesh.strip_index,
                'lattice_chord_index':mesh.chord_index,'lattice_vortex_rows_m':mesh.vortex_rows,
                'lattice_circulation_m2_s':solution.lattice_circulation.reshape(-1),
                'lattice_panel_force_N':solution.lattice_force.reshape(-1,3),
                'lattice_panel_couple_N_m':solution.lattice_couple.reshape(-1,3),
                'lattice_relative_velocity_m_s':solution.lattice_relative_velocity.reshape(-1,3),
                'lattice_induced_velocity_m_s':solution.lattice_induced_velocity.reshape(-1,3),
                'lattice_gamma_rate_m2_s2':solution.lattice_gamma_rate.reshape(-1),
                'lattice_boundary_residual_m_s':solution.lattice_boundary_residual_m_s.reshape(-1),
                'lattice_structural_velocity_m_s':state.lattice_structural_velocity,
                'trailing_edge_jump_m2_s':solution.trailing_edge_jump_m2_s,
                'network_nodes_m':nodes,'network_edges':edges,'network_circulation_m2_s':gamma,
                'network_core_radius_m':lifting_surface_edge_radii(state)}
        if solution.polar_coupling is not None:
            self.schema = POLAR_LATTICE_SCHEMA
            self.file.attrs['schema'] = self.schema
            self.manifest['schema'] = self.schema
            attributes['polar_coupling_contract'] = 'grid_section_v2'
            from .aero.polar_coupling import archive_fields
            coupled_attributes, coupled_arrays = archive_fields(solution.polar_coupling)
            attributes.update(coupled_attributes); arrays.update(coupled_arrays)
            expected_polar = self.case.get('coupled_source_polars_sha256')
            if expected_polar is not None and expected_polar != coupled_attributes['polar_source_sha256']:
                raise ValueError('coupled source polar differs from declared case input')
            attributes['polar_input_binding'] = 'case_hash_checked' if expected_polar is not None else 'unrecorded'
            semantics = field_semantics(self.method, polar_coupled=True)
            attributes.update(semantics)
            for name,value in semantics.items():
                self.file.attrs[name] = value
            self.manifest['polar_coupling_model'] = 'grid_section'
        camber_attributes, camber_arrays = _camber_fields(state.wing, self.method)
        attributes.update(camber_attributes); arrays.update(camber_arrays)
        if self.method=='vlm':
            arrays['prescribed_wake_row_m']=state.prescribed_row
        else:
            arrays['wake_circulation_m2_s']=state.wake_circulation
            if state.settings.wake_basis == 'linear_potential':
                self.schema = (POTENTIAL_POLAR_LATTICE_SCHEMA if solution.polar_coupling is not None
                               else POTENTIAL_LATTICE_SCHEMA)
                self.file.attrs['schema'] = self.schema
                self.manifest.update(schema=self.schema, wake_basis='linear_potential',
                                     wake_potential_contract='material_nodes_v1')
                if state.config.free_wake:
                    from .aero.lifting_surface import linear_wake_convection_contract
                    contract = linear_wake_convection_contract(state.settings)
                    attributes['linear_wake_convection_contract'] = contract
                    self.manifest['linear_wake_convection_contract'] = contract
                if int(state.config.core_model) != 0:
                    attributes['wake_core_mapping'] = 'material_edge_q_v1'
                    self.manifest['wake_core_mapping'] = 'material_edge_q_v1'
                arrays['wake_node_potential_m2_s'] = state.wake_node_potential_m2_s
            arrays['previous_lattice_circulation_m2_s'] = (
                np.empty(0) if state.previous_lattice_circulation is None else
                state.previous_lattice_circulation.reshape(-1))
            if state.cores is not None:
                attributes.update(core_evolution=state.cores.policy,
                                  initial_core_radius_m=state.cores.initial_radius,
                                  core_diffusion_rate_m2_s=state.cores.diffusion_rate)
                arrays.update(span_core_radius_m=state.cores.span,
                              stream_core_radius_m=state.cores.stream)
        if state.settings.initial_cap == 'wake_core':
            self.schema = INITIAL_CAP_LATTICE_SCHEMA
            self.file.attrs['schema'] = self.schema
            attributes['initial_cap_contract'] = 'closed_te_birth_core_v1'
            attributes['bound_core_model'] = 'ideal bound edges; zero-history closing TE uses recorded birth core'
            self.manifest.update(schema=self.schema, wake_basis=state.settings.wake_basis,
                initial_cap_contract='closed_te_birth_core_v1')
        convection_metadata = _bound_convection_metadata(state.settings, state.config)
        attributes.update(convection_metadata)
        self.manifest.update(convection_metadata)
        if self.method == 'vlm_free_wake' and state.initial_gamma_rate is not None:
            from .aero.initial_rate import archive_fields, startup_ratio
            supplied = state.initial_gamma_rate
            initial_attributes, initial_arrays = archive_fields(supplied)
            attributes.update(initial_attributes); arrays.update(initial_arrays)
            saved_input = supplied.to_dict()
            if self.count and self.manifest.get('initial_gamma_rate_input') != saved_input:
                raise ValueError('initial Gamma rate input changed inside archive')
            self.schema = INITIAL_RATE_LATTICE_SCHEMA
            self.file.attrs['schema'] = self.schema
            self.manifest.update(schema=self.schema, initial_gamma_rate_input=saved_input,
                                 initial_gamma_rate_contract='provided_d0_bdf2_start_v1')
            if state.step == 1:
                ratio, valid = startup_ratio(supplied, solution.lattice_circulation,
                    state.previous_lattice_circulation, state.config.dt_s)
                arrays.update(initial_rate_resolution_ratio=ratio.ravel(),
                              initial_rate_resolution_valid=valid.ravel())
        for name,array in arrays.items():
            if not np.isfinite(array).all():
                raise ValueError(f'nonfinite archive field: {name}')
        attributes.update(_lattice_resolution_fields(dict(arrays,**attributes)))
        _reject_non_vsm_jacobian(self.method, self.case, attributes, self.file.attrs, self.manifest)
        _validate_common_frame(dict(arrays,**attributes))
        _validate_lattice_frame(dict(arrays,**attributes))
        _validate_local_core_history(dict(arrays, **attributes),
            _lattice_previous_frame(self.file['frames'], self.count), archive_index=self.count)
        _validate_lattice_time(dict(arrays, **attributes),
                               _lattice_previous_frame(self.file['frames'], self.count))
        group=self.file['frames'].create_group(f'{self.count:06d}')
        for name,value in attributes.items():
            group.attrs[name]=value
        for name,array in arrays.items():
            group.create_dataset(name,data=array,compression='gzip' if array.size else None)
        self.file.flush();self.count+=1;self.last_time=state.time_s

    def finish(self):
        if not self.count: raise ValueError("cannot complete an empty run")
        if aero_source_identity()['files_sha256'] != self.source['files_sha256']:
            raise ValueError("source changed during the aerodynamic run")
        self.file.attrs['frames']=self.count;self.file.close()
        self.manifest.update(status='complete',finished_utc=utc_now(),frames=self.count,
                             wake_sha256=sha256_file(self.directory/'wake.h5'))
        write_json(self.directory/'manifest.json',self.manifest)

    def fail(self,error):
        if self.file:
            self.file.close()
        self.manifest.update(status='failed',finished_utc=utc_now(),frames=self.count,
                             error_type=type(error).__name__,error=str(error))
        if hasattr(error, 'geometry_report'):
            self.manifest['geometry_guard_failure'] = error.geometry_report
        if hasattr(error, 'trace'):
            self.manifest['solver_trace'] = error.trace
        diagnostics = getattr(error, 'diagnostics', None)
        if diagnostics:
            arrays = {name:diagnostics[name] for name in (
                'circulation', 'residual', 'jacobian', 'direction', 'linearization_circulation')
                if name in diagnostics}
            saved = {name:diagnostics[name] for name in (
                'updates', 'merit', 'linear_solve', 'domain_rejected', 'no_decrease', 'history')
                if name in diagnostics}
            saved['contract'] = 'last_newton_state_v1; not a Picard iterate or a completed replay'
            if arrays:
                path = self.directory/f'solver-diagnostics-{uuid4().hex}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream, **arrays)
                saved.update(arrays_file=path.name, arrays_sha256=sha256_file(path),
                             array_names=sorted(arrays))
            self.manifest['solver_diagnostics'] = saved
        write_json(self.directory/'manifest.json',self.manifest)


class AeroReader:
    """Verify completion and content hashes before exposing copied replay frames."""
    def __init__(self,directory):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text(encoding='utf-8'))
        schema=self.manifest.get('schema')
        if schema not in SUPPORTED_SCHEMAS or self.manifest.get('status') != 'complete':
            raise ValueError("not a complete aerodynamic wake archive")
        for file,key in [('case.json','case_sha256'),('wake.h5','wake_sha256')]:
            if sha256_file(self.directory/file) != self.manifest.get(key):
                raise ValueError(f"archive hash mismatch: {file}")
        self.case=json.loads((self.directory/'case.json').read_text(encoding='utf-8'))
        if self.case.get('counterfactual') is not None:
            raise ValueError('quarantined counterfactual cannot open as a native aerodynamic run')
        self.method=AeroMethod(self.case.get('aero_method','free_wake')).value
        if (schema in LATTICE_SCHEMAS)!=(self.method in LATTICE_METHODS):
            raise ValueError('aerodynamic method does not match lattice archive schema')
        self.file=h5py.File(self.directory/'wake.h5','r')
        try:
            self.count=self.manifest['frames']
            if type(self.count) is not int or self.count<1 or self.file.attrs.get('schema')!=schema or self.file.attrs.get('frames')!=self.count:
                raise ValueError("invalid frame count/schema")
            if schema in (SCHEMA,*LATTICE_SCHEMAS) and (self.manifest.get('aero_method')!=self.method or self.file.attrs.get('aero_method')!=self.method):
                raise ValueError('inconsistent aerodynamic method metadata')
            if sorted(self.file['frames']) != [f'{i:06d}' for i in range(self.count)]:
                raise ValueError("incomplete frame sequence")
            self.times=np.array([self.file['frames'][f'{i:06d}'].attrs['time_s'] for i in range(self.count)])
            if not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):
                raise ValueError("invalid physical time sequence")
            self.forces=np.array([self.frame(i)['panel_force'].sum(axis=0) for i in range(self.count)])
        except BaseException:
            self.file.close();raise

    def frame(self,index):
        if isinstance(index,(bool,np.bool_)) or not isinstance(index,(int,np.integer)) or not 0<=index<self.count:
            raise ValueError("frame index out of range")
        group=self.file['frames'][f'{index:06d}']
        frame={name:dataset[:] for name,dataset in group.items()}
        frame.update(dict(group.attrs))
        if frame.get('aero_method',self.method)!=self.method:
            raise ValueError('frame method differs from run')
        frame['aero_method']=self.method
        _reject_non_vsm_jacobian(self.method, self.case, frame, self.file.attrs, self.manifest)
        if self.method == 'vsm':
            derivative = frame.get('vsm_jacobian_policy')
            root_derivative = self.file.attrs.get('vsm_jacobian_policy')
            manifest_derivative = self.manifest.get('vsm_jacobian_policy')
            declared_derivative = _declared_vsm_jacobian(self.case)
            if derivative is None:
                if root_derivative is not None or manifest_derivative is not None or declared_derivative not in (None,'finite_difference'):
                    raise ValueError('VSM Jacobian archive lacks a frame policy')
                derivative = 'finite_difference'  # Legacy Newton arithmetic.
            elif (derivative not in ('finite_difference','active_segment')
                    or root_derivative != derivative or manifest_derivative != derivative):
                raise ValueError('inconsistent VSM Jacobian archive policies')
            if declared_derivative is not None and declared_derivative != derivative:
                raise ValueError('case and archived VSM Jacobian policies differ')
            trace = json.loads(frame['solver_trace_json']) if 'solver_trace_json' in frame else None
            if trace is not None and trace.get('jacobian','finite_difference') != derivative:
                raise ValueError('archived trace contradicts its Jacobian policy')
            if derivative == 'active_segment' and trace is None and np.any(frame['circulation']):
                raise ValueError('active VSM Newton archive lacks a solver trace')
            marker = frame.get('vsm_core_policy')
            declared = (self.case.get('aerodynamics') or {}).get('core_policy')
            if declared is not None:
                core_kernel_identity(declared)
            if marker is None:
                if (any(frame.get(key) is not None or self.manifest.get(key) is not None
                        or self.file.attrs.get(key) is not None
                        for key in ('vsm_core_policy','vsm_core_kernel_identity'))
                        or declared == 'source_pinned_v1'):
                    raise ValueError('explicit VSM core archive lacks frame policy')
                frame['vsm_core_policy'] = 'unrecorded'
                frame['vsm_core_kernel_identity'] = 'unrecorded'
            else:
                identity = core_kernel_identity(marker)
                for key,value in [('vsm_core_policy',marker),('vsm_core_kernel_identity',identity)]:
                    if (frame.get(key) != value or self.manifest.get(key) != value
                            or self.file.attrs.get(key) != value):
                        raise ValueError('inconsistent archived VSM core identity')
                if declared is not None and declared != marker:
                    raise ValueError('case and frame VSM core policies differ')
        schema = self.manifest['schema']
        lattice_settings = json.loads(frame.get('lattice_settings_json', '{}'))
        explicit_rate = schema == INITIAL_RATE_LATTICE_SCHEMA
        capped = schema in (INITIAL_CAP_LATTICE_SCHEMA, INITIAL_RATE_LATTICE_SCHEMA)
        linear = lattice_settings.get('wake_basis') == 'linear_potential'
        if self.method in LATTICE_METHODS:
            from .aero.lifting_surface import LiftingSurfaceConfig
            from types import SimpleNamespace
            settings = LiftingSurfaceConfig(**lattice_settings)
            convection_metadata = _bound_convection_metadata(settings,
                SimpleNamespace(**json.loads(frame['wake_config_json'])))
            for name, value in convection_metadata.items():
                if self.manifest.get(name) != value or frame.get(name) != value:
                    raise ValueError('bound convection core manifest/frame binding is inconsistent')
            if not convection_metadata and any(name in self.manifest for name in (
                    'bound_convection_core', 'convection_bound_core_radius_m')):
                raise ValueError('bound convection core manifest evidence stored for a legacy field')
        coupled = lattice_settings.get('polar_coupling', 'none') == 'grid_section'
        if explicit_rate:
            from .aero.initial_rate import from_archive
            if (self.method != 'vlm_free_wake' or lattice_settings.get('gamma_derivative') != 'bdf2_consistent' or
                    self.manifest.get('initial_gamma_rate_contract') != 'provided_d0_bdf2_start_v1'):
                raise ValueError('v8 initial Gamma rate archive lacks its mandatory contract')
            supplied = from_archive(frame, (lattice_settings['chordwise_panels'], len(frame['circulation'])))
            if supplied.to_dict() != self.manifest.get('initial_gamma_rate_input'):
                raise ValueError('initial Gamma rate input differs from manifest')
        elif (lattice_settings.get('gamma_derivative') == 'bdf2_consistent' or
                any(name.startswith('initial_gamma_rate') or name.startswith('initial_rate_resolution') for name in frame) or
                any(name.startswith('initial_gamma_rate') for name in self.manifest)):
            raise ValueError('explicit initial Gamma rate requires the v8 archive schema')
        if capped:
            if (self.method != 'vlm_free_wake' or lattice_settings.get('initial_cap') != 'wake_core' or
                    self.manifest.get('initial_cap_contract') != 'closed_te_birth_core_v1' or
                    frame.get('initial_cap_contract') != 'closed_te_birth_core_v1' or
                    self.manifest.get('wake_basis') != lattice_settings.get('wake_basis', 'constant')):
                raise ValueError('v7 initial cap archive lacks its consistent settings/contract')
        elif (lattice_settings.get('initial_cap', 'legacy_zero') != 'legacy_zero' or
                frame.get('initial_cap_contract') is not None or self.manifest.get('initial_cap_contract') is not None):
            raise ValueError('explicit initial cap requires the v7 archive schema')
        if schema in POTENTIAL_SCHEMAS or (capped and linear):
            if (self.method != 'vlm_free_wake' or self.manifest.get('wake_basis') != 'linear_potential' or
                    self.manifest.get('wake_potential_contract') != 'material_nodes_v1' or
                    json.loads(frame['lattice_settings_json']).get('wake_basis') != 'linear_potential'):
                raise ValueError('v5 potential archive lacks its mandatory nodal contract')
            if (int(json.loads(frame['wake_config_json'])['core_model']) != 0 and
                    self.manifest.get('wake_core_mapping') != 'material_edge_q_v1'):
                raise ValueError('v5 regularized potential archive lacks its material core mapping')
            from .aero.lifting_surface import linear_wake_convection_contract
            expected_convection = (linear_wake_convection_contract(settings)
                if json.loads(frame['wake_config_json'])['free_wake'] else None)
            if self.manifest.get('linear_wake_convection_contract') != expected_convection:
                raise ValueError('v5 free potential archive lacks its convection contract')
        elif json.loads(frame.get('lattice_settings_json', '{}')).get('wake_basis') == 'linear_potential':
            raise ValueError('linear potential history requires the v5/v6 archive schema')
        if schema == POTENTIAL_POLAR_LATTICE_SCHEMA or (capped and linear and coupled):
            if (lattice_settings.get('polar_coupling') != 'grid_section' or
                    self.manifest.get('polar_coupling_model') != 'grid_section'):
                raise ValueError('v6 potential-polar archive lacks its combined closure settings')
        elif schema == POTENTIAL_LATTICE_SCHEMA and (
                lattice_settings.get('polar_coupling', 'none') != 'none' or frame.get('section_polar_used')):
            raise ValueError('combined potential and polar history requires the v6 archive schema')
        if schema in POLAR_SCHEMAS or (capped and coupled):
            if frame.get('polar_coupling_contract') != 'grid_section_v2' or not frame.get('section_polar_used'):
                raise ValueError('v4 coupled archive lacks its mandatory closure contract')
            expected_polar = self.case.get('coupled_source_polars_sha256')
            binding = 'case_hash_checked' if expected_polar is not None else 'unrecorded'
            if frame.get('polar_input_binding') != binding or (expected_polar is not None
                    and expected_polar != frame.get('polar_source_sha256')):
                raise ValueError('coupled polar input binding differs from case')
        elif frame.get('section_polar_used') and self.method in LATTICE_METHODS:
            frame['polar_seed_provenance'] = 'legacy_v3: initial seed may be unrecorded'
        # Infer the pre-versioning VSM convention from the saved arrays, never
        # from today's writer defaults. Old file bytes remain untouched.
        marker = frame.get('vsm_load_ambient')
        has_closure = 'closure_ambient_velocity' in frame
        separate_load = (None if has_closure else False) if marker is None else {
            'legacy_closure_reuse':False, 'separate_load_input_v1':True, 'unrecorded':None}.get(marker)
        if self.method in ('vsm','vsm_free_wake'):
            if marker is not None and marker not in ('legacy_closure_reuse','separate_load_input_v1','unrecorded'):
                raise ValueError('unknown archived VSM load ambient sampling')
            if marker == 'separate_load_input_v1' and not has_closure:
                raise ValueError('explicit separate VSM sampling lacks closure ambient data')
        for name,value in field_semantics(self.method, separate_vsm_load=separate_load).items():
            frame.setdefault(name,self.file.attrs.get(name,value))
        if self.method in ('vsm', 'vsm_free_wake'):
            # Frame markers are authoritative; root metadata and array presence
            # cannot turn an unrecorded intermediate version into a known one.
            semantics = field_semantics(self.method, separate_vsm_load=separate_load)
            for name in ('vsm_load_ambient', 'load_point_ambient_sampling', 'alpha_rad_definition'):
                frame[name] = semantics[name]
        _validate_common_frame(frame)
        if self.method in ('free_wake', 'vsm_free_wake', 'vlm_free_wake'):
            _validate_local_core_history(frame, _lattice_previous_frame(self.file['frames'], index),
                                         archive_index=index)
        if self.method in ('free_wake', 'vsm_free_wake'):
            _validate_wake_frame(frame)
        if self.method in LATTICE_METHODS:
            settings = json.loads(frame['lattice_settings_json'])
            spacing = settings.get('chordwise_spacing', 'uniform')
            if spacing not in ('uniform', 'cosine') or frame.get('chordwise_spacing', spacing) != spacing:
                raise ValueError('invalid or inconsistent archived chordwise spacing')
            frame.setdefault('chordwise_spacing', spacing)
            _validate_lattice_frame(frame)
            if frame['step'] != index:
                raise ValueError('lattice frame index differs from saved step')
            _validate_lattice_time(frame, _lattice_previous_frame(self.file['frames'], index))
            for name,value in _lattice_resolution_fields(frame).items():
                frame.setdefault(name,value)
        if self.method=='ll_straight':
            _validate_straight_ll_frame(frame)
        if self.method in ('free_wake', 'vsm_free_wake', 'vlm_free_wake'):
            frame.setdefault('core_cumulative_log_strain_status', 'unrecorded')
        return frame

    def close(self): self.file.close()
