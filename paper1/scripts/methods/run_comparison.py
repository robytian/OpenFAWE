"""Reproducible corrected-normal lattice cases, derived from the retained
OF-0006-models20260922/evidence/run_model_aircraft.py runner.

Owner-approved optional source-polar coupling. Derived runner preserves historical jobs.
Only VLM/UVLM are recomputed. LL/VSM remain hash-identified carry-forward
results because they do not use the explicit mean surface or the new normal.
Every run is archived with actual solver configuration and time histories.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from openfawe.aero.aircraft import load_aircraft
from openfawe.aero.methods import AeroBackend
from openfawe.aero.lifting_surface import LiftingSurfaceConfig, lifting_surface_network
from openfawe.aero.ll_straight import LLStraightConfig, straight_network
from openfawe.aero.reference import inflow, load_coefficients
from openfawe.aero.wake import WakeConfig, wake_network
from openfawe.aero.wing import Wing, SectionPolar, SolveSettings
from openfawe.aero_archive import AeroWriter
from openfawe.aero.polar_coupling import encode_polars, polar_identity, polar_tables


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


METHODS = ('ll_straight', 'vsm', 'free_wake', 'vsm_free_wake', 'vlm', 'vlm_free_wake')


def jobs(group):
    if group != 'latticegap': raise ValueError(group)
    rows=[]
    for model, angles, duration, dt in (
        ('V3',(3.0810784665258444,5.412846420261046,7.350324457982788),8.,.2),
        ('M600',(-4.,0.,4.),.8,.02)):
        variants=(('v3-integral','cases/aircraft/v3_cad_camber_integral.json',METHODS[-2:]),) if model=='V3' else (
                  ('makani-camber','cases/qualification/m600_makani_reconstructed.json',METHODS[-2:]),)
        for variant, aircraft_file, methods in variants:
            for alpha in angles:
                for method in methods:
                    rows.append(dict(model=model,variant=variant,aircraft_file=aircraft_file,
                        method=method,alpha=alpha,beta=0.,duration=duration,
                        dt=duration if method in ('ll_straight','vsm','vlm') else dt,
                        chordwise_panels=8,chordwise_spacing='cosine',
                        normal_scheme='reconstructed_slope',core='constant',
                        integrator='euler',wake_length_m=1000.))
    for index,row in enumerate(rows):
        row['initial_virtual_incidence_deg'] = 2. if row['model']=='V3' and row['method']=='vlm_free_wake' else 0.
        row['name']=f'{index:03d}-{row["variant"]}-{row["method"]}-a{row["alpha"]:.5f}'
    return rows


def run_one(specification):
    job, output = specification['job'], Path(specification['output'])
    lattice = job['method'] in ('vlm', 'vlm_free_wake')
    straight = job['method'] in ('vsm', 'vlm', 'll_straight')
    if job['model'] == 'rectangle':
        y = np.linspace(-4., 4., 9)
        le = np.column_stack((np.zeros_like(y), y, np.zeros_like(y)))
        wing, polar = Wing(le, le + [1., 0., 0.]), SectionPolar.linear()
        speed, area, chord = 20., 8., 1.
        identity = dict(geometry='Analytical flat rectangular wing: 8 m span, 1 m chord, 8 span strips')
    else:
        path = ROOT / job['aircraft_file']
        definition = load_aircraft(path)
        wing, polar = definition.wing, definition.polar
        if job['variant'].startswith('makani'):
            area, chord, speed = 32.9285, 1.2831, 20.
            description = 'M600 Makani-input reconstructed wing-tail; ideal companion polars; not native AVL'
        elif job['model'] == 'M600':
            area = 33.0367429346866
            span = float(np.ptp(np.concatenate([w.bound for w in wing.wings[:2]])[:, 1]))
            chord, speed = area / span, 20.
            description = 'M600 continuous-center five-surface candidate; no body, pylons or rotors'
        else:
            area, chord = .46 * 6.5**2, .395 * 6.5
            speed = 5e5 * 1.4639e-5 / chord
            description = ('V3 rigid CAD geometry; source-integral fidelity mean '
                           'surface and fixed source CFD polars; optional grid-consistent lattice coupling')
        identity = dict(geometry=description, aircraft_input_json=definition.raw_json,
            aircraft_input_sha256=definition.source_sha256, selected_surfaces=list(wing.names),
            panel_polar_bounds_deg=[[p.alpha_deg[0],p.alpha_deg[-1]] for p in polar.polars])
    steps = round(job['duration'] / job['dt'])
    if not np.isclose(steps * job['dt'], job['duration'], rtol=0, atol=1e-12):
        raise ValueError('duration must be an integer number of physical time steps')
    wind = inflow(speed, job['alpha'], job['beta'])
    config = WakeConfig(dt_s=job['dt'], max_rows=steps, core_radius_m=.15,
        convection_integrator=job['integrator'], core_evolution=job['core'],
        solve=SolveSettings(solver='newton', max_iterations=60))
    backend = AeroBackend(job['method'], wake=config, lifting_surface=LiftingSurfaceConfig(
        chordwise_panels=job['chordwise_panels'], chordwise_spacing=job['chordwise_spacing'], polar_coupling='grid_section', polar_initial_virtual_incidence_deg=job.get('initial_virtual_incidence_deg',0.),
        camber_normal=job['normal_scheme'], wake_length_m=job['wake_length_m'],
        wake_release=job.get('release','previous'), gamma_derivative=job.get('derivative','backward_euler')),
        ll_straight=LLStraightConfig(wake_length_m=job['wake_length_m']))
    case = dict(name='OpenFAWE / ' + job['name'], model=job['model'], aero_method=job['method'],
        **identity, panels=wing.panels, alpha_deg=job['alpha'], beta_deg=job['beta'],
        aerodynamics=backend.metadata(), wake=None if straight else asdict(config),
        inflow_m_s=wind.tolist(), duration_s=job['duration'], reference_area_m2=area,
        reference_chord_m=chord, reference_origin_m=[0., 0., 0.], campaign_job=job,
        actual_surface_area_m2=float(wing.area.sum()), body_and_propeller_loads=False,
        prescribed_motion='Fixed aircraft, impulsive start for history methods',
        polar_usage='grid_section: fixed source Cl/Cd/Cm; circulation re-solved; geometric reference Cm subtracted' if lattice else ('ideal thin-airfoil companion polar' if job['variant'].startswith('makani') else 'source sectional polar closure'),
        physical_qualification=False, runner_sha256=sha(__file__))
    if lattice:
        case['coupled_source_polars_sha256'] = polar_identity(encode_polars(polar_tables(polar,wing.panels)))
    writer = AeroWriter(output, case)
    records = []
    result = dict(job=job, status='running', started_utc=utc())
    started = time.perf_counter()
    attempted = 0.
    try:
        state = backend.start(wing, polar, wind)
        for index in range(steps + 1):
            attempted = index * job['dt']
            if index:
                state = backend.advance(state, polar, wind, next_time_s=attempted)
            writer.append(state)
            solution = state.solution
            nodes, edges, strengths = ((lifting_surface_network(state) if lattice else straight_network(state) if job['method']=='ll_straight' else wake_network(state))
                                       if job['method'] != 'vsm' else (np.empty((0,3)), np.empty((0,2),int), np.empty(0)))
            divergence = np.zeros(len(nodes))
            np.add.at(divergence, edges[:,0], strengths)
            np.add.at(divergence, edges[:,1], -strengths)
            residual = solution.residual_m_s if lattice else solution.residual
            limit = solution.residual_limit_m_s if lattice else solution.residual_limit
            row = dict(time_s=state.time_s, step=index, **load_coefficients(solution, wing, wind, area, chord),
                Fx_N=float(solution.force[0]), Fy_N=float(solution.force[1]), Fz_N=float(solution.force[2]),
                residual_over_limit=float(residual / limit),
                graph_divergence_m2_s=float(np.max(np.abs(divergence), initial=0.)),
                max_abs_load_alpha_deg=float(np.max(np.abs(np.rad2deg(solution.alpha_rad)))))
            if lattice:
                row.update(boundary_residual_m_s=float(residual),
                    max_trailing_edge_jump_m2_s=float(np.max(np.abs(solution.trailing_edge_jump_m2_s), initial=0.)))
            if not lattice:
                closure=np.rad2deg(getattr(solution,'polar_alpha_rad',solution.alpha_rad))
                bounds=np.array([[p.alpha_deg[0],p.alpha_deg[-1]] for p in polar.polars])
                row.update(max_abs_closure_alpha_deg=float(np.max(np.abs(closure))),
                    closure_inside_polar=bool(np.all((closure>=bounds[:,0]-1e-10)&(closure<=bounds[:,1]+1e-10))),
                    min_forward_m_s=float(np.min(np.sum(solution.relative_velocity*wing.tangent,axis=1))),
                    attached_flow_angle_warning=bool(row['max_abs_load_alpha_deg']>20.))
            if lattice:
                coupling=solution.polar_coupling
                row.update(polar_cl_residual=float(np.max(abs(coupling.cl_residual))),
                    max_virtual_incidence_deg=float(np.rad2deg(np.max(abs(coupling.delta_rad)))),
                    max_geometric_residual_m_s=float(np.max(abs(coupling.geometric_residual_m_s))),
                    polar_iterations=coupling.iterations)
            records.append(row)
        writer.finish()
        result['status'] = 'complete'
    except (ValueError, RuntimeError, FloatingPointError) as error:
        writer.fail(error)
        result.update(status='failed', error_type=type(error).__name__, error=str(error),
                      attempted_failure_time_s=attempted)
    finally:
        if records:
            with (output / 'history.csv').open('x', encoding='utf-8', newline='') as stream:
                table = csv.DictWriter(stream, fieldnames=list(records[0]))
                table.writeheader()
                table.writerows(records)
        result.update(finished_utc=utc(), elapsed_s=time.perf_counter()-started, frames=len(records),
            last=records[-1] if records else None, history_sha256=sha(output/'history.csv') if records else None)
        if records:
            result.update(max_residual_over_limit=max(r['residual_over_limit'] for r in records),
                max_graph_divergence_m2_s=max(r['graph_divergence_m2_s'] for r in records),
                max_abs_load_alpha_deg=max(r['max_abs_load_alpha_deg'] for r in records))
        dump(output / 'result.json', result)
    return result


def campaign(output, group, workers):
    output.mkdir(parents=True, exist_ok=False)
    specifications = output / 'jobs'
    specifications.mkdir()
    plan = jobs(group)
    dump(output / 'plan.json', dict(group=group, jobs=plan, created_utc=utc(),
        runner_sha256=sha(__file__), runtime_root=str(ROOT)))
    def execute(job):
        path = specifications / (job['name'] + '.json')
        dump(path, dict(job=job, output=str((output / job['name']).resolve())))
        with path.with_suffix('.log').open('xb') as stream:
            process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', str(path)],
                stdout=stream, stderr=subprocess.STDOUT)
        result_file = output / job['name'] / 'result.json'
        result = read(result_file) if result_file.exists() else dict(job=job, status='process_failed')
        result['process_exit_code'] = process.returncode
        return result
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, job) for job in plan]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(dict(name=result['job']['name'], status=result['status'],
                frames=result.get('frames')), ensure_ascii=False), flush=True)
    dump(output / 'summary.json', dict(group=group, runs=results, finished_utc=utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--group', choices=['latticegap'], default='latticegap')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        run_one(read(args.worker))
    else:
        campaign(args.output.resolve(), args.group, args.workers)
