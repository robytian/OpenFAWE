"""Compute a short prescribed-wing Euler wake and save it for Python/ParaView."""

import argparse
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np

from .aero.wing import Wing,SectionPolar,SolveSettings,positive,wake_config_metadata
from .aero.wake import WakeConfig,count
from .aero.methods import AeroBackend,AeroMethod
from .aero.lifting_surface import LiftingSurfaceConfig
from .aero.ll_straight import LLStraightConfig
from .aero.aircraft import load_aircraft
from .aero_archive import AeroWriter
from .aero_vtk import export_paraview


def run_demo(output,*,panels=12,dt_s=.05,duration_s=3.,heave_m=.25,free_wake=True,export_vtk=False,
             aero_method='free_wake',aircraft_path=None,core_evolution='constant',eddy_viscosity_factor=1.,
             circulation_solver='fixed_point', chordwise_panels=4, straight_wake_length_m=1000.,
             convection_integrator='euler', geometry_substeps=1):
    panels=count(panels,'panels',2);duration_s=positive(duration_s,'duration')
    heave_m=positive(heave_m,'heave amplitude',zero=True)
    cfg=WakeConfig(dt_s=dt_s,free_wake=free_wake,core_evolution=core_evolution,
                   eddy_viscosity_factor=eddy_viscosity_factor,solve=SolveSettings(solver=circulation_solver),
                   convection_integrator=convection_integrator,geometry_substeps=geometry_substeps)
    backend=AeroBackend(aero_method,wake=cfg,vsm_solve=SolveSettings(relaxation=.03,max_iterations=2000,solver=circulation_solver),
        lifting_surface=LiftingSurfaceConfig(chordwise_panels=chordwise_panels,wake_length_m=straight_wake_length_m),
        ll_straight=LLStraightConfig(wake_length_m=straight_wake_length_m))
    straight=backend.method in (AeroMethod.VSM,AeroMethod.VLM,AeroMethod.LL_STRAIGHT)
    lattice=backend.method in (AeroMethod.VLM,AeroMethod.VLM_FREE_WAKE)
    steps=round(duration_s/cfg.dt_s)
    if steps<1 or steps>cfg.max_rows or not np.isclose(steps*cfg.dt_s,duration_s,rtol=0,atol=1e-12):
        raise ValueError('duration must be a positive integer number of dt, within the row limit')
    definition=load_aircraft(aircraft_path) if aircraft_path is not None else None
    wing=definition.wing if definition else Wing.ellipse(panels)
    polar=definition.polar if definition else SectionPolar.linear()
    panels=wing.panels
    wind=np.array([5.,0.,.5]);base=np.array([3.5,0.,5.]);frequency=.25
    def prescribed(time):
        phase=2*np.pi*frequency*time
        pose=wing.transformed(np.eye(3),base+[0,0,heave_m*np.sin(phase)])
        velocity=np.array([0.,0.,heave_m*2*np.pi*frequency*np.cos(phase)])
        return pose,velocity
    first,velocity=prescribed(0.)
    polar_metadata={'name':'per-panel tables','bounds':'per-panel'} if definition else {'name':polar.name,'bounds':polar.bounds,'alpha_deg':polar.alpha_deg.tolist(),'cl':polar.cl.tolist(),'cd':polar.cd.tolist(),'cm':polar.cm.tolist()}
    case={'name':definition.name if definition else 'elliptic-prescribed-heave',
          'geometry':'explicit disconnected lifting surfaces from aircraft input' if definition else 'analytic elliptic wing, 5 m span / 1 m root chord; not DE2019',
          'panels':panels,'inflow_m_s':wind.tolist(),'base_position_m':base.tolist(),
          'heave_amplitude_m':heave_m,'heave_frequency_hz':frequency,'duration_s':duration_s,
          'aero_method':backend.method.value,'aerodynamics':backend.metadata(),
          'time_integrator':f'{convection_integrator} frozen-strength geometry; {geometry_substeps} substeps' if not straight else 'instantaneous straight-wake solve at each prescribed motion sample',
          'wake':wake_config_metadata(cfg) if not straight else None,
          'polar':polar_metadata,
          'initialization':'bound-only impulsive start, no prefilled wake' if not straight else 'instantaneous prescribed straight wake, no released history',
          'vsm_display_wake_length_m':20. if backend.method==AeroMethod.VSM else None,
          'ground_schematic':'Viewer only; no line force, winch state or generation calculated',
          'source_contract':'docs/design/of-0006-wing-wake-contract.md'}
    if lattice:
        case.update(chordwise_panels=chordwise_panels,polar_usage='not used by inviscid lifting-surface closure',
            source_contract='docs/tasks/OF-0006-10/algorithm.md')
    if backend.method==AeroMethod.LL_STRAIGHT:
        case['source_contract']='docs/tasks/OF-0006-11/contract.md'
    if definition:
        case.update(aircraft_input_sha256=definition.source_sha256,aircraft_input_json=definition.raw_json,
                    surface_names=list(wing.names),panel_polar_bounds_deg=[[p.alpha_deg[0],p.alpha_deg[-1]] for p in polar.polars])
    # Validate/init solve before creating the owned output directory.
    state=backend.start(first,polar,wind,structural_velocity=velocity)
    writer=AeroWriter(output,case)
    try:
        writer.append(state)
        for step in range(1,steps+1):
            target,velocity=prescribed(step*cfg.dt_s)
            state=backend.advance(state,polar,wind,next_wing=target,structural_velocity=velocity,next_time_s=step*cfg.dt_s)
            writer.append(state)
        writer.finish()
    except BaseException as error:
        writer.fail(error);raise
    if export_vtk:
        export_paraview(output,Path(output)/'paraview')
    return {'output':str(Path(output).resolve()),'frames':steps+1,'final_time_s':state.time_s,
            'final_force_N':state.solution.force.tolist(),'wake_rows':len(state.free_rows),
            'aero_method':backend.method.value}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--panels',type=int,default=12);parser.add_argument('--dt',type=float,default=.05)
    parser.add_argument('--duration',type=float,default=3.);parser.add_argument('--heave',type=float,default=.25)
    parser.add_argument('--frozen-wake',action='store_true');parser.add_argument('--paraview',action='store_true')
    parser.add_argument('--aero-method',choices=[m.value for m in AeroMethod],default='free_wake',
                        help='Aerodynamic load backend; default preserves the existing free-wake solver')
    parser.add_argument('--aircraft',type=Path,help='Versioned geometry and per-panel polar JSON; otherwise use the analytic ellipse')
    parser.add_argument('--core-evolution',choices=['constant','age','age_stretch'],default='constant')
    parser.add_argument('--eddy-viscosity-factor',type=float,default=1.)
    parser.add_argument('--circulation-solver',choices=['fixed_point','newton','newton_picard'],default='fixed_point')
    parser.add_argument('--chordwise-panels',type=int,default=4)
    parser.add_argument('--straight-wake-length',type=float,default=1000.)
    parser.add_argument('--convection-integrator',choices=['euler','heun','rk4'],default='euler')
    parser.add_argument('--geometry-substeps',type=int,default=1)
    args=parser.parse_args(argv)
    print(json.dumps(run_demo(args.output,panels=args.panels,dt_s=args.dt,duration_s=args.duration,
                              heave_m=args.heave,free_wake=not args.frozen_wake,export_vtk=args.paraview,
                              aero_method=args.aero_method,aircraft_path=args.aircraft,
                              core_evolution=args.core_evolution,eddy_viscosity_factor=args.eddy_viscosity_factor,
                              circulation_solver=args.circulation_solver,chordwise_panels=args.chordwise_panels,
                              straight_wake_length_m=args.straight_wake_length,convection_integrator=args.convection_integrator,
                              geometry_substeps=args.geometry_substeps),indent=2))


if __name__=='__main__': main()
