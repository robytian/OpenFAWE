"""Explicit reference-case transformations and aerodynamic coefficient reporting.

These helpers do not alter either aerodynamic solver or interpret an integrated
wing curve as full-aircraft dynamics. Angles are degrees, lengths metres.
"""
from dataclasses import replace
import json
import operator
import numpy as np
from .aircraft import AircraftDefinition, PanelPolars, WingAssembly
from .wing import Wing, SectionPolar, positive
from .vortex import _real


def inflow(speed, alpha_deg, beta_deg=0.):
    speed=positive(speed,'speed')
    alpha,beta=(_real(x,n) for x,n in [(alpha_deg,'alpha'),(beta_deg,'beta')])
    if alpha.shape or beta.shape or abs(alpha)>=90 or abs(beta)>=90:
        raise ValueError('reference inflow requires scalar angles strictly inside +/-90 degrees')
    a,b=np.deg2rad([alpha,beta])
    return speed*np.array([np.cos(a)*np.cos(b),np.sin(b),np.sin(a)*np.cos(b)])


def load_coefficients(solution, wing, velocity, area, chord, reference=(0.,0.,0.),
                      density=1.225, body_y=(0.,1.,0.), moment_frame='body'):
    """Wind-axis forces and explicitly body- or wind-axis moments / qSc.

    Wind moments use the same drag/side/lift basis as force coefficients.
    Matching a particular experiment still requires its exact origin and axes.
    """
    if moment_frame not in ('body', 'wind'):
        raise ValueError('moment_frame must be body or wind')
    v=_real(velocity,'velocity');y=_real(body_y,'body y')
    if v.shape!=(3,) or y.shape!=(3,):raise ValueError('velocity and body_y need three components')
    speed=positive(np.linalg.norm(v),'speed');drag=v/speed
    lift=np.cross(drag,y);length=positive(np.linalg.norm(lift),'lift axis length');lift/=length
    side=np.cross(lift,drag)
    qS=.5*positive(density,'density')*speed**2*positive(area,'area')
    moment=solution.moment(wing,reference=reference)/(qS*positive(chord,'chord'))
    if moment_frame == 'wind':
        moment=np.array([moment@drag,moment@side,moment@lift])
    return dict(CL=float(solution.force@lift/qS),CD=float(solution.force@drag/qS),
                CS=float(solution.force@side/qS),CMx=float(moment[0]),CMy=float(moment[1]),CMz=float(moment[2]))


def refine_aircraft(definition, subdivisions):
    """Refine each original segment, preserving source stations and polar domain.

V3 raw section tables are interpolated at subsegment midpoints. Generic source
panel-ID data, including M600, remain constant within each original segment.
Explicit mean-surface offsets are linearly interpolated between original nodes
in the declared common section frame. Original nodes are copied exactly.
"""
    if isinstance(subdivisions,(bool,np.bool_)):raise ValueError('subdivisions must be a positive integer')
    try:n=operator.index(subdivisions)
    except TypeError as error:raise ValueError('subdivisions must be a positive integer') from error
    if n<1:raise ValueError('subdivisions must be positive')
    if not isinstance(definition,AircraftDefinition):raise ValueError('aircraft definition required')
    if n==1:return definition
    raw=json.loads(definition.raw_json);prov=raw.get('provenance',{})
    section_data=prov.get('source_sections');tables=prov.get('section_polars')
    wings=[];polars=[];offset=0
    for k,wing in enumerate(definition.wing.wings):
        if section_data is not None and len(section_data)!=wing.panels+1:
            raise ValueError('refine from the original source definition, not an already refined view')
        le=[];te=[];camber=[]
        for i in range(wing.panels):
            for j in range(n):
                t=j/n;le.append((1-t)*wing.leading_edge[i]+t*wing.leading_edge[i+1]);te.append((1-t)*wing.trailing_edge[i]+t*wing.trailing_edge[i+1])
                if wing.camber_x is not None:
                    camber.append(wing.camber_offset_m[i].copy() if j==0 else
                        (1-t)*wing.camber_offset_m[i]+t*wing.camber_offset_m[i+1])
                parent=definition.polar.polars[offset+i]
                if section_data is not None and tables is not None and len(definition.wing.wings)==1:
                    a,b=[np.array(tables[str(int(row[0]))]) for row in section_data[i:i+2]]
                    if not np.array_equal(a[:,0],b[:,0]):raise ValueError('section alpha grids differ')
                    tmid=(j+.5)/n;values=(1-tmid)*a+tmid*b
                    # Preserve the admissible view, never revive rejected raw rows.
                    keep=(values[:,0]>=parent.alpha_deg[0])&(values[:,0]<=parent.alpha_deg[-1]);values=values[keep]
                    polars.append(SectionPolar(*values.T,name=parent.name+f'/sub-{j}',bounds=parent.bounds))
                else:polars.append(parent)
        le.append(wing.leading_edge[-1]);te.append(wing.trailing_edge[-1])
        if wing.camber_x is not None:camber.append(wing.camber_offset_m[-1].copy())
        wings.append(Wing(le,te,wing.camber_x,camber) if wing.camber_x is not None else Wing(le,te))
        offset+=wing.panels
    return replace(definition,wing=WingAssembly(tuple(wings),definition.wing.names),polar=PanelPolars(tuple(polars)))


def source_curve(reference_data, alpha_deg):
    """Evaluate the original DE2019 integrated wing table without extrapolation."""
    table=reference_data['parameters']['initAircraft'];x=_real(table['alpha'],'source alpha')
    a=_real(alpha_deg,'requested alpha')
    if x.ndim!=1 or len(x)<2 or np.any(np.diff(x)<=0):raise ValueError('invalid source angle grid')
    if np.any(a<x[0]) or np.any(a>x[-1]):raise ValueError('outside original integrated-wing curve')
    return {label:np.interp(a,x,_real(table[key],key)) for label,key in
            [('CL','wing_cL_Static'),('CD','wing_cD_Static'),('CM','wing_cM_Static')]}
