"""Disconnected lifting surfaces and explicit, source-independent aircraft inputs.

The aerodynamic aircraft is rigid prescribed geometry, not a six-DOF, fuselage
or propulsion solver. Each panel has its own polar; all surfaces share one joint
circulation solve and, when selected, one interacting free-wake state.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

from .wing import Wing,SectionPolar,owned
from .vortex import _real


@dataclass(frozen=True)
class WingAssembly:
    wings: tuple
    names: tuple

    def __post_init__(self):
        object.__setattr__(self,'wings',tuple(self.wings))
        object.__setattr__(self,'names',tuple(self.names))
        if not self.wings or len(self.names)!=len(self.wings) or any(not isinstance(w,Wing) for w in self.wings):
            raise ValueError('assembly needs named Wing surfaces')
        if any(not isinstance(n,str) or not n.strip() for n in self.names) or len(set(self.names))!=len(self.names):
            raise ValueError('surface names must be nonempty and unique')
        for name in ('leading_edge','trailing_edge','bound','cp','chord_cp','tangent','normal','orthogonal','dl','area'):
            object.__setattr__(self,name,owned(np.concatenate([getattr(w,name) for w in self.wings])))
        cambered = [w for w in self.wings if w.camber_x is not None]
        x = cambered[0].camber_x if cambered else None
        if any(not np.array_equal(w.camber_x, x) for w in cambered):
            raise ValueError('assembly camber sampling grids must match explicitly')
        offsets = None if x is None else owned(np.concatenate([
            np.zeros((w.node_count,len(x),3)) if w.camber_offset_m is None else w.camber_offset_m
            for w in self.wings]))
        object.__setattr__(self, 'camber_x', x)
        object.__setattr__(self, 'camber_offset_m', offsets)
        offset=0;edges=[]
        for wing in self.wings:
            edges.extend((wing.panel_edges+offset).tolist());offset+=wing.node_count
        values=np.array(edges,dtype=np.int64)
        object.__setattr__(self,'panel_edges',np.frombuffer(values.tobytes(),dtype=np.int64).reshape(-1,2))

    @property
    def panels(self):return len(self.area)

    @property
    def node_count(self):return len(self.bound)

    def transformed(self,rotation,translation):
        return WingAssembly(tuple(w.transformed(rotation,translation) for w in self.wings),self.names)


@dataclass(frozen=True)
class PanelPolars:
    polars: tuple

    def __post_init__(self):
        object.__setattr__(self,'polars',tuple(self.polars))
        if not self.polars or any(not isinstance(p,SectionPolar) for p in self.polars):
            raise ValueError('one SectionPolar per panel is required')

    def coefficients(self,alpha_rad):
        alpha=_real(alpha_rad,'panel angles')
        if alpha.shape!=(len(self.polars),):raise ValueError('panel polar count and angles differ')
        values=np.array([p.coefficients(a) for p,a in zip(self.polars,alpha)])
        return tuple(values[:,i] for i in range(3))

    def cl_slope(self,alpha_rad):
        """One declared lift slope per panel, per radian."""
        alpha=_real(alpha_rad,'panel angles')
        if alpha.shape!=(len(self.polars),):raise ValueError('panel polar count and angles differ')
        return np.array([p.cl_slope(a) for p,a in zip(self.polars,alpha)])

    def clipped(self,alpha_rad):
        alpha=_real(alpha_rad,'panel angles')
        if alpha.shape!=(len(self.polars),):raise ValueError('panel polar count and angles differ')
        return np.array([a<np.deg2rad(p.alpha_deg[0]) or a>np.deg2rad(p.alpha_deg[-1]) for p,a in zip(self.polars,alpha)])

    def attached(self,alpha_rad):
        """Per-panel attached-range label (True without a declaration)."""
        alpha=_real(alpha_rad,'panel angles')
        if alpha.shape!=(len(self.polars),):raise ValueError('panel polar count and angles differ')
        return np.array([bool(p.attached(a)) for p,a in zip(self.polars,alpha)])


@dataclass(frozen=True)
class AircraftDefinition:
    name: str
    wing: WingAssembly
    polar: PanelPolars
    source_sha256: str
    raw_json: str


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError(f'duplicate input key: {key}')
        result[key]=value
    return result


def _keys(obj,required,optional=()):
    if not isinstance(obj,dict) or set(required)-obj.keys() or obj.keys()-set(required)-set(optional):
        raise ValueError('missing or unknown aircraft input fields')


def load_aircraft(path):
    """Read a declared aero geometry/polar file without guessing units or frames."""
    raw=Path(path).read_bytes()
    data=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique_object)
    _keys(data,('schema','name','frame','length_unit','angle_unit','polars','surfaces'),('provenance',))
    if data['schema'] not in ('openfawe.aircraft.aero.v1','openfawe.aircraft.aero.v2') or data['frame']!='x_downstream_y_span_z_up' or data['length_unit']!='m' or data['angle_unit']!='deg':
        raise ValueError('unsupported aircraft schema, units or coordinate frame')
    if not isinstance(data['name'],str) or not data['name'].strip():raise ValueError('aircraft name required')
    if not isinstance(data['polars'],dict) or not data['polars']:raise ValueError('named polar tables required')
    polars={}
    for name,table in data['polars'].items():
        _keys(table,('alpha_deg','cl','cd','cm'),('bounds','provenance','periodic','attached_deg'))
        if 'periodic' in table and not isinstance(table['periodic'],bool):
            raise ValueError('polar periodic flag must be a JSON boolean')
        polars[name]=SectionPolar(table['alpha_deg'],table['cl'],table['cd'],table['cm'],name,table.get('bounds','error'),
                                  periodic=table.get('periodic',False),attached_deg=table.get('attached_deg'))
    if not isinstance(data['surfaces'],list) or not data['surfaces']:raise ValueError('surfaces required')
    wings=[];names=[];panel_polars=[]
    for surface in data['surfaces']:
        _keys(surface,('name','leading_edge','trailing_edge','polar_ids'),
              ('camber',) if data['schema']=='openfawe.aircraft.aero.v2' else ())
        camber = surface.get('camber')
        if camber is not None:
            _keys(camber, ('x_over_c','offsets_m'))
        elif 'camber' in surface:
            raise ValueError('camber must be an explicit block or omitted, not null')
        wing=Wing(surface['leading_edge'],surface['trailing_edge'],
                  None if camber is None else camber['x_over_c'],
                  None if camber is None else camber['offsets_m'])
        ids=surface['polar_ids']
        if not isinstance(ids,list) or len(ids)!=wing.panels or any(not isinstance(i,str) or i not in polars for i in ids):
            raise ValueError('each panel needs a known polar ID')
        wings.append(wing);names.append(surface['name']);panel_polars.extend(polars[i] for i in ids)
    return AircraftDefinition(data['name'],WingAssembly(tuple(wings),tuple(names)),PanelPolars(tuple(panel_polars)),
                              hashlib.sha256(raw).hexdigest(),raw.decode('utf-8'))
