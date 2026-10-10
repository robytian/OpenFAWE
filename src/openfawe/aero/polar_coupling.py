# SPDX-License-Identifier: Apache-2.0
"""Optional grid-consistent, quasi-steady sectional closure for a vortex lattice.

Virtual boundary normals change circulation; physical geometry and unsteady
loads are retained. This is not a boundary-layer or dynamic-stall solver.
See docs/tasks/OF-0006-polarcoupled20260924/contract.md for equations and limits.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from .vortex import _norm, _real
from .wing import SectionPolar, owned


class PolarCouplingError(ValueError):
    """A rejected closure, never an implicit return to inviscid loads."""
    def __init__(self, message, trace=()):
        self.trace = tuple(trace)
        super().__init__(message)


def section_frames(wing):
    """Geometric mid-strip frames; independent of legacy full-cosine CPs."""
    le, te = np.asarray(wing.leading_edge), np.asarray(wing.trailing_edge)
    a, b = np.asarray(wing.panel_edges).T
    chord = .5*((te[a]+te[b])-(le[a]+le[b]))
    lengths = _norm(chord)
    tangent = chord/lengths[:, None]
    bound = le+.25*(te-le)
    normal = np.cross(chord, bound[b]-bound[a])
    normal /= _norm(normal)[:, None]
    span = np.cross(normal, tangent)
    middle = .5*(le+te)
    area = _norm(np.cross(chord, middle[b]-middle[a]))
    origin = .5*(le[a]+le[b])
    return origin, lengths, tangent, normal, span, area


def polar_tables(polar, count):
    tables = (polar,)*count if isinstance(polar, SectionPolar) else getattr(polar, 'polars', ())
    if len(tables) != count or any(not isinstance(p, SectionPolar) for p in tables):
        raise ValueError('polar coupling requires one SectionPolar per strip')
    return tuple(tables)


def encode_polars(tables):
    """Deduplicate exact source tables without modifying their coefficients."""
    unique, indices, lookup = [], [], {}
    for table in tables:
        data = {key:getattr(table, key).tolist() for key in ('alpha_deg','cl','cd','cm')}
        data.update(name=table.name, bounds=table.bounds)
        identity = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if identity not in lookup:
            lookup[identity] = len(unique); unique.append(data)
        indices.append(lookup[identity])
    return json.dumps(dict(tables=unique, panel_ids=indices), sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def decode_polars(text, count):
    data = json.loads(text)
    tables = tuple(SectionPolar(**table) for table in data['tables'])
    indices = data['panel_ids']
    if (len(indices) != count or any(type(i) is not int or not 0 <= i < len(tables)
                                    for i in indices)):
        raise ValueError('invalid archived source-polar indices')
    return tuple(tables[i] for i in indices)


def polar_identity(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class SectionReferences:
    """Infinite-span geometric reference on the exact selected chord grid."""
    def __init__(self, wing, mesh):
        self.count = wing.panels
        self.nc = len(mesh.vortex_rows)-1
        (self.origin, self.chord, self.tangent, self.normal,
         self.span, self.area) = section_frames(wing)
        self.quarter = self.origin+.25*self.chord[:, None]*self.tangent
        a, b = np.asarray(wing.panel_edges).T
        vortex = .5*(mesh.vortex_rows[:-1, a]+mesh.vortex_rows[:-1, b])
        points = mesh.collocation_points.reshape(self.nc, self.count, 3)
        basis = np.stack((self.tangent, self.normal), axis=-1)
        self.vortex = np.einsum('kni,nij->nkj', vortex-self.origin, basis)/self.chord[:,None,None]
        cp = np.einsum('kni,nij->nkj', points-self.origin, basis)/self.chord[:,None,None]
        delta = cp[:,:,None,:]-self.vortex[:,None,:,:]
        radius2 = np.sum(delta*delta, axis=-1)
        if np.any(radius2 <= 0):
            raise ValueError('coincident two-dimensional vortex and collocation point')
        self.kernel = np.stack((delta[:,:,:,1], -delta[:,:,:,0]), axis=-1)/(2*np.pi*radius2[:,:,:,None])
        self.original_normals = np.asarray(mesh.normals)
        axis = self.span[mesh.strip_index]
        self.parallel = np.sum(mesh.normals*axis, axis=1)[:,None]*axis
        self.perpendicular = mesh.normals-self.parallel
        self.cross = np.cross(axis, self.perpendicular)
        self.strip_index = mesh.strip_index
        self.basis = basis

    def normals(self, delta):
        delta = _real(delta, 'virtual incidence')
        if delta.shape != (self.count,):
            raise ValueError('one virtual incidence per strip required')
        angle = delta[self.strip_index, None]
        normals = (self.parallel + np.cos(angle)*self.perpendicular
                   - np.sin(angle)*self.cross)
        if not np.any(delta):
            normals = self.original_normals.copy()
        derivative = -np.sin(angle)*self.perpendicular-np.cos(angle)*self.cross
        return normals, derivative

    def response(self, delta, *, derivative=False):
        normals, dn = self.normals(delta)
        project = lambda a: np.einsum('kni,nij->nkj', a.reshape(self.nc,self.count,3), self.basis)
        n2 = project(normals)
        matrix = np.einsum('nkli,nki->nkl', self.kernel, n2)
        try:
            strengths = np.linalg.solve(matrix, -n2)
            change = None
            if derivative:
                dn2 = project(dn)
                dm = np.einsum('nkli,nki->nkl', self.kernel, dn2)
                change = np.linalg.solve(matrix, -dn2-np.einsum('nkl,nli->nki', dm, strengths))
        except np.linalg.LinAlgError as error:
            raise ValueError('singular geometric section reference') from error
        return strengths, change

    def moment(self, strengths, alpha):
        direction = np.column_stack((np.cos(alpha), np.sin(alpha)))
        gamma = np.einsum('nki,ni->nk', strengths, direction)
        arm = ((self.vortex[:,:,0]-.25)*direction[:,None,0]
               + self.vortex[:,:,1]*direction[:,None,1])
        return -2*np.sum(gamma*arm, axis=1)


def source_values(tables, alpha, physical_alpha):
    values, slopes = [], []
    for table, effective, physical in zip(tables, alpha, physical_alpha):
        lower, upper = np.deg2rad(table.alpha_deg[[0,-1]])
        # A source compatibility clamp is not permission to use unknown data.
        if not lower <= effective <= upper or not lower <= physical <= upper:
            raise ValueError('effective or prescribed incidence outside source polar table')
        index = int(np.clip(np.searchsorted(table.alpha_deg, np.rad2deg(effective), side='right')-1,
                            0, len(table.alpha_deg)-2))
        slope = (table.cl[index+1]-table.cl[index])/np.deg2rad(table.alpha_deg[index+1]-table.alpha_deg[index])
        if slope <= 0:
            raise ValueError('source-polar coupling requires a positive active lift slope')
        values.append(table.coefficients(effective)); slopes.append(slope)
    return np.asarray(values), np.asarray(slopes)


def inverse_reference(coefficients, cl, physical_alpha):
    a, b = coefficients.T
    radius = np.hypot(a, b)
    if np.any(radius <= 0) or np.any(np.abs(cl) >= radius):
        raise ValueError('section circulation is outside the positive-slope reference branch')
    phase = np.arctan2(a, b)
    if np.any(np.cos(physical_alpha+phase) <= 0):
        raise ValueError('prescribed flow is outside the positive-slope reference branch')
    alpha = np.arcsin(cl/radius)-phase
    alpha += 2*np.pi*np.rint((physical_alpha-alpha)/(2*np.pi))
    distance = np.arctan2(np.sin(alpha-physical_alpha), np.cos(alpha-physical_alpha))
    if np.any(np.abs(distance) >= np.pi/2):
        raise ValueError('ambiguous section incidence branch')
    slope = -a*np.sin(alpha)+b*np.cos(alpha)
    if np.any(slope <= 0):
        raise ValueError('nonpositive geometric reference slope')
    return alpha, slope


@dataclass(frozen=True)
class PolarCouplingResult:
    gamma: np.ndarray
    virtual_normals: np.ndarray
    delta_rad: np.ndarray
    effective_alpha_rad: np.ndarray
    physical_alpha_rad: np.ndarray
    circulation_cl: np.ndarray
    source_coefficients: np.ndarray
    potential_cm: np.ndarray
    cl_residual: np.ndarray
    speed_m_s: np.ndarray
    chord_m: np.ndarray
    area_m2: np.ndarray
    reference_points_m: np.ndarray
    ambient_m_s: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    span: np.ndarray
    source_polars_json: str
    trace_json: str
    iterations: int
    geometric_residual_m_s: np.ndarray | None = None
    potential_force_N: np.ndarray | None = None
    potential_couple_N_m: np.ndarray | None = None
    correction_force_N: np.ndarray | None = None
    correction_couple_N_m: np.ndarray | None = None

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                object.__setattr__(self, name, owned(value))

    def corrections(self, density):
        qarea = .5*density*self.speed_m_s**2*self.area_m2
        direction = (np.cos(self.effective_alpha_rad)[:,None]*self.tangent
                     + np.sin(self.effective_alpha_rad)[:,None]*self.normal)
        force = (qarea*self.source_coefficients[:,1])[:,None]*direction
        couple = (qarea*self.chord_m*(self.source_coefficients[:,2]-self.potential_cm))[:,None]*self.span
        return force, couple


class ClosureProblem:
    """A fixed geometry/wake nonlinear solve; rejected trials leave no state."""
    def __init__(self, wing, mesh, influence, ambient_known, strip_ambient, polar, settings):
        self.reference = SectionReferences(wing, mesh)
        self.influence, self.ambient_known = influence, ambient_known
        self.ambient = np.asarray(strip_ambient)
        self.tables = polar_tables(polar, wing.panels)
        self.settings = settings
        self.strip_index = mesh.strip_index
        transverse = self.ambient-np.sum(self.ambient*self.reference.span, axis=1)[:,None]*self.reference.span
        self.speed = _norm(transverse)
        forward = np.sum(transverse*self.reference.tangent, axis=1)
        if np.any(self.speed <= 0) or np.any(forward <= 0):
            raise ValueError('polar coupling requires nonzero forward strip-relative flow')
        self.physical_alpha = np.arctan2(np.sum(transverse*self.reference.normal,axis=1), forward)
        self.scale = 2/(self.speed*self.reference.chord)

    def evaluate(self, delta, *, jacobian=False):
        ref = self.reference
        normals, dn = ref.normals(delta)
        matrix = np.einsum('ijk,ik->ij', self.influence, normals)
        rhs = -np.sum(self.ambient_known*normals, axis=1)
        try:
            gamma = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as error:
            raise ValueError('singular coupled lattice boundary matrix') from error
        cl = self.scale*gamma[-ref.count:]
        strengths, derivative = ref.response(delta, derivative=jacobian)
        coefficients = 2*strengths.sum(axis=1)
        alpha, reference_slope = inverse_reference(coefficients, cl, self.physical_alpha)
        source, source_slope = source_values(self.tables, alpha, self.physical_alpha)
        residual = cl-source[:,0]
        data = dict(gamma=gamma, normals=normals, matrix=matrix, cl=cl, alpha=alpha,
                    source=source, residual=residual, potential_cm=ref.moment(strengths,alpha))
        if jacobian:
            total = self.ambient_known+np.einsum('ijk,j->ik', self.influence, gamma)
            forcing = np.zeros((len(gamma), ref.count))
            forcing[np.arange(len(gamma)), self.strip_index] = -np.sum(total*dn, axis=1)
            dgamma = np.linalg.solve(matrix, forcing)
            dcl = self.scale[:,None]*dgamma[-ref.count:]
            dcoeff = 2*derivative.sum(axis=1)
            reference_delta = dcoeff[:,0]*np.cos(alpha)+dcoeff[:,1]*np.sin(alpha)
            factor = source_slope/reference_slope
            data['jacobian'] = (1-factor[:,None])*dcl+np.diag(factor*reference_delta)
        return data

    def solve(self, initial=None):
        cfg = self.settings; ref = self.reference
        bound = np.deg2rad(cfg.polar_max_virtual_incidence_deg)
        delta = np.zeros(ref.count) if initial is None else np.clip(_real(initial,'initial virtual incidence'),-bound,bound)
        trace = []
        try:
            data = self.evaluate(delta, jacobian=True)
        except ValueError as error:
            raise PolarCouplingError('invalid initial polar-coupled state: '+str(error)) from error
        for iteration in range(cfg.polar_max_iterations+1):
            error = float(np.max(np.abs(data['residual'])))
            trace.append(dict(iteration=iteration, maximum_cl_residual=error,
                              maximum_virtual_incidence_deg=float(np.rad2deg(np.max(abs(delta))))))
            if iteration == 0:
                trace[-1]['initial_virtual_incidence_rad'] = delta.tolist()
            if error <= cfg.polar_cl_tolerance:
                condition = float(np.linalg.cond(data['matrix']))
                if (not np.isfinite(condition) or (cfg.max_condition_number is not None
                                                  and condition > cfg.max_condition_number)):
                    raise PolarCouplingError('coupled lattice condition exceeds declared limit', trace)
                result = PolarCouplingResult(data['gamma'], data['normals'], delta, data['alpha'],
                    self.physical_alpha, data['cl'], data['source'], data['potential_cm'],
                    data['residual'], self.speed, ref.chord, ref.area, ref.quarter, self.ambient,
                    ref.tangent, ref.normal, ref.span, encode_polars(self.tables),
                    json.dumps(trace,sort_keys=True,allow_nan=False), iteration)
                return result, condition
            if iteration == cfg.polar_max_iterations:
                break
            try:
                step = np.linalg.solve(data['jacobian'], -data['residual'])
            except np.linalg.LinAlgError as failure:
                raise PolarCouplingError('singular polar-coupling Newton Jacobian', trace) from failure
            accepted = None; rejected = []
            for reduction in range(21):
                trial = np.clip(delta+(.5**reduction)*step, -bound, bound)
                try:
                    candidate = self.evaluate(trial)
                    norm = float(np.max(np.abs(candidate['residual'])))
                    if np.isfinite(norm) and norm < error:
                        accepted = trial; break
                    rejected.append('residual_not_reduced')
                except ValueError as failure:
                    rejected.append(str(failure))
            trace[-1].update(backtracks=len(rejected), rejected_trials=rejected)
            if accepted is None:
                raise PolarCouplingError(f'polar-coupling line search failed; max Cl residual {error:.6g}', trace)
            delta = accepted
            data = self.evaluate(delta, jacobian=True)
        raise PolarCouplingError(f'polar coupling did not converge; max Cl residual {error:.6g}', trace)


def archive_fields(result):
    """Self-contained numeric fields and UTF-8 tables for independent replay."""
    attributes = {'polar_coupling_model':'grid_section',
                  'polar_source_sha256':polar_identity(result.source_polars_json),
                  'polar_iterations':result.iterations}
    arrays = {'polar_'+name:value for name,value in vars(result).items()
              if isinstance(value, np.ndarray)}
    for name in ('source_polars_json','trace_json'):
        arrays['polar_'+name] = np.frombuffer(getattr(result,name).encode('utf-8'),dtype=np.uint8)
    return attributes, arrays


def validate_archive(frame, wing, mesh, settings, density):
    """Recompute closure and corrections, not just stored residual subtraction."""
    ref = SectionReferences(wing, mesh)
    def check(name, expected):
        saved = frame.get('polar_'+name)
        if (saved is None or np.shape(saved) != np.shape(expected) or
                not np.allclose(saved, expected, rtol=2e-12, atol=2e-12)):
            raise ValueError('archived polar coupling mismatch: '+name)
    text = bytes(frame['polar_source_polars_json']).decode('utf-8')
    if frame.get('polar_source_sha256') != polar_identity(text):
        raise ValueError('archived source polar identity differs')
    tables = decode_polars(text, wing.panels)
    delta = frame['polar_delta_rad']
    if delta.shape != (wing.panels,) or np.any(abs(delta)>np.deg2rad(settings.polar_max_virtual_incidence_deg)):
        raise ValueError('archived virtual incidence exceeds declared limit')
    normals,_ = ref.normals(delta)
    check('virtual_normals', normals)
    for name, value in dict(chord_m=ref.chord,area_m2=ref.area,
            reference_points_m=ref.quarter,tangent=ref.tangent,normal=ref.normal,span=ref.span).items():
        check(name,value)
    ambient = frame['polar_ambient_m_s']
    if ambient.shape != (wing.panels,3):
        raise ValueError('invalid archived coupling ambient shape')
    u = ambient-np.sum(ambient*ref.span,axis=1)[:,None]*ref.span
    speed = _norm(u); forward = np.sum(u*ref.tangent,axis=1)
    if np.any(speed<=0) or np.any(forward<=0):
        raise ValueError('invalid archived coupling forward speed')
    physical = np.arctan2(np.sum(u*ref.normal,axis=1),forward)
    check('speed_m_s',speed); check('physical_alpha_rad',physical)
    gamma = frame['lattice_circulation_m2_s']
    check('gamma',gamma)
    cl = 2*gamma[-wing.panels:]/(speed*ref.chord)
    strengths,_ = ref.response(delta)
    alpha,_ = inverse_reference(2*strengths.sum(axis=1),cl,physical)
    source,_ = source_values(tables,alpha,physical)
    cm = ref.moment(strengths,alpha)
    residual = cl-source[:,0]
    for name,value in dict(circulation_cl=cl,effective_alpha_rad=alpha,
            source_coefficients=source,potential_cm=cm,cl_residual=residual).items():
        check(name,value)
    if np.max(abs(residual))>settings.polar_cl_tolerance:
        raise ValueError('archived source-polar closure did not converge')
    velocity = frame['lattice_relative_velocity_m_s']
    check('geometric_residual_m_s',np.sum(velocity*mesh.normals,axis=1))
    if not np.allclose(np.sum(velocity*normals,axis=1),frame['lattice_boundary_residual_m_s'],rtol=0,atol=2e-12):
        raise ValueError('archived virtual boundary residual differs from velocity')
    qarea = .5*density*speed**2*ref.area
    force = (qarea*source[:,1])[:,None]*(np.cos(alpha)[:,None]*ref.tangent+np.sin(alpha)[:,None]*ref.normal)
    couple = (qarea*ref.chord*(source[:,2]-cm))[:,None]*ref.span
    weights = mesh.area.reshape(ref.nc,ref.count)
    weights = weights/weights.sum(axis=0)
    df = weights[:,:,None]*force
    dc = weights[:,:,None]*couple+np.cross(ref.quarter-mesh.collocation_points.reshape(ref.nc,ref.count,3),df)
    check('correction_force_N',df); check('correction_couple_N_m',dc)
    for actual,base,addition in [('lattice_panel_force_N','potential_force_N',df),
                                 ('lattice_panel_couple_N_m','potential_couple_N_m',dc)]:
        if np.shape(frame['polar_'+base]) != addition.shape or not np.allclose(
                frame[actual],(frame['polar_'+base]+addition).reshape(-1,3),rtol=2e-12,atol=2e-12):
            raise ValueError('archived polar load correction routing differs: '+actual)
    trace = json.loads(bytes(frame['polar_trace_json']).decode('utf-8'))
    iterations = frame['polar_iterations']
    if (not isinstance(iterations,(int,np.integer)) or not 0<=iterations<=settings.polar_max_iterations
            or len(trace)!=iterations+1 or frame['iterations']!=iterations+1
            or not np.isclose(trace[-1]['maximum_cl_residual'],np.max(abs(residual)),rtol=0,atol=2e-12)):
        raise ValueError('archived polar iteration trace inconsistent')
