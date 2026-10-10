# SPDX-License-Identifier: Apache-2.0
# Transcribed from google/makani 1d66923 analysis/aero/airfoil.py (Airfoil.__init__,
# _GetInterpolation, _AddPostStallData, GetCoeffs). See third_party/MAKANI-NOTICE.md.
"""Full-circle section polars for the stage-4 wing closure (decision 0020).

Makani's hover model reads XFOIL tables ``{alphas, flap_deflections, lift/drag/
moment_coeffs}`` (radians, rows = flap deflections) and, when stall angles are
given, blends them over 0.1 rad into a NACA 0012 full-circle table on a fixed
400-point grid over [-pi, pi]. Its lookups are scipy ``RectBivariateSpline(kx=1,
ky=1)`` evaluations, i.e. bilinear interpolation whose FITPACK evaluator clamps
arguments to the knot range (single-row tables use ``numpy.interp``, which also
clamps). This module reproduces those operations with NumPy only; the unit tests
compare against scipy when it is installed.

Everything here is data preparation. The resulting ``SectionPolar`` is periodic
over +-180 deg and carries the declared attached range used for regime labels.
The Viterna-Corrigan extension is an alternative post-stall option kept for
sensitivity; its reverse-flow half is a symmetry assumption named in the code.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

from .vortex import _real
from .wing import SectionPolar, owned

#: Grid of the source blend: numpy.linspace(-pi, pi, 400) in _AddPostStallData.
BLEND_POINTS = 400
#: Linear blend width [rad] past each stall angle in _AddPostStallData.
BLEND_WIDTH_RAD = 0.1


@dataclass(frozen=True)
class MakaniAirfoil:
    """One source table: angles in radians, coefficient rows per flap deflection."""
    name: str
    alpha_rad: np.ndarray
    flap_rad: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm: np.ndarray
    source: str = ''
    source_sha256: str = ''
    stall_rad: tuple | None = None

    def __post_init__(self):
        for key in ('alpha_rad', 'flap_rad', 'cl', 'cd', 'cm'):
            object.__setattr__(self, key, owned(getattr(self, key)))
        if self.alpha_rad.ndim != 1 or len(self.alpha_rad) < 2 or np.any(np.diff(self.alpha_rad) <= 0):
            raise ValueError('airfoil alphas must increase strictly')
        if self.flap_rad.ndim != 1 or not len(self.flap_rad) or np.any(np.diff(self.flap_rad) <= 0):
            raise ValueError('airfoil flap deflections must increase strictly')
        shape = (len(self.flap_rad), len(self.alpha_rad))
        if any(getattr(self, key).shape != shape for key in ('cl', 'cd', 'cm')):
            raise ValueError('airfoil coefficient tables must be (flaps, alphas)')

    def lookup(self, alpha_rad, delta_rad=0.):
        """Clamped bilinear evaluation, the source ``_GetInterpolation`` result."""
        alpha = _real(alpha_rad, 'angle of attack')
        delta = np.broadcast_to(_real(delta_rad, 'flap deflection'), alpha.shape)
        return tuple(_bilinear_clamped(self.flap_rad, self.alpha_rad, table, delta, alpha)
                     for table in (self.cl, self.cd, self.cm))

    def coefficients(self, alpha_rad, delta_rad=0.):
        """Source ``GetCoeffs``: wrap the angle to [-pi, pi), saturate the flap."""
        alpha = _real(alpha_rad, 'angle of attack')
        wrapped = np.mod(alpha+np.pi, 2.*np.pi)-np.pi
        delta = np.clip(_real(delta_rad, 'flap deflection'), self.flap_rad[0], self.flap_rad[-1])
        return self.lookup(wrapped, delta)


def _bilinear_clamped(rows, columns, table, row_values, column_values):
    """Bilinear interpolation on a rectangular grid with arguments clamped to it.

    ``rows``/``columns`` are the strictly increasing knots of axes 0/1 of
    ``table``. A single row degenerates to linear interpolation along the
    columns (numpy.interp in the source).
    """
    x = np.clip(np.asarray(column_values, dtype=float), columns[0], columns[-1])
    if len(rows) == 1:
        return np.interp(x, columns, table[0])
    y = np.clip(np.asarray(row_values, dtype=float), rows[0], rows[-1])
    j = np.clip(np.searchsorted(columns, x, side='right')-1, 0, len(columns)-2)
    i = np.clip(np.searchsorted(rows, y, side='right')-1, 0, len(rows)-2)
    tx = (x-columns[j])/(columns[j+1]-columns[j])
    ty = (y-rows[i])/(rows[i+1]-rows[i])
    return ((1.-ty)*((1.-tx)*table[i, j]+tx*table[i, j+1])
            + ty*((1.-tx)*table[i+1, j]+tx*table[i+1, j+1]))


def load_makani_airfoil(path):
    """Read a Makani airfoil JSON (radians) without altering any value."""
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8'))
    for key in ('name', 'alphas', 'flap_deflections', 'lift_coeffs', 'drag_coeffs', 'moment_coeffs'):
        if key not in data:
            raise ValueError(f'Makani airfoil file lacks {key}')
    return MakaniAirfoil(str(data['name']), np.asarray(data['alphas'], dtype=float),
                         np.asarray(data['flap_deflections'], dtype=float),
                         np.asarray(data['lift_coeffs'], dtype=float), np.asarray(data['drag_coeffs'], dtype=float),
                         np.asarray(data['moment_coeffs'], dtype=float), source=str(path),
                         source_sha256=hashlib.sha256(raw).hexdigest())


def post_stall_blend(airfoil, stall_deg, high_aoa):
    """Source ``_AddPostStallData``: blend into the full-circle table past stall.

    Returns a new table on the 400-point grid over [-pi, pi] with the airfoil's
    own flap rows. Weights are 1 inside [stall_low, stall_high], fall linearly
    to 0 over 0.1 rad outside, so the result equals the (clamped) source table
    inside the stall angles and the high-angle table beyond the blend band.
    """
    low, high = (float(v) for v in _real(stall_deg, 'stall angles'))
    if not high > low:
        raise ValueError('stall angles must be (low, high) with high > low')
    stall = (np.deg2rad(low), np.deg2rad(high))
    alphas = np.linspace(-np.pi, np.pi, BLEND_POINTS)
    past = np.maximum(np.maximum(stall[0]-alphas, alphas-stall[1]), 0.)
    weights = 1.-np.minimum(past/BLEND_WIDTH_RAD, 1.)
    blended = []
    for table, high_table in ((airfoil.cl, high_aoa.cl), (airfoil.cd, high_aoa.cd), (airfoil.cm, high_aoa.cm)):
        rows = np.empty((len(airfoil.flap_rad), len(alphas)))
        for i, flap in enumerate(airfoil.flap_rad):
            flap_column = np.full(alphas.shape, flap)
            low_values = _bilinear_clamped(airfoil.flap_rad, airfoil.alpha_rad, table, flap_column, alphas)
            high_values = _bilinear_clamped(high_aoa.flap_rad, high_aoa.alpha_rad, high_table, flap_column, alphas)
            rows[i] = weights*low_values+(1.-weights)*high_values
        blended.append(rows)
    return MakaniAirfoil(airfoil.name, alphas, airfoil.flap_rad, *blended, source=airfoil.source,
                         source_sha256=airfoil.source_sha256, stall_rad=stall)


def section_polar(airfoil, delta_rad=0., *, name=None, attached_deg=None):
    """A ``SectionPolar`` at one flap deflection, periodic when the table is full-circle.

    Values are the table rows interpolated in the flap direction only (the alpha
    knots are kept), so lookups at the knots reproduce the source exactly.
    """
    delta = float(np.clip(_real(delta_rad, 'flap deflection'), airfoil.flap_rad[0], airfoil.flap_rad[-1]))
    cl, cd, cm = airfoil.lookup(airfoil.alpha_rad, delta)
    alpha_deg = np.rad2deg(airfoil.alpha_rad)
    periodic = bool(alpha_deg[0] == -180. and alpha_deg[-1] == 180.)
    if attached_deg is None and airfoil.stall_rad is not None:
        attached_deg = tuple(float(np.rad2deg(v)) for v in airfoil.stall_rad)
    if periodic:
        # Exact end knots: linspace(-pi, pi) converts to -180.0/180.0 exactly.
        alpha_deg = alpha_deg.copy()
        alpha_deg[0], alpha_deg[-1] = -180., 180.
    return SectionPolar(alpha_deg, cl, np.maximum(cd, 0.), cm,
                        name or f'{airfoil.name} delta {np.rad2deg(delta):g} deg',
                        'error', periodic=periodic, attached_deg=attached_deg)


def viterna_extension(polar, stall_deg, aspect_ratio, *, grid_deg=1.):
    """Viterna-Corrigan post-stall extension of an attached-range table (option).

    Inside ``stall_deg`` the given table is kept. From each stall angle to +-90
    deg: CL = A1 sin 2a + A2 cos^2 a / sin a, CD = B1 sin^2 a + B2 cos a with
    CDmax = 1.11 + 0.018 AR (AR <= 50), A1 = CDmax/2, B1 = CDmax and A2, B2 set by
    continuity of CL and CD at the stall point (Viterna & Corrigan 1982, as
    restated by Tangler & Kocurek 2005; verify before citing). Beyond +-90 deg
    the curves are mirrored about +-90 deg with reversed lift sign; this reverse
    flow half is a symmetry assumption, not part of the 1982 method. CM is
    continued at its stall value and reversed in sign through +-90 deg.
    """
    if not isinstance(polar, SectionPolar):
        raise ValueError('viterna_extension needs a SectionPolar')
    low, high = (float(v) for v in _real(stall_deg, 'stall angles'))
    if not (polar.alpha_deg[0] <= low < high <= polar.alpha_deg[-1]):
        raise ValueError('stall angles must lie inside the table')
    ar = float(_real(aspect_ratio, 'aspect ratio'))
    if not 0 < ar <= 50:
        raise ValueError('Viterna CDmax fit requires 0 < AR <= 50')
    cd_max = 1.11+0.018*ar
    step = float(_real(grid_deg, 'grid'))
    alpha = np.unique(np.concatenate([np.arange(-180., 180.+step/2, step), [low, high], polar.alpha_deg]))
    alpha = alpha[(alpha >= -180.) & (alpha <= 180.)]
    cl = np.empty_like(alpha); cd = np.empty_like(alpha); cm = np.empty_like(alpha)
    inside = (alpha >= low) & (alpha <= high)
    cl[inside], cd[inside], cm[inside] = polar.coefficients(np.deg2rad(alpha[inside]))

    def branch(stall_angle_deg, sign):
        a_s = np.deg2rad(stall_angle_deg)
        cl_s, cd_s, cm_s = (float(v) for v in polar.coefficients(a_s))
        s, c = np.sin(abs(a_s)), np.cos(a_s)
        a2 = (sign*cl_s-cd_max*s*c)*s/c**2
        b2 = (cd_s-cd_max*s**2)/c

        def evaluate(angles_deg):
            a = np.deg2rad(angles_deg)
            mirrored = np.abs(a) > np.pi/2
            a_eff = np.where(mirrored, np.pi-np.abs(a), np.abs(a))
            s_e, c_e = np.sin(a_eff), np.cos(a_eff)
            lift = cd_max/2*np.sin(2*a_eff)+a2*c_e**2/np.where(s_e > 0, s_e, np.inf)
            drag = cd_max*s_e**2+b2*c_e
            lift = np.where(mirrored, -lift, lift)*sign
            moment = np.where(mirrored, -cm_s, cm_s)
            return lift, np.maximum(drag, 0.), moment
        return evaluate
    upper = (alpha > high)
    cl[upper], cd[upper], cm[upper] = branch(high, 1.)(alpha[upper])
    lower = (alpha < low)
    cl[lower], cd[lower], cm[lower] = branch(low, -1.)(alpha[lower])
    alpha[0], alpha[-1] = -180., 180.
    return SectionPolar(alpha, cl, cd, cm, polar.name+' + Viterna', 'error', periodic=True, attached_deg=(low, high))
