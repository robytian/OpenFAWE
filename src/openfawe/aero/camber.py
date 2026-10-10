# SPDX-License-Identifier: Apache-2.0
"""Explicit profile-to-mean-surface conversion and ideal companion polars.

SHARPy's generate_strip motivates sampling an explicit mean line, scaling by
chord and placing it in a section frame. This implementation uses the retained
OpenFAWE chord-plane frame, not SHARPy's beam frame or shifted grid convention.
No thickness, separation, viscous drag or automatic polar correction is added.
See third_party/CAMBER-SOURCES.md and the mean-surface implementation contract.
"""
import numpy as np

from .vortex import _real, _norm
from .wing import owned


def normalize_contour(coordinates, *, datum='source_chord'):
    """Retain a source chord datum, or explicitly remove an endpoint chord line.

    Every option retains the normalized source chord origin. Any trailing-edge
    mean-line closure is explicit and recorded separately from polar data.
    Input order is preserved; nonmonotone branches are rejected downstream.
    """
    points = _real(coordinates, 'airfoil contour').copy()
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError('airfoil contour requires at least five ordered x,z points')
    nose = int(np.argmin(points[:,0]))
    if nose in (0, len(points)-1):
        raise ValueError('contour must run from trailing edge around nose to trailing edge')
    original_nose = points[nose].copy()
    te = points[[0,-1]].mean(axis=0)
    metrics = dict(datum=datum, source_nose=original_nose.tolist(), source_te_mean=te.tolist(),
                   source_te_gap=float(abs(points[0,1]-points[-1,1])))
    if datum in ('source_chord','source_chord_te_linear_closure'):
        if (np.min(points[:,0]) < 0 or np.max(points[:,0]) > 1
                or original_nose[0] > .001 or abs(original_nose[1]) > .003
                or np.any(points[[0,-1],0] != 1.)):
            raise ValueError('source chord endpoint discrepancy exceeds declared limit')
        if datum == 'source_chord' and abs(te[1]) > 1e-12:
            raise ValueError('source chord has a nonzero mean trailing-edge ordinate')
        if datum == 'source_chord_te_linear_closure':
            # Explicit geometric approximation for a retained finite-thickness
            # trailing edge. The strict source_chord option still rejects it.
            if abs(te[1]) > .0005 or abs(points[0,1]-points[-1,1]) > .001:
                raise ValueError('source trailing-edge gap exceeds the declared converter range')
            points[:,1] -= te[1]*points[:,0]
            metrics.update(mean_line_linear_correction_slope=float(-te[1]),
                polar_alpha_datum_changed=False,
                limitation='Closed mean-surface approximation; source CFD polar is unchanged')
    elif datum == 'normalized_source_chord':
        # Retain the normalized input chord origin: a sampled leftmost point
        # need not be the true nose (notably the nearly symmetric f8 contour).
        if (np.min(points[:,0]) < -2e-5 or np.max(points[:,0]) > 1.00001
                or np.max(abs(points[[0,-1],0]-1.)) > .0002):
            raise ValueError('normalized airfoil coordinate range is unsupported')
        points[:,0]=np.clip(points[:,0],0.,1.)
        points[[0,-1],0]=1.
        points[:,1]-=te[1]*points[:,0]
        roundoff=(points[:,0]==0.) & (abs(points[:,1])<1e-8)
        points[roundoff,1]=0.
        metrics.update(mean_line_linear_correction_slope=float(-te[1]),
            minimum_x_clipped_to_zero=bool(original_nose[0]<0),
            zero_origin_roundoff_count=int(np.count_nonzero(roundoff)),
            leading_datum='Original normalized (0,0), not the lowest-x sampled ordinate')
    else:
        raise ValueError('unknown contour chord datum')
    return owned(points), metrics


def mean_line(contour, x):
    """Interpolate monotone contour branches, retaining vertical-edge envelopes."""
    points, x = _real(contour, 'contour'), _real(x, 'mean-line abscissae')
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 5
            or x.ndim != 1 or np.any(x < 0) or np.any(x > 1)):
        raise ValueError('normalized contour and one-dimensional abscissae required')
    nose = int(np.argmin(points[:,0]))
    branches = [points[:nose+1][::-1], points[nose:]]
    if (nose in (0,len(points)-1) or np.any(points[:,0]<0) or np.any(points[:,0]>1)
            or any(np.any(np.diff(b[:,0])<0) or b[-1,0]!=1. for b in branches)):
        raise ValueError('contour branches must increase in x without reordering')
    # Determine which branch is upper away from endpoints, before collapsing
    # repeated x coordinates. No arbitrary sorting of a folded contour.
    upper = int(np.interp(.5,branches[1][:,0],branches[1][:,1]) >
                np.interp(.5,branches[0][:,0],branches[0][:,1]))
    values=[]
    for index, branch in enumerate(branches):
        ux, starts = np.unique(branch[:,0],return_index=True)
        z = (np.maximum.reduceat if index==upper else np.minimum.reduceat)(branch[:,1],starts)
        if ux[0]>0:
            ux=np.r_[0.,ux];z=np.r_[0.,z]
        values.append(np.interp(x,ux,z))
    if np.any(values[upper] < values[1-upper]-1e-12):
        raise ValueError('airfoil upper/lower branches cross')
    mean=.5*(values[0]+values[1])
    # The source-chord closure and normalized-source processing define zero
    # mean endpoints. Reject a meaningful mismatch rather than remove camber.
    if np.any(np.abs(mean[(x==0)|(x==1)])>1e-12):
        raise ValueError('mean line does not meet the declared chord endpoints')
    mean[(x==0)|(x==1)]=0.
    return owned(mean)


def thin_airfoil_coefficients(x, camber):
    """Exact thin-airfoil integrals of the supplied piecewise-linear mean line.

    x=(1-cos(theta))/2. alpha0=integral(z'*(1-cos(theta)))/pi;
    Cm(c/4)=integral(z'*(cos(2theta)-cos(theta)))/2. Positive camber
    gives negative zero-lift angle and nose-down (negative y) quarter-chord Cm.
    This is an ideal small-disturbance companion model, not source polar data.
    """
    x, z = _real(x,'camber x'), _real(camber,'camber')
    if (x.ndim!=1 or len(x)<2 or z.shape!=x.shape or x[0]!=0 or x[-1]!=1
            or np.any(np.diff(x)<=0) or z[0]!=0 or z[-1]!=0):
        raise ValueError('closed mean line on strictly increasing normalized x required')
    theta=np.arccos(1.-2*x); slope=np.diff(z)/np.diff(x)
    i0=float(slope@np.diff(theta)); i1=float(slope@np.diff(np.sin(theta)))
    i2=float(.5*slope@np.diff(np.sin(2*theta)))
    return (i0-i1)/np.pi, .5*(i2-i1)


def node_camber_normals(wing):
    """Node ordinate directions from ordered chord-plane panel normals."""
    if hasattr(wing,'wings'):
        raise ValueError('construct camber per connected Wing, not across an assembly')
    normals=np.empty((wing.node_count,3))
    normals[0],normals[-1]=wing.normal[0],wing.normal[-1]
    normals[1:-1]=wing.normal[:-1]+wing.normal[1:]
    chord=wing.trailing_edge-wing.leading_edge
    tangent=chord/_norm(chord)[:,None]
    normals-=np.sum(normals*tangent,axis=1)[:,None]*tangent
    length=_norm(normals)
    if np.any(length<=1e-12):
        raise ValueError('ambiguous node camber normal at a folded span')
    return owned(normals/length[:,None])


def section_offsets(wing, normalized_mean_lines):
    """Place explicit normalized profiles in the declared chord-plane frames."""
    z=_real(normalized_mean_lines,'section mean lines')
    if z.ndim!=2 or z.shape[0]!=wing.node_count:
        raise ValueError('one mean line per wing node required')
    chord=_norm(wing.trailing_edge-wing.leading_edge)
    return owned(chord[:,None,None]*z[:,:,None]*node_camber_normals(wing)[:,None,:])
