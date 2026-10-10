# SPDX-License-Identifier: Apache-2.0
"""Caller-supplied initial circulation derivative; no automatic initial-state inference."""

from dataclasses import dataclass
import json
import numpy as np
from .wing import owned
from .vortex import _real


@dataclass(frozen=True)
class InitialGammaRate:
    """Per-cell dGamma/dt [m^2/s^2], in chord-row/strip order at reference_time_s.

    Estimated right limits describe a numerical one-sided derivative, not a
    resolved physical start impulse. Uncertainty is a caller-declared estimate.
    """
    rate_m2_s2: np.ndarray
    reference_time_s: float
    provenance: str
    evidence_json: str
    uncertainty_m2_s2: np.ndarray | None = None

    def __post_init__(self):
        rate = owned(self.rate_m2_s2)
        if rate.ndim != 2 or not all(rate.shape):
            raise ValueError('initial Gamma rate must be a nonempty (chord, strip) array')
        time = _real(self.reference_time_s, 'initial rate reference time')
        if time.shape:
            raise ValueError('initial rate reference time must be scalar')
        if self.provenance not in ('declared_kinematic', 'estimated_right_limit'):
            raise ValueError('unknown initial Gamma rate provenance')
        try:
            evidence = json.loads(self.evidence_json)
            canonical = json.dumps(evidence, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError('initial rate evidence must be finite JSON') from error
        if not isinstance(evidence, dict) or not isinstance(evidence.get('source'), str) or not evidence['source'].strip():
            raise ValueError('initial rate evidence requires a nonempty source description')
        uncertainty = self.uncertainty_m2_s2
        if uncertainty is not None:
            uncertainty = owned(uncertainty)
            if uncertainty.shape != rate.shape or np.any(uncertainty < 0):
                raise ValueError('initial rate uncertainty must be nonnegative and match the cells')
        if self.provenance == 'estimated_right_limit':
            eps = _real(evidence.get('epsilon_s', []), 'initial rate epsilon sequence')
            if (uncertainty is None or eps.ndim != 1 or len(eps) < 2 or
                    np.any(eps <= 0) or np.any(np.diff(eps) >= 0)):
                raise ValueError('estimated initial rate requires uncertainty and decreasing positive epsilon_s')
        object.__setattr__(self, 'rate_m2_s2', rate)
        object.__setattr__(self, 'reference_time_s', time.item())
        object.__setattr__(self, 'evidence_json', canonical)
        object.__setattr__(self, 'uncertainty_m2_s2', uncertainty)

    def to_dict(self):
        return dict(rate_m2_s2=self.rate_m2_s2.tolist(), reference_time_s=self.reference_time_s,
            provenance=self.provenance, evidence_json=self.evidence_json,
            uncertainty_m2_s2=None if self.uncertainty_m2_s2 is None else self.uncertainty_m2_s2.tolist())


def check_initial_rate(value, settings, shape, initial_time):
    enabled = settings.gamma_derivative == 'bdf2_consistent'
    if (enabled and not isinstance(value, InitialGammaRate)) or (not enabled and value is not None):
        raise ValueError('bdf2_consistent requires an explicit InitialGammaRate; other schemes reject it')
    if value is not None:
        if value.rate_m2_s2.shape != shape or value.reference_time_s != initial_time:
            raise ValueError('initial Gamma rate shape or reference time differs from the run')


def startup_ratio(initial, current, previous, dt):
    """Diagnostic h*D0/DeltaGamma; invalid exact-zero denominators carry zero + mask."""
    difference = np.asarray(current)-previous
    valid = difference != 0
    ratio = np.divide(dt*initial.rate_m2_s2.reshape(difference.shape), difference,
        out=np.zeros_like(difference), where=valid)
    if not np.isfinite(ratio).all():
        raise ValueError('initial rate resolution ratio is not representable')
    return ratio, valid


def archive_fields(value):
    attributes = dict(initial_gamma_rate_contract='provided_d0_bdf2_start_v1',
        initial_gamma_rate_reference_time_s=value.reference_time_s,
        initial_gamma_rate_provenance=value.provenance,
        initial_gamma_rate_evidence_json=value.evidence_json)
    arrays = dict(initial_gamma_rate_m2_s2=value.rate_m2_s2.ravel(),
        initial_gamma_rate_uncertainty_m2_s2=np.empty(0) if value.uncertainty_m2_s2 is None
            else value.uncertainty_m2_s2.ravel())
    return attributes, arrays


def from_archive(frame, shape):
    if frame.get('initial_gamma_rate_contract') != 'provided_d0_bdf2_start_v1':
        raise ValueError('missing explicit initial Gamma rate contract')
    try:
        rate = np.asarray(frame['initial_gamma_rate_m2_s2'])
        uncertainty = np.asarray(frame['initial_gamma_rate_uncertainty_m2_s2'])
        if rate.shape != (int(np.prod(shape)),) or uncertainty.shape not in ((0,),rate.shape):
            raise ValueError('invalid archived initial rate or uncertainty shape')
        return InitialGammaRate(rate.reshape(shape),frame['initial_gamma_rate_reference_time_s'],
            frame['initial_gamma_rate_provenance'],frame['initial_gamma_rate_evidence_json'],
            None if not uncertainty.size else uncertainty.reshape(shape))
    except KeyError as error:
        raise ValueError('missing archived initial Gamma rate input') from error
