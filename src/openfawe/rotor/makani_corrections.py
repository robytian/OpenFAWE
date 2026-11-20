"""Makani's empirical rotor-airframe corrections, transcribed for comparison only.

Decision 0020: these are implemented so that the physical ring-wake results can
be compared with what the Makani simulator adds empirically; they are OFF by
default and never enter the delivered loads.

Source (google/makani 1d66923, Apache-2.0):
  sim/models/actuators/rotor.cc CalcBlownWingForceMomentPos and
  config/m600/sim/rotor_sim.py: "During hover flight tests, the wing experiences
  more tension than expected ... attributed to the blown wing effect", modelled
  as a rotation of each rotor's thrust vector about the body y axis by
  thrust_vectoring_angle = 0.3 rad (empirical, 2016-08-24 flight tests), faded
  with Crossfade(0.3, 0, V_freestream, 6 m/s, 10 m/s); only the difference
  between the rotated and the original thrust is booked, at
  thrust_vectoring_pos_b = [0, 0, 0]. AngleToDcm(0, -angle, 0, ZYX) gives
  [[cos a, 0, sin a], [0, 1, 0], [-sin a, 0, cos a]] in the Makani body frame; the
  matrix is unchanged under the OpenFAWE mapping diag(-1, 1, -1), so the extra
  force points to +z (up) in the OpenFAWE frame: ~0.30 T of extra "lift".
  The tail thrust-coefficient tables (RotorThrustCoeff feeding the aero
  database deltas) belong to the whole-kite table module and are not here.
"""

import numpy as np

THRUST_VECTORING_ANGLE_RAD = 0.3
FULL_BLOWN_WING_FREESTREAM_M_S = 6.0
ZERO_BLOWN_WING_FREESTREAM_M_S = 10.0
THRUST_VECTORING_POSITION_M = np.zeros(3)


def crossfade(y0, y1, x, x_low, x_high):
    """common/c_math/util.c Crossfade: linear from y0 at x_low to y1 at x_high, saturated."""
    c = min(max((x - x_low) / max(x_high - x_low, np.finfo(float).tiny), 0.0), 1.0)
    return (1.0 - c) * y0 + c * y1


def thrust_vectoring_angle(axial_freestream_m_s):
    return crossfade(THRUST_VECTORING_ANGLE_RAD, 0.0, axial_freestream_m_s,
                     FULL_BLOWN_WING_FREESTREAM_M_S, ZERO_BLOWN_WING_FREESTREAM_M_S)


def blown_wing_force(thrust_force_N, axial_freestream_m_s):
    """Extra force (OpenFAWE frame) Makani books for one rotor: rotated thrust minus thrust."""
    angle = thrust_vectoring_angle(axial_freestream_m_s)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    force = np.asarray(thrust_force_N, dtype=float).reshape(3)
    return rotation @ force - force


def blown_wing_corrections(records):
    """Per-rotor blown-wing extra forces for a list of coupled.py load records; returns (per_rotor, total)."""
    per_rotor = [blown_wing_force(row["force_N"], row["axial_freestream_m_s"]) for row in records]
    total = np.sum(per_rotor, axis=0) if per_rotor else np.zeros(3)
    return per_rotor, total
