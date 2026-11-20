# SPDX-License-Identifier: Apache-2.0
# The control-surface terms below are transcribed from the Makani simulator, google/makani commit 1d66923
# (Apache-2.0): sim/physics/avl_aero_database.cc (CalcForceMomentCoeff, CalcAvlAeroCoeffs, AdjustNonlinearFlaps),
# sim/models/rigid_bodies/wing.cc (the flight condition and the dimensional loads), config/m600/sim/aero_sim.py and
# config/m600/wing.py (the settings). Modifications (2026): NumPy implementation; only the control-surface terms of
# the coefficient sum are kept (the zero-deflection loads come from the OpenFAWE aerodynamics); the result is given
# in the OpenFAWE body axes.
"""Control-surface increments from Makani's low-incidence aerodynamic database (a declared stand-in).

**What this is.** The OpenFAWE crosswind section data have no flap dimension, so the lifting-surface solution is that
of the kite with all control surfaces at zero. For the first closed loop (G5 step 2, closed-loop contract, item D2)
the effect of the eight control surfaces is added as a whole-aircraft force and moment at the body origin, taken from
the same tables Makani's own simulator uses in forward flight:

    dC = sum_i C_di(alpha, beta) * delta_i [deg]

with the derivative tables of ``database/m600/m600_aswing_baseline.json`` (per degree; bilinear in the incidence and
sideslip in degrees, held at the edge values outside the grid). It is **not** the OpenFAWE method: nothing here sheds
vorticity or changes the circulation of a panel. A result that uses it must say so.

**What is transcribed** (``avl_aero_database.cc`` lines 307-383 and 435-456, in the source's order):

1. the rudder deflection beyond +-10 deg is scaled (0.4692 above, 0.5374 below; ``use_nonlinear_flaps``);
2. every deflection is saturated to ``[min_avl_flap_angles, max_avl_flap_angles]`` (-0.5 rad to +5 deg for the six
   wing flaps, +-0.5 rad for elevator and rudder);
3. the elevator deflection in degrees is reduced by the table ``de1(alpha, beta)``;
4. the six derivatives of the elevator and of the rudder are multiplied by ``1 + thrust_coeff`` (the mean rotor
   thrust coefficient of the step before, ``T / (rho/2 pi R^2 max(|V|, 30 m/s)^2)``);
5. the coefficients are made dimensional with ``A = 32.9 m^2``, ``b = 25.66 m``, ``c = 1.28 m`` at the body origin:
   ``F = q A (CX, CY, CZ)``, ``M = q A (b Cl, c Cm, b Cn)`` in Makani's body axes (x forward, y starboard, z down).

Not transcribed: the offsets and scale factors of the database (the configured scale factors of the flap derivatives
are all 1), the spoiler tables of the centre flaps, the large-incidence database and the blend with it, hinge moments.

The flight condition is Makani's: the velocity of the body origin relative to the air in Makani's body axes
``V``, ``alpha = atan2(V_z, V_x)``, ``beta = asin(V_y / |V|)``, ``q = rho |V|^2 / 2`` (``wing.cc`` lines 537-562).
Flap order: A1, A2, A4, A5, A7, A8, elevator, rudder (``system/labels.h``).
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np

FLAPS = ("A1", "A2", "A4", "A5", "A7", "A8", "Ele", "Rud")
ELEVATOR, RUDDER = 6, 7
COEFFICIENTS = ("CX", "CY", "CZ", "Cl", "Cm", "Cn")
MAKANI_BODY_TO_BODY = np.diag([-1.0, 1.0, -1.0])          # forward/starboard/down -> chord-aft/span/up
FIVE_DEG = math.radians(5.0)


@dataclass(frozen=True)
class FlapSettings:
    """Settings of ``config/m600/sim/aero_sim.py`` (lines 93-108, 320-324) and ``config/m600/wing.py`` (193-200)."""
    area_m2: float = 32.9
    span_m: float = 25.66
    chord_m: float = 1.28
    min_rad: tuple = (-0.5,)*8
    max_rad: tuple = (FIVE_DEG,)*6+(0.5, 0.5)
    nonlinear_rudder: bool = True
    rudder_positive_threshold_rad: float = math.radians(10.0)
    rudder_positive_scaling: float = 0.4692
    rudder_negative_threshold_rad: float = -math.radians(10.0)
    rudder_negative_scaling: float = 0.5374


def _saturating_bilinear(xs, ys, table, x, y):
    """Bilinear interpolation of ``table[i, j]`` on the grid ``(xs, ys)``, held at the edge values outside it."""
    x, y = min(max(float(x), xs[0]), xs[-1]), min(max(float(y), ys[0]), ys[-1])
    i = int(min(max(np.searchsorted(xs, x, side="right")-1, 0), len(xs)-2))
    j = int(min(max(np.searchsorted(ys, y, side="right")-1, 0), len(ys)-2))
    s, t = (x-xs[i])/(xs[i+1]-xs[i]), (y-ys[j])/(ys[j+1]-ys[j])
    return ((1.0-s)*(1.0-t)*table[..., i, j]+s*(1.0-t)*table[..., i+1, j]
            + (1.0-s)*t*table[..., i, j+1]+s*t*table[..., i+1, j+1])


@dataclass(frozen=True)
class FlapIncrements:
    """Derivative tables ``derivatives[flap, coefficient, alpha, beta]`` per degree and the elevator offset table."""
    alphas_deg: np.ndarray
    betas_deg: np.ndarray
    derivatives: np.ndarray
    elevator_offset_deg: np.ndarray
    settings: FlapSettings = field(default_factory=FlapSettings)
    source: str = ""

    def __post_init__(self):
        for name in ("alphas_deg", "betas_deg", "derivatives", "elevator_offset_deg"):
            value = np.array(getattr(self, name), dtype=float)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        na, nb = len(self.alphas_deg), len(self.betas_deg)
        if self.alphas_deg.ndim != 1 or self.betas_deg.ndim != 1 or na < 2 or nb < 2:
            raise ValueError("the incidence and sideslip grids need at least two points each")
        if np.any(np.diff(self.alphas_deg) <= 0) or np.any(np.diff(self.betas_deg) <= 0):
            raise ValueError("the incidence and sideslip grids must increase strictly")
        if self.derivatives.shape != (len(FLAPS), len(COEFFICIENTS), na, nb) or self.elevator_offset_deg.shape != (na, nb):
            raise ValueError("the derivative tables must be (8 flaps, 6 coefficients, alphas, betas)")
        if not (np.isfinite(self.derivatives).all() and np.isfinite(self.elevator_offset_deg).all()):
            raise ValueError("the tables must be finite")
        if len(self.settings.min_rad) != len(FLAPS) or len(self.settings.max_rad) != len(FLAPS):
            raise ValueError("one lower and one upper deflection limit per flap")

    @classmethod
    def from_makani_json(cls, path, settings=None):
        """Read a Makani low-incidence database (``m600_aswing_*.json``): the keys the simulator reads for the flaps."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("num_deltas") != 1:
            raise ValueError("the database must hold one deflection (num_deltas = 1), as the source requires")
        na, nb = int(data["num_alphas"]), int(data["num_betas"])

        def table(key):
            value = np.asarray(data[key], dtype=float)
            if value.shape != (1, na, nb):
                raise ValueError(f"{key} must have the shape (1, num_alphas, num_betas)")
            return value[0]
        derivatives = np.array([[table(f"{coefficient}d{flap+1}") for coefficient in COEFFICIENTS] for flap in range(len(FLAPS))])
        return cls(data["alphads"], data["betads"], derivatives, table("de1"), settings or FlapSettings(), source=str(path))

    # ----- the source's steps ----------------------------------------------------------------------------------------
    def effective_deflections_deg(self, flaps_rad, alpha_rad, beta_rad):
        """Deflections as they multiply the derivatives: rudder scaling, saturation, degrees, elevator offset."""
        flaps = np.array(flaps_rad, dtype=float).reshape(len(FLAPS))
        if not np.isfinite(flaps).all():
            raise FloatingPointError("flap deflections left the finite range")
        s = self.settings
        if s.nonlinear_rudder:                              # AdjustNonlinearFlaps (its two tests cannot both apply)
            rudder = flaps[RUDDER]
            if rudder-s.rudder_positive_threshold_rad > 0.0:
                flaps[RUDDER] = s.rudder_positive_threshold_rad+s.rudder_positive_scaling*(rudder-s.rudder_positive_threshold_rad)
            if rudder-s.rudder_negative_threshold_rad < 0.0:
                flaps[RUDDER] = s.rudder_negative_threshold_rad+s.rudder_negative_scaling*(rudder-s.rudder_negative_threshold_rad)
        flaps = np.minimum(np.maximum(flaps, s.min_rad), s.max_rad)
        degrees = flaps*(180.0/math.pi)
        degrees[ELEVATOR] -= float(_saturating_bilinear(self.alphas_deg, self.betas_deg, self.elevator_offset_deg,
                                                        math.degrees(alpha_rad), math.degrees(beta_rad)))
        return degrees

    def derivatives_at(self, alpha_rad, beta_rad):
        """The 8 x 6 derivatives per degree at an incidence and sideslip (radians in, the tables are in degrees)."""
        return _saturating_bilinear(self.alphas_deg, self.betas_deg, self.derivatives, math.degrees(alpha_rad), math.degrees(beta_rad))

    def coefficients(self, alpha_rad, beta_rad, flaps_rad, thrust_coeff=0.0):
        """Increments ``(CX, CY, CZ)`` and ``(Cl, Cm, Cn)`` of the eight deflections, Makani's body axes."""
        table = np.array(self.derivatives_at(alpha_rad, beta_rad))
        table[[ELEVATOR, RUDDER]] *= 1.0+float(thrust_coeff)
        total = self.effective_deflections_deg(flaps_rad, alpha_rad, beta_rad)@table
        return total[:3], total[3:]

    def loads(self, velocity_through_air_body_m_s, density_kg_m3, flaps_rad, thrust_coeff=0.0):
        """Force [N] and moment about the body origin [N m] in the **OpenFAWE** body axes, with the flight condition.

        ``velocity_through_air_body_m_s`` is the velocity of the body origin relative to the air in the OpenFAWE body
        axes (minus the relative wind of ``flight.kinematics.relative_wind_body``).
        """
        velocity = MAKANI_BODY_TO_BODY@np.asarray(velocity_through_air_body_m_s, dtype=float).reshape(3)
        speed = float(np.linalg.norm(velocity))
        if not speed > 0.0:
            raise ValueError("no relative wind: the flap increments are undefined")
        alpha, beta = math.atan2(velocity[2], velocity[0]), math.asin(min(max(velocity[1]/speed, -1.0), 1.0))
        pressure = 0.5*float(density_kg_m3)*speed*speed
        force, moment = self.coefficients(alpha, beta, flaps_rad, thrust_coeff)
        s = self.settings
        force = pressure*s.area_m2*force
        moment = pressure*s.area_m2*np.array([s.span_m, s.chord_m, s.span_m])*moment
        return (MAKANI_BODY_TO_BODY@force, MAKANI_BODY_TO_BODY@moment,
                dict(alpha_rad=alpha, beta_rad=beta, dynamic_pressure_Pa=pressure, airspeed_m_s=speed))
