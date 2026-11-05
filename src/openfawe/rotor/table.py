"""Rotor performance tables in the google/makani ``database/m600/rotor_*.json`` format.

Source (checked 2026-10-01, google/makani commit 1d66923, Apache-2.0):
  sim/physics/rotor_database.cc   thrust = rho n^2 D^4 k_T, power = rho n^3 D^5 k_P
                                  (n = omega / 2 pi; Houghton & Carpenter 4th ed.
                                  Eq. 9.18 / 9.22), torque coefficient k_P / 2 pi,
                                  look-ups with kInterpOptionSaturate;
  sim/math/util.cc, common/c_math/util.c
                                  gsl_interp2 / InterpIndex: fractional index with
                                  end-bin extrapolation, saturated to [0, n-1],
                                  then bilinear in (omega, v) index space;
  sim/models/actuators/rotor.cc   torque = power / omega, zero below 10 rad/s.
The JSON arrays ``thrust`` and ``power`` equal the coefficient formulas at
rho = 1.225 kg/m^3 to 9e-12 relative (verified for rotor_rev4.json).
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

TORQUE_OMEGA_CUTOFF_RAD_S = 10.0  # rotor.cc CalcAeroTorque: no torque below this rate


def _fractional_index(grid, value):
    """InterpIndex with kInterpOptionDefault: bin index + fraction, extrapolating in the end bins."""
    grid = np.asarray(grid, dtype=float)
    n = len(grid)
    # Highest index < n-1 with value >= grid[index]; the last bin for value >= grid[n-2].
    index = int(np.searchsorted(grid[1:n - 1], value, side="right"))
    low, high = grid[index], grid[index + 1]
    return index + (value - low) / (high - low)


def _bilinear(z, s, t, saturate=True):
    """gsl_interp2_scaled: bilinear in fractional indices, saturated to the table."""
    size1, size2 = z.shape
    s0 = min(max(int(np.floor(s)), 0), size1 - 2)
    t0 = min(max(int(np.floor(t)), 0), size2 - 2)
    if saturate:
        s = min(max(s, 0.0), float(size1 - 1))
        t = min(max(t, 0.0), float(size2 - 1))
    ds, dt = s - s0, t - t0
    z00, z10, z01, z11 = z[s0, t0], z[s0 + 1, t0], z[s0, t0 + 1], z[s0 + 1, t0 + 1]
    return z00 + (z10 - z00) * ds + (z01 - z00) * dt + (z00 - z10 - z01 + z11) * ds * dt


@dataclass(frozen=True)
class RotorTable:
    """Thrust/power coefficients on an (omega, freestream) grid, Makani conventions."""
    name: str
    diameter_m: float
    omegas_rad_s: np.ndarray
    freestreams_m_s: np.ndarray
    thrust_coeffs: np.ndarray
    power_coeffs: np.ndarray
    source: str = ""

    def __post_init__(self):
        omegas = np.ascontiguousarray(self.omegas_rad_s, dtype=float)
        speeds = np.ascontiguousarray(self.freestreams_m_s, dtype=float)
        kt = np.ascontiguousarray(self.thrust_coeffs, dtype=float)
        kp = np.ascontiguousarray(self.power_coeffs, dtype=float)
        if omegas.ndim != 1 or speeds.ndim != 1 or len(omegas) < 2 or len(speeds) < 2:
            raise ValueError("rotor table needs at least two angular rates and two freestreams")
        if np.any(np.diff(omegas) <= 0) or np.any(np.diff(speeds) <= 0):
            raise ValueError("rotor table axes must increase strictly")
        if kt.shape != (len(omegas), len(speeds)) or kp.shape != kt.shape:
            raise ValueError("rotor coefficient tables must be (num_omegas, num_freestreams)")
        if not (np.all(np.isfinite(kt)) and np.all(np.isfinite(kp)) and np.isfinite(self.diameter_m)
                and self.diameter_m > 0):
            raise ValueError("rotor table values must be finite with a positive diameter")
        # rotor_database.cc IsValid: positive thrust coefficient near zero freestream.
        if np.any(kt[1:, 0] < 0):
            raise ValueError("thrust coefficient must be positive at the lowest freestream")
        for name, value in (("omegas_rad_s", omegas), ("freestreams_m_s", speeds),
                            ("thrust_coeffs", kt), ("power_coeffs", kp)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @classmethod
    def from_makani_json(cls, path):
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        omegas = np.asarray(data["omegas"], dtype=float)
        speeds = np.asarray(data["v_freestreams"], dtype=float)
        if data.get("num_omegas", len(omegas)) != len(omegas) or data.get("num_v_freestreams", len(speeds)) != len(speeds):
            raise ValueError(f"{path}: declared table size differs from the arrays")
        corrections = data.get("corrections", {})
        for key in ("thrust", "power"):
            factors = corrections.get(key)
            if factors is not None and np.any(np.asarray(factors, dtype=float) != 1.0):
                raise NotImplementedError(f"{path}: non-unit {key} corrections are not transcribed")
        return cls(str(data.get("name", path.stem)), float(data["diameter"]), omegas, speeds,
                   np.asarray(data["thrust_coeffs"], dtype=float), np.asarray(data["power_coeffs"], dtype=float),
                   source=str(path))

    # -- look-ups --------------------------------------------------------------
    def indices(self, omega_rad_s, freestream_m_s):
        """Fractional (s, t) indices before saturation (diagnostics)."""
        return (_fractional_index(self.omegas_rad_s, float(omega_rad_s)),
                _fractional_index(self.freestreams_m_s, float(freestream_m_s)))

    def saturated(self, omega_rad_s, freestream_m_s):
        """True when the look-up is clamped to the table range (T8 bookkeeping)."""
        return bool(omega_rad_s < self.omegas_rad_s[0] or omega_rad_s > self.omegas_rad_s[-1]
                    or freestream_m_s < self.freestreams_m_s[0] or freestream_m_s > self.freestreams_m_s[-1])

    def thrust_coefficient(self, omega_rad_s, freestream_m_s):
        s, t = self.indices(omega_rad_s, freestream_m_s)
        return float(_bilinear(self.thrust_coeffs, s, t))

    def power_coefficient(self, omega_rad_s, freestream_m_s):
        s, t = self.indices(omega_rad_s, freestream_m_s)
        return float(_bilinear(self.power_coeffs, s, t))

    def torque_coefficient(self, omega_rad_s, freestream_m_s):
        return self.power_coefficient(omega_rad_s, freestream_m_s) / (2.0 * np.pi)

    def thrust_N(self, omega_rad_s, freestream_m_s, density_kg_m3):
        n = omega_rad_s / (2.0 * np.pi)
        return density_kg_m3 * n * n * self.diameter_m ** 4 * self.thrust_coefficient(omega_rad_s, freestream_m_s)

    def power_W(self, omega_rad_s, freestream_m_s, density_kg_m3):
        """Aerodynamic (shaft) power; negative while thrusting (motor drives the rotor)."""
        n = omega_rad_s / (2.0 * np.pi)
        return density_kg_m3 * n ** 3 * self.diameter_m ** 5 * self.power_coefficient(omega_rad_s, freestream_m_s)

    def torque_Nm(self, omega_rad_s, freestream_m_s, density_kg_m3):
        """power / omega for the unsigned angular speed (rotor.cc CalcAeroTorque).

        Negative while thrusting (the air opposes the rotation); zero below the
        10 rad/s cutoff. The signed moment about the axis is direction * torque.
        """
        if omega_rad_s < TORQUE_OMEGA_CUTOFF_RAD_S:
            return 0.0
        return self.power_W(omega_rad_s, freestream_m_s, density_kg_m3) / omega_rad_s
