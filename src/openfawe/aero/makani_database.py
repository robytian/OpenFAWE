# SPDX-License-Identifier: Apache-2.0
# The aerodynamic model below is transcribed from the Makani simulator, google/makani commit 1d66923 (Apache-2.0):
# sim/physics/aero.cc (blend of the two databases, spoilers, empirical pitching moment), sim/physics/
# avl_aero_database.cc (low-incidence database, coefficient offsets), sim/physics/dvl_aero_database.cc
# (high-incidence database), sim/physics/delta_coeff_aero_database.cc (spoiler offsets), sim/physics/aero_frame.cc
# (wind axes), sim/models/rigid_bodies/wing.cc (flight condition, dimensional loads), sim/math/util.cc (table
# look-ups), config/m600/sim/aero_sim.py and config/m600/wing.py (settings). Modifications (2026): NumPy
# implementation; one database of each kind (no interpolation in the Reynolds number; for the high-incidence side this
# is not the simulator's default set, see the module text); results in the OpenFAWE body axes; the control-surface
# terms of the low-incidence database are those of ``makani_flap_increments``.
"""The whole-aircraft aerodynamics of the Makani simulator, from its databases: a borrowed reference model.

**What this is.** The Makani simulator does not solve a flow: it reads force and moment coefficients of the whole
kite from two databases and adds corrections taken from flight test. This module transcribes that model so that the
same rigid body, tether and controller can be flown on Makani's own aerodynamics (reference-plant contract of the
G5 task). **It is not the OpenFAWE method**, and nothing here sheds vorticity. A result made with it must say so.

**The model** (in the source's order; angles in radians unless a table is in degrees):

1. flight condition (``wing.cc`` lines 537-559): ``V`` is the velocity of the body origin relative to the air in
   Makani's body axes (x forward, y starboard, z down); ``alpha = atan2(V_z, V_x)``, ``beta = asin(V_y/|V|)``,
   ``q = rho |V|^2/2``, ``omega_hat = (p b, q c, r b)/(2 |V|)`` with ``|V|`` not below 1e-4 m/s;
2. low-incidence database (``avl_aero_database.cc`` lines 307-383): six coefficients, each
   ``C(alpha, beta) + C_p dp + C_q dq + C_r dr + sum_i C_di delta_i``, ``d = omega_hat - omega_hat_0`` of the file,
   the control-surface terms as in ``makani_flap_increments``; all tables bilinear in the two angles in degrees and
   held at the edge values;
3. coefficient offsets (lines 485-541, applied to the tables at their nodes when they are read): the force
   coefficients are taken to wind axes, ``(-dCD, -dCC, -dCL)`` is added, and they are taken back; the moment
   coefficients get ``(dCl + dCldbeta beta, dCm + dCmdalpha alpha, dCn + dCndbeta beta)``. The scale factors of the
   configuration are all one and are not carried;
4. spoilers (``aero.cc`` lines 268-272): only when the low-incidence database is used alone, the flaps marked in the
   spoiler file add six coefficient offsets, bilinear in the incidence and the deflection in degrees;
5. high-incidence database (``dvl_aero_database.cc`` lines 256-338): coefficients and their derivatives to the three
   rates on a grid of the two angles in radians, plus for every flap an increment on a grid of the two angles and its
   deflection (trilinear); the rate derivatives multiply ``omega_hat`` itself; the roll-damping derivative is faded
   towards ``min(C_lp, -0.2)`` between ``|p_hat|`` 0.1 and 0.15 (the source calls it a hack);
6. blend (``aero.cc`` lines 214-285): ``w = 1 - (1 - c_alpha)(1 - c_beta)`` with
   ``c = 1/2 + max(sat((x - high)/width, +-1/2), sat((low - x)/width, +-1/2))``; ``w <= 1e-6``: low-incidence only
   (with spoilers); ``w >= 1``: high-incidence only; between: ``w high + (1 - w) low`` (without spoilers);
7. dimensional loads at the body origin (``wing.cc`` lines 796-806): ``F = q A C_F``,
   ``M = q A (b C_l, c C_m, b C_n)``, and the pitching moment gets ``3500 N m * max(sin alpha, 0)``.

Not transcribed: interpolation between databases of different Reynolds number, hinge moments, the blown-wing force
of the rotors (zero above an axial inflow of 10 m/s), wind-induced apparent rates (uniform wind).

**The high-incidence side is not the simulator's default.** The default configuration
(``config/m600/sim/aero_models.py`` line 24, ``aero_sim.py`` lines 54-66) takes ten databases
``hover_model_sn03_wind_{1 ... 63}.json`` made with a rotor-wake model and interpolates them in the Reynolds number;
none of them is in the public snapshot. The one public table, ``hover_model_sn02_wind_10_no_wake_model.json``, is
the simulator's other option (``'flat plate'``) for another airframe serial: it differs in the serial, in having no
rotor-wake model and in its single Reynolds number. Whatever is computed with a blend weight above zero is therefore
not Makani's default model; a run reports how often and how far the weight leaves zero.

``MAKANI_BODY_TO_BODY`` maps Makani's body axes to those of the case files (x aft, y span, z up) and back.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from .makani_flap_increments import FLAPS, FlapIncrements, FlapSettings, _saturating_bilinear

MAKANI_BODY_TO_BODY = np.diag([-1.0, 1.0, -1.0])
COEFFICIENTS = ("CX", "CY", "CZ", "Cl", "Cm", "Cn")
RATES = ("p", "q", "r")


@dataclass(frozen=True)
class DatabaseSettings:
    """``config/m600/sim/aero_sim.py``: coefficient offsets (lines 174-244), blend angles (lines 262-270), switches."""
    cd_offset: float = 0.075
    cc_offset: float = 0.0
    cl_offset: float = -0.125
    roll_offset: float = 0.01
    pitch_offset: float = 0.0
    yaw_offset: float = 0.0
    dCldbeta: float = 0.0
    dCmdalpha: float = 0.0
    dCndbeta: float = 0.0
    low_alpha_stall_rad: float = -math.radians(20.0)
    high_alpha_stall_rad: float = math.radians(10.0)
    low_beta_stall_rad: float = -math.radians(20.0)
    high_beta_stall_rad: float = math.radians(20.0)
    blending_angle_rad: float = math.radians(10.0)
    merge_databases: bool = True
    use_spoilers: bool = True
    empirical_pitching_moment_Nm: float = 3500.0
    flaps: FlapSettings = field(default_factory=FlapSettings)

    def without_offsets(self):
        """The same settings with every coefficient offset at zero (for checks of the bare tables)."""
        from dataclasses import replace
        return replace(self, cd_offset=0.0, cc_offset=0.0, cl_offset=0.0, roll_offset=0.0, pitch_offset=0.0, yaw_offset=0.0,
                       dCldbeta=0.0, dCmdalpha=0.0, dCndbeta=0.0)


def dcm_wind_to_body(alpha_rad, beta_rad):
    """``CalcDcmWToB``: ``AngleToDcm(-beta, alpha, 0, ZYX)``; it takes wind-axis components to body-axis components."""
    ca, sa, cb, sb = math.cos(alpha_rad), math.sin(alpha_rad), math.cos(beta_rad), math.sin(beta_rad)
    about_z = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])           # passive rotation by -beta about z
    about_y = np.array([[ca, 0.0, -sa], [0.0, 1.0, 0.0], [sa, 0.0, ca]])            # passive rotation by alpha about y
    return about_y@about_z


def blend_coefficient(value_rad, low_rad, high_rad, width_rad):
    """One factor of the blend weight (``aero.cc`` lines 216-231)."""
    def half(x):
        return min(max(x, -0.5), 0.5)
    return 0.5+max(half((value_rad-high_rad)/width_rad), half((low_rad-value_rad)/width_rad))


def _fractional_index(grid, value):
    """``gsl_interp_index``: the position of ``value`` in ``grid`` in units of its index (linear beyond the ends)."""
    grid = np.asarray(grid, dtype=float)
    k = int(min(max(np.searchsorted(grid, value, side="right")-1, 0), len(grid)-2))
    return k+(float(value)-grid[k])/(grid[k+1]-grid[k])


def _trilinear(table, u, v, w):
    """``gsl_interp3`` with the saturate option on ``table[..., i, j, k]`` at fractional indices (edge values outside)."""
    n1, n2, n3 = table.shape[-3:]
    u0 = int(min(max(math.floor(u), 0), n1-2))
    v0 = int(min(max(math.floor(v), 0), n2-2))
    w0 = int(min(max(math.floor(w), 0), n3-2))
    du = min(max(u, 0.0), n1-1.0)-u0
    dv = min(max(v, 0.0), n2-1.0)-v0
    dw = min(max(w, 0.0), n3-1.0)-w0
    z = table[..., u0:u0+2, v0:v0+2, w0:w0+2]
    z000, z001, z010, z011 = z[..., 0, 0, 0], z[..., 0, 0, 1], z[..., 0, 1, 0], z[..., 0, 1, 1]
    z100, z101, z110, z111 = z[..., 1, 0, 0], z[..., 1, 0, 1], z[..., 1, 1, 0], z[..., 1, 1, 1]
    return (z000+(z100-z000)*du+(z010-z000)*dv+(z001-z000)*dw+(z000-z100-z010+z110)*du*dv+(z000-z100-z001+z101)*du*dw
            + (z000-z010-z001+z011)*dv*dw+(-z000+z100+z010+z001-z110-z101-z011+z111)*du*dv*dw)


def _bilinear_index(table, u, v):
    """``gsl_interp2_scaled`` with the saturate option on ``table[..., i, j]`` at fractional indices."""
    n1, n2 = table.shape[-2:]
    u, v = min(max(u, 0.0), n1-1.0), min(max(v, 0.0), n2-1.0)
    u0, v0 = int(min(math.floor(u), n1-2)), int(min(math.floor(v), n2-2))
    s, t = u-u0, v-v0
    return ((1.0-s)*(1.0-t)*table[..., u0, v0]+s*(1.0-t)*table[..., u0+1, v0]+(1.0-s)*t*table[..., u0, v0+1]
            + s*t*table[..., u0+1, v0+1])


class MakaniDatabase:
    """Coefficients and loads of the whole kite from Makani's databases; see the module text."""

    def __init__(self, low, *, spoilers=None, high=None, settings=None, source=None):
        self.settings = settings or DatabaseSettings()
        self.source = dict(source or {})
        s = self.settings
        self.alphas_deg = np.array(low["alphads"], dtype=float)
        self.betas_deg = np.array(low["betads"], dtype=float)
        na, nb = len(self.alphas_deg), len(self.betas_deg)
        if low.get("num_deltas") != 1:
            raise ValueError("the low-incidence database must hold one deflection (num_deltas = 1)")

        def table(key):
            value = np.asarray(low[key], dtype=float)
            if value.shape != (1, na, nb):
                raise ValueError(f"{key} must have the shape (1, num_alphas, num_betas)")
            return value[0]
        self.omega_hat_0 = np.array(low["omega_hat"], dtype=float).reshape(3)
        base = np.array([table(f"{name}tot") for name in COEFFICIENTS])                   # [6, na, nb]
        self.rate_derivatives = np.array([[table(f"{name}{rate}") for name in COEFFICIENTS] for rate in RATES])   # [3, 6, na, nb]
        # ApplyCoeffAdjustments at every node (the scale factors of the configuration are one). With every offset at
        # zero the tables are left as they are read (the source would take them to wind axes and back, which changes
        # the last bits).
        offset_w = np.array([-s.cd_offset, -s.cc_offset, -s.cl_offset])
        moment_offsets = (s.roll_offset, s.pitch_offset, s.yaw_offset, s.dCldbeta, s.dCmdalpha, s.dCndbeta)
        if np.any(offset_w != 0.0) or any(value != 0.0 for value in moment_offsets):
            for i, alpha_deg in enumerate(self.alphas_deg):
                for j, beta_deg in enumerate(self.betas_deg):
                    alpha, beta = math.radians(alpha_deg), math.radians(beta_deg)
                    dcm = dcm_wind_to_body(alpha, beta)
                    base[:3, i, j] = dcm@(dcm.T@base[:3, i, j]+offset_w)
                    base[3:, i, j] += np.array([s.dCldbeta*beta+s.roll_offset, s.dCmdalpha*alpha+s.pitch_offset,
                                                s.dCndbeta*beta+s.yaw_offset])
        self.base = base
        self.flap_increments = FlapIncrements(self.alphas_deg, self.betas_deg,
                                              np.array([[table(f"{name}d{flap+1}") for name in COEFFICIENTS] for flap in range(len(FLAPS))]),
                                              table("de1"), s.flaps)
        for value in (self.base, self.rate_derivatives):
            if not np.isfinite(value).all():
                raise ValueError("the low-incidence tables must be finite")
            value.setflags(write=False)
        self.spoilers = None
        if spoilers is not None:
            flaps = [index for index, used in enumerate(spoilers["flap_list"]) if used]
            alphas, deltas = np.array(spoilers["alphads"], dtype=float), np.array(spoilers["deltads"], dtype=float)
            tables = np.array([np.asarray(spoilers[f"{name}_offset"], dtype=float) for name in COEFFICIENTS])   # [6, flaps, na, nd]
            if tables.shape != (6, len(flaps), len(alphas), len(deltas)):
                raise ValueError("the spoiler tables must be (flaps, alphas, deltas) for each coefficient")
            self.spoilers = dict(flaps=flaps, alphas_deg=alphas, deltas_deg=deltas, tables=np.moveaxis(tables, 1, 0))   # [flap, 6, na, nd]
        self.high = None
        if high is not None:
            alphas, betas = np.array(high["alphas"], dtype=float), np.array(high["betas"], dtype=float)
            ha, hb = len(alphas), len(betas)

            def grid(key, *shape):
                value = np.asarray(high[key], dtype=float)
                if value.size != 6*int(np.prod(shape)):
                    raise ValueError(f"{key} has the wrong size")
                return value.reshape(6, *shape)
            deltas = [np.array(high[f"delta{flap+1}s"], dtype=float) for flap in range(len(FLAPS))]
            self.high = dict(alphas=alphas, betas=betas, deltas=deltas, cfm=grid("cfm", ha, hb),
                             rates=[grid(f"dcfm_d{rate}", ha, hb) for rate in RATES],
                             flap_cfm=[grid(f"dcfm{flap+1}", ha, hb, len(deltas[flap])) for flap in range(len(FLAPS))],
                             flap_rates=[[grid(f"dcfm{flap+1}_d{rate}", ha, hb, len(deltas[flap])) for rate in RATES]
                                         for flap in range(len(FLAPS))])

    @classmethod
    def from_files(cls, low_path, *, spoiler_path=None, high_path=None, settings=None):
        def read(path):
            return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(read(low_path), spoilers=read(spoiler_path), high=read(high_path), settings=settings,
                   source=dict(low=str(low_path), spoilers=None if spoiler_path is None else str(spoiler_path),
                               high=None if high_path is None else str(high_path)))

    # ----- the two databases ----------------------------------------------------------------------------------------
    def low_coefficients(self, alpha_rad, beta_rad, omega_hat, flaps_rad, thrust_coeff=0.0):
        """``AvlAeroDatabase::CalcForceMomentCoeff``: six coefficients in Makani's body axes."""
        alpha_deg, beta_deg = math.degrees(alpha_rad), math.degrees(beta_rad)
        total = np.array(_saturating_bilinear(self.alphas_deg, self.betas_deg, self.base, alpha_deg, beta_deg))
        rates = _saturating_bilinear(self.alphas_deg, self.betas_deg, self.rate_derivatives, alpha_deg, beta_deg)   # [3, 6]
        total = total+(np.asarray(omega_hat, dtype=float).reshape(3)-self.omega_hat_0)@rates
        force, moment = self.flap_increments.coefficients(alpha_rad, beta_rad, flaps_rad, thrust_coeff)
        return total+np.concatenate([force, moment])

    def spoiler_coefficients(self, alpha_rad, flaps_rad):
        """``DeltaCoeffAeroDatabase::AdjustForceMomentCoeff``: the six offsets of the flaps marked in the spoiler file."""
        total = np.zeros(6)
        if self.spoilers is None:
            return total
        sp = self.spoilers
        flaps = np.asarray(flaps_rad, dtype=float).reshape(len(FLAPS))
        for table, flap in zip(sp["tables"], sp["flaps"]):
            total += _saturating_bilinear(sp["alphas_deg"], sp["deltas_deg"], table, math.degrees(alpha_rad), math.degrees(flaps[flap]))
        return total

    def high_coefficients(self, alpha_rad, beta_rad, omega_hat, flaps_rad):
        """``DvlAeroDatabase::CalcForceMomentCoeff``: six coefficients in Makani's body axes."""
        if self.high is None:
            raise ValueError("no high-incidence database was given")
        h = self.high
        omega_hat = np.asarray(omega_hat, dtype=float).reshape(3)
        flaps = np.asarray(flaps_rad, dtype=float).reshape(len(FLAPS))
        u, v = _fractional_index(h["alphas"], alpha_rad), _fractional_index(h["betas"], beta_rad)
        total = np.array(_bilinear_index(h["cfm"], u, v))
        rates = [np.array(_bilinear_index(table, u, v)) for table in h["rates"]]
        # The source limits the roll-damping derivative once a roll rate is reached (its own comment: a hack).
        damping = rates[0][3]
        fade = min(max((abs(omega_hat[0])-0.1)/(0.15-0.1), 0.0), 1.0)
        rates[0][3] = (1.0-fade)*damping+fade*min(damping, -0.2)
        for flap in range(len(FLAPS)):
            w = _fractional_index(h["deltas"][flap], flaps[flap])
            total = total+_trilinear(h["flap_cfm"][flap], u, v, w)
            for k in range(3):
                rates[k] = rates[k]+_trilinear(h["flap_rates"][flap][k], u, v, w)
        return total+sum(omega_hat[k]*rates[k] for k in range(3))

    # ----- the blend ------------------------------------------------------------------------------------------------
    def blend_weight(self, alpha_rad, beta_rad):
        s = self.settings
        if not s.merge_databases:
            return 0.0
        c_alpha = blend_coefficient(alpha_rad, s.low_alpha_stall_rad, s.high_alpha_stall_rad, s.blending_angle_rad)
        c_beta = blend_coefficient(beta_rad, s.low_beta_stall_rad, s.high_beta_stall_rad, s.blending_angle_rad)
        return 1.0-(1.0-c_alpha)*(1.0-c_beta)

    def coefficients(self, alpha_rad, beta_rad, omega_hat, flaps_rad, thrust_coeff=0.0):
        """``Aero::CalcForceMomentCoeff``: ``(force[3], moment[3], info)`` in Makani's body axes."""
        weight = self.blend_weight(alpha_rad, beta_rad)
        info = dict(blend_weight=weight, spoilers=False)
        if 1e-6 < weight < 1.0:
            total = (weight*self.high_coefficients(alpha_rad, beta_rad, omega_hat, flaps_rad)
                     + (1.0-weight)*self.low_coefficients(alpha_rad, beta_rad, omega_hat, flaps_rad, thrust_coeff))
        elif weight <= 1e-6:
            total = self.low_coefficients(alpha_rad, beta_rad, omega_hat, flaps_rad, thrust_coeff)
            if self.settings.use_spoilers and self.spoilers is not None:
                offsets = self.spoiler_coefficients(alpha_rad, flaps_rad)
                total = total+offsets
                info["spoilers"] = bool(np.any(offsets != 0.0))
        else:
            total = self.high_coefficients(alpha_rad, beta_rad, omega_hat, flaps_rad)
        return total[:3], total[3:], info

    # ----- dimensional loads ----------------------------------------------------------------------------------------
    def condition(self, velocity_through_air_body_m_s, angular_velocity_body_rad_s, density_kg_m3):
        """Flight condition of Makani's ``Wing::UpdateAeroStates`` from OpenFAWE body-axis vectors."""
        s = self.settings.flaps
        velocity = MAKANI_BODY_TO_BODY@np.asarray(velocity_through_air_body_m_s, dtype=float).reshape(3)
        omega = MAKANI_BODY_TO_BODY@np.asarray(angular_velocity_body_rad_s, dtype=float).reshape(3)
        speed = max(float(np.linalg.norm(velocity)), 1e-4)
        alpha, beta = math.atan2(velocity[2], velocity[0]), math.asin(min(max(velocity[1]/speed, -1.0), 1.0))
        omega_hat = omega*np.array([s.span_m, s.chord_m, s.span_m])/(2.0*speed)
        return dict(alpha_rad=alpha, beta_rad=beta, airspeed_m_s=speed, dynamic_pressure_Pa=0.5*float(density_kg_m3)*speed*speed,
                    omega_hat=omega_hat)

    def loads(self, velocity_through_air_body_m_s, angular_velocity_body_rad_s, density_kg_m3, flaps_rad, thrust_coeff=0.0):
        """Force [N] and moment about the body origin [N m] in the **OpenFAWE** body axes, with the flight condition."""
        condition = self.condition(velocity_through_air_body_m_s, angular_velocity_body_rad_s, density_kg_m3)
        force, moment, info = self.coefficients(condition["alpha_rad"], condition["beta_rad"], condition["omega_hat"], flaps_rad, thrust_coeff)
        s, pressure = self.settings.flaps, condition["dynamic_pressure_Pa"]
        force = pressure*s.area_m2*force
        moment = pressure*s.area_m2*np.array([s.span_m, s.chord_m, s.span_m])*moment
        moment[1] += self.settings.empirical_pitching_moment_Nm*max(math.sin(condition["alpha_rad"]), 0.0)
        condition = dict(condition, omega_hat=condition["omega_hat"].tolist(), **info)
        return MAKANI_BODY_TO_BODY@force, MAKANI_BODY_TO_BODY@moment, condition
