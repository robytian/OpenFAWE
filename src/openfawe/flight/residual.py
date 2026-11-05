"""Residual and work account of a prescribed-motion run (G4 contract, section 3 items 5 and 6).

The motion is given, so Newton's and Euler's equations are not solved: they are evaluated. For every
recorded step the account takes the loads the models gave (lifting surfaces, and rotors and tether
when the run had them), adds gravity, and reports what is still missing for the body to follow the
motion:

    F_req = m a_cm - (F_aero + F_rotor + F_tether + m g)
    M_req = dH/dt  - (M_aero + M_rotor + M_tether)            about the centre of mass,

with ``a_cm = a_O + alpha x c + omega x (omega x c)`` (``c`` from the body origin to the centre of
mass), ``H = I omega`` and ``dH/dt = I alpha + omega x (I omega)``, ``I`` being the inertia about the
centre of mass turned into the inertial frame. For a declared motion the residual is the load the
controller and everything not modelled would have to supply; for a replayed flight it is model error
plus measurement error.

Accelerations are the motion's stored ones where it stores them; otherwise they are central
differences of the stored velocities (one-sided at the two ends), which is said in the result
(``acceleration_source``). A differenced acceleration is wrong at a sample where the true one jumps.

Rotor moments that the rotor records leave out are added here when the rotor inertia is given
(transcribed from google/makani sim/models/actuators/rotor.cc, commit 1d66923, Apache-2.0,
``CalcGyroMoment`` and ``CalcReactMoment``): the gyroscopic moment ``L x omega`` with
``L = I_r Omega dir axis`` and the reaction moment ``-I_r dOmega/dt dir axis``, where ``Omega`` is the
unsigned rotor speed and ``dir`` the rotation sense about the thrust axis; ``dOmega/dt`` is the
difference of the recorded speeds.

Power: every load's rate of work on the body is ``F . v_O + M_O . omega``; gravity's is
``m g . v_cm``; the required power is ``P_req = F_req . v_cm + M_req . omega``. By the definitions
above their sum is ``m a_cm . v_cm + (dH/dt) . omega``, which is the rate of the kinetic energy
``E_kin = m |v_cm|^2 / 2 + omega . I omega / 2`` IF the accelerations used are the true derivatives of
the stored velocities. The account reports that sum as ``kinetic_energy_rate_W``: it is an identity
of the account, not a check of any load. What is a check - of the motion samples and of the
accelerations taken, not of the loads - is the comparison with the difference quotient of the
kinetic energy over the record times, reported as ``kinetic_energy_rate_from_samples_W`` (three
records or more); the two differ where the accelerations are differenced across a jump or come from
a measured motion. The shaft power of the rotors is a separate quantity (the table's power, negative
while the motors drive the rotors) and is not part of this mechanical balance of the airframe.

One record list is one run. Under the pylon model of decision 0028 the pylons are a run of their
own: their loads are not in the account unless the caller adds them to the records first, and the
caller says so through ``not_included``.
"""

from dataclasses import dataclass

import numpy as np

LOADS = ("aero", "rotor", "tether")
_FORCE = dict(aero="force_inertial_N", rotor="rotor_force_inertial_N", tether="tether_force_inertial_N")
_MOMENT = dict(aero="moment_origin_inertial_Nm", rotor="rotor_moment_origin_inertial_Nm",
               tether="tether_moment_origin_inertial_Nm")


@dataclass(frozen=True)
class RigidBody:
    """Mass, centre of mass (body axes) and inertia about the centre of mass (body axes)."""
    mass_kg: float
    centre_of_mass_body_m: np.ndarray
    inertia_body_kg_m2: np.ndarray

    def __post_init__(self):
        centre = np.array(self.centre_of_mass_body_m, dtype=float).reshape(3)
        inertia = np.array(self.inertia_body_kg_m2, dtype=float).reshape(3, 3)
        if not (np.isfinite(self.mass_kg) and self.mass_kg > 0. and np.all(np.isfinite(centre)) and np.all(np.isfinite(inertia))):
            raise ValueError("mass must be positive and all rigid-body properties finite")
        if not np.allclose(inertia, inertia.T, rtol=1e-12, atol=1e-9) or np.any(np.linalg.eigvalsh(inertia) <= 0.):
            raise ValueError("the inertia tensor must be symmetric and positive definite")
        for value in (centre, inertia):
            value.setflags(write=False)
        object.__setattr__(self, "mass_kg", float(self.mass_kg))
        object.__setattr__(self, "centre_of_mass_body_m", centre)
        object.__setattr__(self, "inertia_body_kg_m2", inertia)


def load_rigid_body(path):
    """The rigid-body properties of an ``openfawe.rigid_body.v1`` case file."""
    import json
    from pathlib import Path
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "openfawe.rigid_body.v1" or data.get("frame") != "x_downstream_y_span_z_up":
        raise ValueError(f"{path}: not an openfawe.rigid_body.v1 file in the x_downstream_y_span_z_up frame")
    return RigidBody(data["mass_kg"], data["centre_of_mass_m"], data["inertia_about_centre_of_mass_kg_m2"])


def differences(times, values):
    """Central differences on the samples, one-sided (second order) at the two ends."""
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if len(times) < 3:
        raise ValueError("differenced accelerations need at least three samples")
    return np.gradient(values, times, axis=0, edge_order=2)


def accelerations(motion):
    """Origin acceleration and angular acceleration at every sample, with where each came from."""
    out, source = [], []
    for stored, velocity in ((motion.acceleration_m_s2, motion.velocity_m_s),
                             (motion.angular_acceleration_rad_s2, motion.angular_velocity_rad_s)):
        if stored is not None:
            out.append(np.asarray(stored, dtype=float))
            source.append("stored with the motion")
        else:
            out.append(differences(motion.time_s, velocity))
            source.append("central difference of the stored velocity")
    return out[0], out[1], dict(linear=source[0], angular=source[1])


def rotor_inertial_moments(records, motion, discs, inertia_kg_m2):
    """Gyroscopic and reaction moments of the rotors on the body at every record, inertial components (n, 3) each.

    The reaction moment needs the rate of the rotor speeds, a difference over the records: with fewer
    than three records it cannot be taken and the call is refused.
    """
    if len(records) < 3:
        raise ValueError("the rotors' reaction moment needs at least three records (the speed is differenced)")
    speeds = np.array([record["rotor_omega_rad_s"] for record in records], dtype=float)
    times = np.array([record["time_s"] for record in records], dtype=float)
    rates = differences(times, speeds)
    sense = np.array([disc.direction for disc in discs], dtype=float)
    axes = np.array([disc.axis for disc in discs], dtype=float)            # body axes
    gyro, reaction = np.zeros((len(records), 3)), np.zeros((len(records), 3))
    for k, record in enumerate(records):
        pose = motion.pose(record["time_s"])
        momentum = pose.rotation@((inertia_kg_m2*speeds[k]*sense)@axes)     # sum of the rotors' angular momenta
        gyro[k] = np.cross(momentum, pose.angular_velocity_rad_s)
        reaction[k] = pose.rotation@((-inertia_kg_m2*rates[k]*sense)@axes)
    return gyro, reaction


def account(records, motion, body, *, gravity_m_s2=(0., 0., -9.81), discs=None, rotor_inertia_kg_m2=None,
            not_included=("fuselage aerodynamics", "added mass", "flexibility")):
    """Residual force and moment and the power account for every record of a run.

    ``records`` are the step records of ``run_prescribed`` (times on motion samples). ``discs`` and
    ``rotor_inertia_kg_m2`` add the rotors' gyroscopic and reaction moments (both or neither).
    ``not_included`` names what the run did not model, for the summary: the account cannot know it
    (a run of the wing and tails without the pylons, for example, must say "pylons"). Returns
    ``(rows, summary)``: one dict per record, and what was included and where the accelerations came from.
    """
    if (discs is None) != (rotor_inertia_kg_m2 is None):
        raise ValueError("give both the discs and the rotor inertia, or neither")
    if not records:
        return [], dict(steps=0)
    gravity = np.asarray(gravity_m_s2, dtype=float).reshape(3)
    linear, angular, source = accelerations(motion)
    index = {float(t): k for k, t in enumerate(motion.time_s)}
    present = [name for name in LOADS if _FORCE[name] in records[0]]
    for record in records:                                   # one set of loads for the whole run
        if [name for name in LOADS if _FORCE[name] in record] != present:
            raise ValueError("the records do not all carry the same loads")
    inertial = None
    if discs is not None:
        if "rotor_omega_rad_s" not in records[0]:
            raise ValueError("the records hold no rotor speeds")
        inertial = rotor_inertial_moments(records, motion, discs, float(rotor_inertia_kg_m2))
    rows = []
    for k, record in enumerate(records):
        sample = index.get(float(record["time_s"]))
        if sample is None:
            raise ValueError("a record is not on a motion sample")
        pose = motion.pose(motion.time_s[sample])
        rotation, omega = pose.rotation, pose.angular_velocity_rad_s
        offset = rotation@body.centre_of_mass_body_m
        inertia = rotation@body.inertia_body_kg_m2@rotation.T
        velocity_cm = pose.velocity_m_s+np.cross(omega, offset)
        acceleration_cm = linear[sample]+np.cross(angular[sample], offset)+np.cross(omega, np.cross(omega, offset))
        momentum_rate = inertia@angular[sample]+np.cross(omega, inertia@omega)
        force, moment_cm, powers = np.zeros(3), np.zeros(3), {}
        for name in present:
            f, m = np.asarray(record[_FORCE[name]], dtype=float), np.asarray(record[_MOMENT[name]], dtype=float)
            force += f
            moment_cm += m-np.cross(offset, f)
            powers[name] = float(f@pose.velocity_m_s+m@omega)
        if inertial is not None:
            extra = inertial[0][k]+inertial[1][k]
            moment_cm += extra
            powers["rotor_gyroscopic_and_reaction"] = float(extra@omega)
        weight = body.mass_kg*gravity
        required_force = body.mass_kg*acceleration_cm-force-weight
        required_moment = momentum_rate-moment_cm
        required_power = float(required_force@velocity_cm+required_moment@omega)
        gravity_power = float(weight@velocity_cm)
        row = dict(step=record["step"], time_s=record["time_s"],
                   required_force_inertial_N=required_force.tolist(), required_moment_cm_inertial_Nm=required_moment.tolist(),
                   required_force_body_N=(rotation.T@required_force).tolist(),
                   required_moment_cm_body_Nm=(rotation.T@required_moment).tolist(),
                   inertia_force_inertial_N=(body.mass_kg*acceleration_cm).tolist(), weight_inertial_N=weight.tolist(),
                   angular_momentum_rate_inertial_Nm=momentum_rate.tolist(),
                   modelled_force_inertial_N=force.tolist(), modelled_moment_cm_inertial_Nm=moment_cm.tolist(),
                   power_W=dict(powers, gravity=gravity_power, required=required_power),
                   kinetic_energy_J=float(.5*body.mass_kg*velocity_cm@velocity_cm+.5*omega@inertia@omega),
                   kinetic_energy_rate_W=float(required_power+sum(powers.values())+gravity_power))
        if inertial is not None:
            row.update(rotor_gyroscopic_moment_inertial_Nm=inertial[0][k].tolist(),
                       rotor_reaction_moment_inertial_Nm=inertial[1][k].tolist())
        rows.append(row)
    if len(rows) >= 3:                                       # the check that is not an identity: see the module text
        sampled = differences([row["time_s"] for row in rows], [row["kinetic_energy_J"] for row in rows])
        for row, value in zip(rows, sampled):
            row["kinetic_energy_rate_from_samples_W"] = float(value)
    summary = dict(steps=len(rows), loads_included=present+(["rotor gyroscopic and reaction moments"] if inertial is not None else []),
                   loads_absent=[name for name in LOADS if name not in present], acceleration_source=source,
                   gravity_m_s2=gravity.tolist(), mass_kg=body.mass_kg, centre_of_mass_body_m=body.centre_of_mass_body_m.tolist(),
                   largest_required_force_N=float(max(np.linalg.norm(row["required_force_inertial_N"]) for row in rows)),
                   largest_required_moment_Nm=float(max(np.linalg.norm(row["required_moment_cm_inertial_Nm"]) for row in rows)),
                   largest_kinetic_energy_rate_gap_W=(float(max(abs(row["kinetic_energy_rate_W"]-row["kinetic_energy_rate_from_samples_W"])
                                                                for row in rows)) if len(rows) >= 3 else None),
                   not_modelled=list(not_included)
                   + ([] if inertial is not None or "rotor" not in present else ["rotor gyroscopic and reaction moments"]))
    return rows, summary
