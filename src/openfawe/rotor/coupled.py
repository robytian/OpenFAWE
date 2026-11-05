"""Rotor discs coupled to a wing free-wake run (G3r contract §3 "interface").

Per physical step, from the state at time t:
  1. each hub's apparent wind = ambient + wing network induction (bound, TE,
     free rows) + the other discs' rings - hub velocity (the disc's own rings
     are excluded: the table already contains the rotor's self-induction);
  2. table loads and one shed ring per disc (Gamma = T dt / (rho A)) are
     appended to the state's ring set at the rim;
  3. the ordinary LL/VSM advance convects rows and rings together and solves
     the new circulation with the rings' induction as known velocity.
Nothing in the wing advance changes when a state carries no rings.
"""

from dataclasses import replace

import numpy as np

from ..aero.cores import edge_radii
from ..aero.vortex import segments_velocity
from ..aero.wake import WakeState, advance_wake, ambient_velocity, kernel_options, wake_network
from ..aero.vsm_wake import VsmWakeState, advance_vsm_wake
from .ring_wake import RingWake


def hub_apparent_wind(state, config, disc, disc_index, inflow, hub_velocity=(0.0, 0.0, 0.0)):
    """Air velocity relative to the hub at the state's time (see module docstring)."""
    hub = disc.centre_m[None, :]
    velocity = ambient_velocity(inflow, hub, state.time_s).astype(float).copy()
    nodes, edges, gamma = wake_network(state)
    if len(edges):
        radii = edge_radii(state.wing, edges, state.cores, config.core_radius_m)
        velocity += segments_velocity(hub, nodes, edges, gamma, model=config.core_model, core_radius=radii,
                                      **kernel_options(config))
    if state.rings is not None and len(state.rings.edges):
        ring_core = getattr(state.rings, "core_m", None)
        velocity += state.rings.induced_velocity(hub, config.core_model, config.core_radius_m if ring_core is None else ring_core,
                                                 exclude_source=disc_index, singularity="zero",
                                                 **kernel_options(config))
    return velocity[0] - np.asarray(hub_velocity, dtype=float)


def shed_rotor_rings(state, config, discs, omegas_rad_s, inflow, *, nodes_per_ring=16,
                     hub_velocities=None, phase=0.0, surface_core_m=None, turbine_extension=None,
                     free_convection_steps=None, ring_core_m=None, ring_bound_core_m=None, ring_surface_overlap=None,
                     ring_sheet_drift=False, ring_ambient_spacing=False):
    """Return (state with the new rings appended, per-disc load records) at the state's time.

    ``surface_core_m`` sets the rings' core for induction at lifting-surface points
    (None = wake core); ``turbine_extension`` passes the optional Glauert/Buhl
    branch for turbine states beyond C_T = 1 (default off, decision 0020);
    ``free_convection_steps`` (None = every ring convects freely) is the frozen far
    wake of the rings (``RingWake.free_convection_steps``), fixed at the first release;
    ``ring_core_m`` (None = wake core) is the core of the ring filaments away from the
    lifting surfaces (``RingWake.core_m``), fixed at the first release;
    ``ring_bound_core_m`` (None = unchanged) is the core of the bound system of the lifting
    surfaces where it induces at the ring nodes (``RingWake.bound_core_m``), fixed at the first release;
    ``ring_surface_overlap`` (None = unchanged) refines the ring sheet where the rings induce at lifting-surface points
    (``RingWake.surface_overlap``), fixed at the first release; ``ring_sheet_drift`` (False = unchanged) lets the
    frozen rings drift with the sheet's own speed (``RingWake.sheet_drift``), fixed at the first release;
    ``ring_ambient_spacing`` (with ``ring_sheet_drift``) releases Gamma = gamma (V_ambient + gamma/2) dt, V_ambient the
    axial component of the ambient flow relative to the hub, so that the drifting tube carries gamma
    (``RingWake.ambient_spacing``), fixed at the first release.
    """
    if not isinstance(state, WakeState):
        raise ValueError("shed_rotor_rings needs a WakeState")
    omegas = np.asarray(omegas_rad_s, dtype=float).reshape(-1)
    if len(omegas) != len(discs):
        raise ValueError("one angular speed per disc is required")
    if hub_velocities is None:
        hub_velocities = np.zeros((len(discs), 3))
    hub_velocities = np.asarray(hub_velocities, dtype=float).reshape(len(discs), 3)
    rings = state.rings if state.rings is not None else RingWake(surface_core_m=surface_core_m,
                                                                 free_convection_steps=free_convection_steps,
                                                                 core_m=ring_core_m, bound_core_m=ring_bound_core_m,
                                                                 surface_overlap=ring_surface_overlap,
                                                                 sheet_drift=ring_sheet_drift,
                                                                 ambient_spacing=ring_ambient_spacing)
    if not isinstance(rings, RingWake):
        raise ValueError("rotor coupling requires RingWake rings on the state")
    if surface_core_m is not None and rings.surface_core_m != surface_core_m:
        raise ValueError("surface_core_m cannot change during a run")
    if free_convection_steps is not None and rings.free_convection_steps != free_convection_steps:
        raise ValueError("free_convection_steps cannot change during a run")
    if ring_core_m is not None and rings.core_m != ring_core_m:
        raise ValueError("ring_core_m cannot change during a run")
    if ring_bound_core_m is not None and rings.bound_core_m != ring_bound_core_m:
        raise ValueError("ring_bound_core_m cannot change during a run")
    if ring_surface_overlap is not None and rings.surface_overlap != ring_surface_overlap:
        raise ValueError("ring_surface_overlap cannot change during a run")
    if bool(ring_sheet_drift) != rings.sheet_drift:
        raise ValueError("ring_sheet_drift cannot change during a run")
    if bool(ring_ambient_spacing) != rings.ambient_spacing:
        raise ValueError("ring_ambient_spacing cannot change during a run")
    records = []
    for index, (disc, omega) in enumerate(zip(discs, omegas)):
        wind = hub_apparent_wind(state, config, disc, index, inflow, hub_velocities[index])
        loads = disc.loads(omega, wind, config.density_kg_m3)
        ring = disc.shed_ring(omega, wind, config.density_kg_m3, config.dt_s, nodes_per_ring,
                              time_s=state.time_s, phase=phase, turbine_extension=turbine_extension)
        if rings.ambient_spacing:
            ambient = ambient_velocity(inflow, disc.centre_m[None, :], state.time_s)[0] - hub_velocities[index]
            spacing = (disc.axial_freestream(ambient) + 0.5 * ring.sheet_strength) * config.dt_s
            if not spacing > 0.0:
                raise ValueError(f"ambient ring spacing {spacing:.4g} m is not positive at disc {disc.label}: "
                                 "the drifting tube would not recede from the disc")
            ring = replace(ring, circulation=float(ring.sheet_strength * spacing), spacing_m=float(spacing))
        rings = rings.shed(ring, disc_index=index, axis=disc.axis)
        records.append(dict(label=disc.label, time_s=float(state.time_s), omega_rad_s=float(omega),
                            apparent_wind_m_s=wind.tolist(), **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                                for k, v in loads.items()},
                            ring_circulation_m2_s=ring.circulation, sheet_strength_m_s=ring.sheet_strength,
                            ring_spacing_m=ring.spacing_m))
    return replace(state, rings=rings), records


def advance_with_rotors(state, polar, config, inflow, discs, omegas_rad_s, *, nodes_per_ring=16,
                        hub_velocities=None, surface_core_m=None, turbine_extension=None, **advance_kwargs):
    """Shed this step's rings, then advance the wing wake (LL or VSM) with them.

    Each record gains ``ring_nodes_within_surface_core``: the number of ring nodes
    (after convection) within the surface core of any load point, so the records
    show how often the surface regularisation is active.
    """
    seeded, records = shed_rotor_rings(state, config, discs, omegas_rad_s, inflow,
                                       nodes_per_ring=nodes_per_ring, hub_velocities=hub_velocities,
                                       surface_core_m=surface_core_m, turbine_extension=turbine_extension)
    advance = advance_vsm_wake if isinstance(state, VsmWakeState) else advance_wake
    new_state = advance(seeded, polar, config, inflow, **advance_kwargs)
    core = new_state.rings.surface_core_m or config.core_radius_m
    near = new_state.rings.near_surface_count(new_state.wing.cp, core)
    for row in records:
        row["ring_nodes_within_surface_core"] = near
        row["surface_core_m"] = core
    return new_state, records


def rotor_totals(records, discs, about=(0.0, 0.0, 0.0)):
    """Total rotor force and moment about ``about``: hub forces at their hubs plus the axis torques."""
    about = np.asarray(about, dtype=float)
    force = np.zeros(3)
    moment = np.zeros(3)
    for row, disc in zip(records, discs):
        f = np.asarray(row["force_N"], dtype=float)
        force += f
        moment += np.cross(disc.centre_m - about, f) + np.asarray(row["moment_Nm"], dtype=float)
    return force, moment
