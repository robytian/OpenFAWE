"""The plain-data description of one studio simulation, and the catalogue of the choices it offers.

A ``Spec`` holds numbers, strings and booleans only, so that it can be written to JSON, read back and sent to the worker
process; ``engine.build`` turns it into OpenFAWE objects. Every choice in the catalogues below says what it runs in the
library and how far it is checked (``status``), so that the interface can show it next to the switch.
"""
from dataclasses import asdict, dataclass, fields

SCHEMA = "openfawe.studio.spec/1"

#: Bundled aircraft cases. Reference area and chord: accel_bench.reference_values("M600") (the M600 values used by the
#: OF-0006 verification); the files are those of cases/qualification.
AIRCRAFT = {
    "m600": dict(label="Makani M600 - wing and tails, thin-airfoil polars",
                 file="cases/qualification/m600_makani_reconstructed.json", area_m2=32.9285, chord_m=1.2831,
                 status="the case of the OF-0006 verification (stages 1-3)"),
    "m600_profile_drag": dict(label="Makani M600 - wing and tails, with profile drag",
                              file="cases/qualification/m600_makani_reconstructed_profile_drag.json", area_m2=32.9285,
                              chord_m=1.2831, status="low-incidence tables of the G5 whole flight (OF-0008)"),
    "m600_xfoil": dict(label="Makani M600 - wing and tails, XFOIL polars (wide incidence)",
                       file="cases/qualification/m600_makani_xfoil_polars.json", area_m2=32.9285, chord_m=1.2831,
                       status="high-incidence partner of the section hand-over"),
    "m600_pylons": dict(label="Makani M600 - wing, tails and four pylons",
                        file="cases/qualification/m600_makani_reconstructed_pylons.json", area_m2=32.9285,
                        chord_m=1.2831, status="pylons as lifting surfaces (OF-0007)"),
}

#: The six aerodynamic methods of openfawe.aero.methods.AeroMethod.
METHODS = {
    "vsm_free_wake": dict(label="VSM + free wake", free_wake=True, status="the main method (decision 0016)"),
    "free_wake": dict(label="Lifting line + free wake", free_wake=True, status="verified (OF-0006)"),
    "vlm_free_wake": dict(label="UVLM (vortex lattice + free wake)", free_wake=True, status="verified (OF-0006)"),
    "vsm": dict(label="VSM, straight wake", free_wake=False, status="verified (OF-0006)"),
    "ll_straight": dict(label="Lifting line, straight wake", free_wake=False, status="verified (OF-0006)"),
    "vlm": dict(label="VLM (vortex lattice), straight wake", free_wake=False, status="verified (OF-0006)"),
}

#: Where the induced velocities are summed (decision 0052 addendum 1). Only the large wake-on-wake sums use the tree or
#: the GPU; the induction at the lifting surfaces is always the exact direct sum (openfawe.aero.wake.kernel_options).
COMPUTE = {
    "cpu_tree": dict(label="CPU - tree code", backend="tree", device=False,
                     status="production setting of OF-0007/0009 (theta 0.2, order 2, leaf 32)"),
    "cpu_direct": dict(label="CPU - exact direct sum", backend="numba", device=False, status="the reference engine"),
    "gpu": dict(label="GPU - direct sum", backend="cuda", device=True,
                status="development only (mixed precision; GPU direct-sum contract)"),
    "cpu_gpu": dict(label="CPU + GPU - tree built on the CPU, evaluated on the GPU", backend="cuda_tree", device=True,
                    status="judged on node 2 (GPU whole-step contract S1, S2)"),
    "numpy": dict(label="CPU - NumPy (slow reference)", backend="numpy", device=False, status="the library default"),
}

MOTIONS = {
    "fixed": dict(label="Fixed aircraft in the wind (wind tunnel)", status="impulsive start, constant attitude"),
    "circle": dict(label="Circular crosswind loop", status="prescribed circle about the wind axis (case A2 of OF-0007)"),
    "free_flight": dict(label="Free flight: Makani controller, whole flight (TurnKey)",
                        status="perch, hover, transition, crosswind loops, landing; ground station, winch, tether, rotors and "
                               "motors computed; Makani aerodynamic database (OF-0008 G5)"),
    "makani_loop": dict(label="Makani crosswind loop (with tether)",
                        status="Makani's nominal loop for the wind speed, section hand-over, lumped tether (OF-0007 A5)"),
}

#: The free flight's aerodynamics (OF-0010 revision 13): the hook of platform_flight.py for the vortex choices.
FLIGHT_AERO = {
    "tables": dict(label="Makani aerodynamic database (tables)",
                   status="the reference plant's tables in every mode (OF-0008 G5; with the power system: plan 3.9, judged)"),
    "vortex": dict(label="Vortex method where it holds (paper 2's flight B)",
                   status="perch and hover on the tables; HoverAccel, TransIn, crosswind, PrepTransOut and HoverTransOut on VSM + "
                          "free wake + rotor rings; blown wing only where the tables act (OF-0010 revision 11)"),
    "vortex_bprime": dict(label="Vortex method with the T3 slipstream and induced hub inflow (paper 2's flight B')",
                          status="B with the rotor slipstream tube T3 and the hub inflow induced by the wing's wake and the other "
                                 "discs' rings (OF-0010 revision 15; B'-S (a) passed on node 2)"),
    "vortex_hover": dict(label="Vortex method in hover too (P5d, a study)",
                         status="every flying mode on the vortex method; the hover loads spike (OF-0010 revision 11 "
                                "addendum 5) - not paper 2's flight"),
}
VORTEX_COMPUTE = ("cpu_tree", "cpu_gpu")      # the hook's two segments: whole+orth+p2sl and whole+orth+gpu+p2sl

GUSTS = {"none": "none", "longitudinal": "along the wind", "vertical": "vertical", "lateral": "lateral"}

TREE = dict(theta=0.2, order=2, leaf_size=32)


@dataclass
class Spec:
    """One simulation. Units: m, s, deg, rad/s, m/s."""
    name: str = "M600 study"
    aircraft: str = "m600"
    method: str = "vsm_free_wake"
    # motion
    motion: str = "fixed"
    alpha_deg: float = 4.0
    beta_deg: float = 0.0
    height_m: float = 150.0             # body origin (fixed) or loop centre (circle) above the ground
    loop_radius_m: float = 120.0
    loop_speed_m_s: float = 50.0
    # inflow
    wind_m_s: float = 20.0              # at the reference height
    shear_exponent: float = 0.0         # power law; 0 = uniform
    shear_reference_height_m: float = 150.0
    gust: str = "none"
    gust_amplitude_m_s: float = 5.0
    gust_start_s: float = 1.0
    gust_duration_s: float = 2.0
    # time
    dt_s: float = 0.02
    steps: int = 200
    # wake
    core_radius_m: float = 0.15
    free_convection_rows: int | None = None
    retained_rows: int | None = None
    # compute
    compute: str = "cpu_tree"
    threads: int = 0                    # 0: all cores
    # rotors
    rotors: bool = False
    rotor_omega_rad_s: float = 120.0
    ring_nodes: int = 16
    ring_surface_core_m: float = 0.3
    retained_ring_steps: int | None = None
    free_convection_ring_steps: int | None = None
    # display and output
    fuselage: bool = True               # drawn; it is not part of the aerodynamics
    tether_drawn: bool = True           # the ground station and, without a computed line, a straight tether drawn
    tether: bool = True                 # the lumped-mass tether computed with the motion (fixed aircraft, Makani loop)
    tether_substeps: int = 10
    # free flight (TurnKey)
    blown_wing: bool = True             # with the tables (the vortex choices set it themselves); E4's setting (OF-0010
                                        # revision 19 addendum 7, ST-03: the default is the judged flight)
    max_periods: int = 6000             # 0: the whole flight (about 144 000 periods of 0.01 s)
    flight_aero: str = "tables"         # OF-0010 revision 13: FLIGHT_AERO
    power_system: bool = True           # plan 3.9's power system (motors, stacked power, tether loss; the energy account)
    keep_rows: int | None = None        # the vortex choices' kept wake rows; empty: 1872 (OF-0010 revision 10)
    makani_build: str = ""              # folder with lib/makani_full_controller_api.py; empty: the bundled one
    frame_every: int = 1                # a 3D frame is stored every that many steps (and the last)
    flight_frame_every: int = 100       # free flight: a frame (pose, tether, rings) every that many periods (ST-07)
    wake_frame_every: int = 200         # free flight: a frame with the vortex method's wake every that many periods

    def to_dict(self):
        return dict(schema=SCHEMA, **asdict(self))

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        schema = data.pop("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported studio spec schema {schema!r}")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown spec fields: {', '.join(unknown)}")
        spec = cls(**data)
        spec.validate()
        return spec

    def validate(self):
        """Raise ValueError with a readable message for a choice the library would refuse later."""
        problems = []
        if self.aircraft not in AIRCRAFT:
            problems.append(f"unknown aircraft {self.aircraft!r}")
        if self.method not in METHODS:
            problems.append(f"unknown method {self.method!r}")
        if self.motion not in MOTIONS:
            problems.append(f"unknown motion {self.motion!r}")
        if self.compute not in COMPUTE:
            problems.append(f"unknown compute choice {self.compute!r}")
        if self.gust not in GUSTS:
            problems.append(f"unknown gust {self.gust!r}")
        if not self.dt_s > 0:
            problems.append("the time step must be positive")
        if int(self.steps) < 1:
            problems.append("at least one step")
        if not self.wind_m_s >= 0 or (self.motion == "fixed" and not self.wind_m_s > 0):
            problems.append("the fixed aircraft needs a positive wind speed")
        if not abs(self.alpha_deg) < 90 or not abs(self.beta_deg) < 90:
            problems.append("incidence and sideslip strictly inside +/-90 deg")
        if self.motion == "circle":
            if not (self.loop_radius_m > 0 and self.loop_speed_m_s > 0):
                problems.append("the loop needs a positive radius and speed")
            if self.shear_exponent and not self.height_m > self.loop_radius_m:
                problems.append("with wind shear the loop must stay above the ground (centre height above the radius)")
        if self.shear_exponent < 0 or (self.shear_exponent and not self.shear_reference_height_m > 0):
            problems.append("the shear exponent is not negative and needs a positive reference height")
        if self.gust != "none" and not self.gust_duration_s > 0:
            problems.append("the gust needs a positive duration")
        if not self.core_radius_m > 0:
            problems.append("the core radius must be positive")
        if self.rotors:
            if not METHODS.get(self.method, {}).get("free_wake") or self.method == "vlm_free_wake":
                problems.append("rotors with ring wakes need 'VSM + free wake' or 'Lifting line + free wake'")
            if int(self.ring_nodes) < 3:
                problems.append("a ring needs at least three nodes")
            if not self.rotor_omega_rad_s > 0:
                problems.append("the rotor speed must be positive")
        if self.motion == "free_flight" and self.rotors is False:
            pass                            # the TurnKey flight always has its eight rotors and motors
        if self.motion == "makani_loop" and (self.method != "vsm_free_wake" or self.aircraft not in ("m600", "m600_profile_drag")):
            problems.append("the Makani loop runs VSM + free wake on the wing-and-tails aircraft (its section hand-over)")
        if self.method in ("vlm", "vlm_free_wake") and (self.free_convection_rows or self.retained_rows):
            problems.append("the lattice methods have no frozen or truncated far wake")
        for name in ("free_convection_rows", "retained_rows", "retained_ring_steps", "free_convection_ring_steps"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                problems.append(f"{name} must be at least 1 or empty")
        if int(self.frame_every) < 1:
            problems.append("frame_every must be at least 1")
        if int(self.flight_frame_every) < 1 or int(self.wake_frame_every) < 1:
            problems.append("the free flight's frame intervals must be at least 1")
        elif int(self.wake_frame_every) % int(self.flight_frame_every):        # review STR-08
            problems.append("the wake frame interval must be a multiple of the frame interval")
        if self.flight_aero not in FLIGHT_AERO:
            problems.append(f"unknown free-flight aerodynamics {self.flight_aero!r}")
        elif self.motion == "free_flight" and self.flight_aero != "tables" and self.compute not in VORTEX_COMPUTE:
            problems.append("the vortex free flight runs on 'CPU - tree code' or 'CPU + GPU'")
        if self.keep_rows is not None and (isinstance(self.keep_rows, bool) or int(self.keep_rows) <= 187):
            problems.append("kept wake rows: above 187, or empty for 1872")
        if problems:
            raise ValueError("; ".join(problems))
        return self
