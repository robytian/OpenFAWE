# SPDX-License-Identifier: Apache-2.0
# The order and the arguments of the estimator calls follow google/makani commit 1d66923 (Apache-2.0):
# control/estimator/estimator.c lines 57-82 (initialisation) and 105-253 (the ground-side part of EstimatorStep);
# control/control_system.c lines 275-379 (the order of one control step). Modifications (2026): Python driver of the
# compiled, unmodified Makani functions; perfect sensors in place of the avionics messages; the kite's navigation
# estimate replaced by the plant's truth (as in flight.closed_loop).
"""The full Makani controller (plan 3.7): the compiled library of all flight modes, driven by the plant's truth.

Opt-in; nothing else imports this module and the crosswind closed loop (``flight.closed_loop``) does not use it.
The library is the one ``makani_lib/build_full_controller.py`` builds (full-cycle contract, part 1).

One control step (``FullController.step``) is Makani's ``ControlSystemStep`` without the avionics conversion and
without ``EstimatorStep``: the shim function ``OpenfaweFullControlStep`` runs ``RunController`` and ``PlannerStep``
on a ``StateEstimate`` given from here. That estimate is filled in two parts:

* the kite: the plant's truth in Makani's definitions, the same fields and conventions as the crosswind closed loop
  (``ClosedLoopController.estimate``); filled by the caller (full-cycle contract, part 2);
* the ground side (``ground_estimate``): Makani's own estimator functions, called in ``EstimatorStep``'s order with
  perfect sensors (``GroundSensors``) and no faults - encoders, joystick, ground station, perch azimuth, vessel,
  winch, tether ground angles, tether anchor. The ground-side conventions (pay-out reset while perched, the source of
  the tether elevation in each ground-station mode and transform stage, the anchor point) are therefore Makani's.

Order within one controller period: ``ground_estimate`` (with the kite's fields already filled) and then ``step``, as in
``ControlSystemStep``. With ``follow_init_state`` (the default) the ground side is initialised and stepped exactly when
Makani would do it: ``EstimatorInit`` at the first loop, ``EstimatorStep`` only while the control system runs
(control_system.c lines 308-318); ``follow_init_state=False`` steps it on every call (unit tests of the estimator
functions alone).

The ground estimator of the ground station itself (``GroundEstimateMessage``, made by Makani's separate ground process
from the platform's GPS and IMU) is replaced by its perfect value for an onshore station: the platform frame's origin is
the ground frame's origin (the top of the slewing bearing, ``config/base_station/ground_frame.py``), at rest, turned
about the vertical by the platform azimuth (``CalcDcmVesselToPlatform`` with the vessel frame equal to the ground
frame, as in Makani's ``estimator_test.cc``).
"""

from dataclasses import dataclass
import ctypes
import importlib.util
from pathlib import Path

import numpy as np


def load_full_api(build):
    """The ctypes binding of the full library in ``build/lib``."""
    path = Path(build)/"lib"/"makani_full_controller_api.py"
    spec = importlib.util.spec_from_file_location("makani_full_controller_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class GroundSensors:
    """Perfect ground-station sensors (GS02), in Makani's units and frames.

    ``winch_position_m`` is Makani's winch position (drum radius times drum angle, negative when reeled in);
    ``levelwind_elevation_rad``, ``gsg_azimuth_rad`` and ``gsg_elevation_rad`` are None when the sensor gives no value
    (reported to the estimator as faulted, as an absent sensor would be).
    """
    winch_position_m: float
    platform_azimuth_rad: float
    gs_mode: int
    platform_azimuth_rate_rad_s: float = 0.0
    transform_stage: int = 0
    proximity: bool = False
    detwist_angle_rad: float = 0.0
    levelwind_elevation_rad: float = None
    gsg_azimuth_rad: float = None
    gsg_elevation_rad: float = None
    proximity_valid: bool = True    # False: the winch estimator sees a faulted proximity input (a test hook; in Makani the
                                    # same ground-station subsystem mask would also invalidate the mode and stage)


def gs02_sensors(api, system, *, drum_angle_rad, platform_azimuth_rad, gs_mode, transform_stage=0, proximity=False,
                 detwist_angle_rad=0.0, levelwind_elevation_rad=0.0, gsg_yoke_rad=0.0, gsg_termination_rad=0.0,
                 platform_azimuth_rate_rad_s=0.0):
    """Perfect GS02 sensors from the plant's truth, through Makani's own sensor chain without its noise and rounding.

    The chain is that of the simulator's GS02 sensors (``sim/models/sensors/ground_station_v2_sensors.cc``: angles
    wrapped to [-pi, pi], lines 47-49, 92-93, 101-102, 113-114) followed by the controller's avionics conversion
    (``control/avionics/avionics_interface.c``: ``winch_pos = ApplyCal(drum_angle, winch.position_cal)`` line 345,
    ``detwist_pos = Wrap(detwist_angle, -PI, PI)`` line 346, ``gsg azi/ele = ApplyCal(axis, gsg.azi_cal/ele_cal)``
    lines 326-327), with the library's own ``Wrap`` and ``ApplyCal``. Left out: the single-precision cast of the
    message fields (``static_cast<float>``), so that the values stay the plant's to double precision. Not done here:
    the simulator's gating of the readings by whether the tether is engaged on the levelwind (GSG angles read zero
    while it is engaged, the levelwind elevation zero while it is not; ground_station_v2_sensors.cc lines 51-61): the
    caller (the plant's ground station) gives the values the sensors would read.
    """
    pi = float(np.pi)
    wrap = lambda x: float(api.Wrap(float(x), -pi, pi))          # noqa: E731
    return GroundSensors(
        winch_position_m=float(api.ApplyCal(float(drum_angle_rad), ctypes.byref(system.winch.position_cal))),
        platform_azimuth_rad=wrap(platform_azimuth_rad), gs_mode=int(gs_mode), transform_stage=int(transform_stage),
        proximity=bool(proximity), detwist_angle_rad=wrap(wrap(detwist_angle_rad)),
        levelwind_elevation_rad=None if levelwind_elevation_rad is None else wrap(levelwind_elevation_rad),
        gsg_azimuth_rad=None if gsg_yoke_rad is None else float(api.ApplyCal(wrap(gsg_yoke_rad), ctypes.byref(system.gsg.azi_cal))),
        gsg_elevation_rad=None if gsg_termination_rad is None else float(api.ApplyCal(wrap(gsg_termination_rad),
                                                                                      ctypes.byref(system.gsg.ele_cal))),
        platform_azimuth_rate_rad_s=float(platform_azimuth_rate_rad_s))


@dataclass
class GroundOperator:
    """The operator's and ground station's discrete inputs that ``EstimatorStep`` copies into the estimate
    (estimator.c lines 128-163); the defaults are Makani's 'no command'."""
    tether_released: bool = False
    force_high_tension: bool = False
    force_reel: bool = False
    gs_unpause_transform: bool = False
    force_detwist_turn_once: bool = False
    stacking_state: int = None          # None: kStackingStateNormal


@dataclass
class Joystick:
    """The pilot's joystick, in Makani's ``JoystickData`` terms (switch: one of the kJoystickSwitchPosition values)."""
    switch_position: int
    throttle: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    release: bool = False


class FullController:
    """Makani's full controller in the library, with the ground-side estimator functions."""

    def __init__(self, api, controller_label=None, *, follow_init_state=True):
        self.api = api
        self.system = api.GetSystemParams().contents
        self.control = api.GetControlParams().contents
        self.label = api.kControllerA if controller_label is None else int(controller_label)
        if self.label != api.kControllerA:
            # With another label the planner follows controller A's synchronisation data (ArbitrateFlightMode,
            # control_planner.c lines 481-520), which nothing here provides.
            raise ValueError("only controller A is supported")
        self.follow_init_state = bool(follow_init_state)
        self.state = api.ControlState()
        api.ControlSystemInit(ctypes.byref(self.control), ctypes.byref(self.state))
        self.sync_in = (api.ControlSyncData*api.kNumControllers)()
        self.output = api.ControlOutput()
        self._init_ground()

    def _init_ground(self):
        """The ground-side estimator states, initialised as EstimatorInit does (estimator.c lines 57-82)."""
        api = self.api
        self.est = api.EstimatorState()
        api.EstimatorEncodersInit(ctypes.byref(self.est.encoders))
        api.EstimatorGroundStationInit(ctypes.byref(self.est.ground_station))
        api.EstimatorJoystickInit(ctypes.byref(self.est.joystick))
        api.EstimatorPerchAziInit(ctypes.byref(self.est.perch_azi))
        api.EstimatorTetherAnchorPointInit(ctypes.byref(self.est.tether_anchor))
        api.EstimatorTetherGroundAnglesInit(ctypes.byref(self.est.tether_ground_angles))
        api.EstimatorVesselInit(ctypes.byref(self.est.vessel))
        api.EstimatorWinchInit(self.system.flight_plan, self.system.gs_model, self.system.winch.r_drum,
                               ctypes.byref(self.est.winch))
        self.no_fault = api.FaultMask()
        self.ground_estimator_faults = (api.FaultMask*api.kNumFaultDetectionGroundStationEstimatorSignals)()
        self.est_time = 0.0                           # EstimatorState.time: counts up to t_initialize (estimator.c)
        self.tether_release_latched = False
        self.gs_unpause_transform_count = 0
        self.ts = float(self.system.ts)

    # ----- the flight status ----------------------------------------------------------------------------------------
    @property
    def flight_status(self):
        return self.state.planner.flight_status

    @property
    def flight_mode(self):
        return int(self.state.planner.flight_status.flight_mode)

    def gates(self):
        """The controller's flight-mode gate words (``ControlTelemetry.flight_mode_gates``), one per mode."""
        telemetry = self.api.GetControlTelemetry().contents
        return [int(v) for v in telemetry.flight_mode_gates]

    # ----- the ground side, through Makani's estimator functions --------------------------------------------------
    def ground_estimate(self, est, sensors, joystick, *, operator=None, ground_estimate_message=None):
        """Fill the ground-side fields of ``est`` (a ``StateEstimate``) from perfect sensors.

        The calls and their order are those of ``EstimatorStep`` (estimator.c lines 105-253) for the ground side;
        ``initializing`` is true while the estimator time is below ``t_initialize`` (line 114), and that time advances
        by one controller period per step until then (line 237). The operator's discrete inputs (``GroundOperator``)
        are copied as in lines 128-163 and 175. The kite's own fields are not touched. Returns (est, encoders,
        ground_station), or (est, None, None) when Makani would not run the estimator this period (see the module text).
        """
        api, ref = self.api, ctypes.byref
        if self.follow_init_state:
            init = int(self.state.init_state)
            if not api.IsControlSystemRunning(init):
                return est, None, None
            if init == api.kInitializationStateFirstLoop:
                self._init_ground()
        state = self.est
        params = self.control.estimator
        operator = GroundOperator() if operator is None else operator
        initializing = self.est_time < params.t_initialize
        # Encoders: both redundant sensors give the same perfect value; an absent value is reported as faulted.
        gsg = (api.GsgData*api.kNumDrums)()
        for k in range(api.kNumDrums):
            gsg[k].azi = 0.0 if sensors.gsg_azimuth_rad is None else float(sensors.gsg_azimuth_rad)
            gsg[k].ele = 0.0 if sensors.gsg_elevation_rad is None else float(sensors.gsg_elevation_rad)
        gsg_faults = (api.FaultMask*api.kNumFaultDetectionGsgSignals)()
        if sensors.gsg_azimuth_rad is None:
            gsg_faults[api.kFaultDetectionGsgSignalAzi].code = 1 << api.kFaultTypeNoUpdate
        if sensors.gsg_elevation_rad is None:
            gsg_faults[api.kFaultDetectionGsgSignalEle].code = 1 << api.kFaultTypeNoUpdate
        levelwind = (ctypes.c_double*api.kNumPlatforms)(*[0.0 if sensors.levelwind_elevation_rad is None
                                                          else float(sensors.levelwind_elevation_rad)]*api.kNumPlatforms)
        levelwind_fault = api.FaultMask()
        if sensors.levelwind_elevation_rad is None:
            levelwind_fault.code = 1 << api.kFaultTypeNoUpdate
        perch = (ctypes.c_double*api.kNumPlatforms)(*[float(sensors.platform_azimuth_rad)]*api.kNumPlatforms)
        encoders = api.EncodersEstimate()
        api.EstimatorEncodersStep(gsg, gsg_faults, gsg_faults, levelwind, ref(levelwind_fault), ref(levelwind_fault),
                                  perch, ref(self.no_fault), ref(self.no_fault), ref(state.encoders), ref(encoders))
        # Stacking state and joystick (estimator.c lines 132-136).
        est.stacking_state = api.kStackingStateNormal if operator.stacking_state is None else int(operator.stacking_state)
        data = api.JoystickData()
        data.switch_position = int(joystick.switch_position)
        data.throttle, data.roll, data.pitch, data.yaw = (float(joystick.throttle), float(joystick.roll),
                                                         float(joystick.pitch), float(joystick.yaw))
        data.release = bool(joystick.release)
        api.EstimatorJoystickStep(ref(data), ref(self.no_fault), ref(params.joystick), ref(state.joystick), ref(est.joystick))
        # No experiment (EstimatorExperimentStep with kExperimentTypeNoTest, lines 138-140).
        est.experiment.active_type = est.experiment.staged_type = api.kExperimentTypeNoTest
        # The operator's discrete inputs (lines 148-160).
        self.tether_release_latched |= bool(operator.tether_released)
        est.tether_released = self.tether_release_latched
        est.force_high_tension, est.force_reel = bool(operator.force_high_tension), bool(operator.force_reel)
        if operator.gs_unpause_transform:
            self.gs_unpause_transform_count = 10                         # GS_UNPAUSE_TRANSFORM_CYCLES, line 55
        elif self.gs_unpause_transform_count > 0:
            self.gs_unpause_transform_count -= 1
        est.gs_unpause_transform = self.gs_unpause_transform_count > 0
        # Ground station, perch azimuth, vessel (estimator.c lines 163-186).
        gs_gps = api.GsGpsData()
        ground_station = api.GroundStationEstimate()
        api.EstimatorGroundStationStep(ref(gs_gps), ref(self.no_fault), int(sensors.gs_mode), ref(self.no_fault),
                                       float(sensors.detwist_angle_rad), ref(self.no_fault), int(sensors.transform_stage),
                                       ref(self.system.ground_frame), ref(params.ground_station), ref(state.ground_station),
                                       ref(ground_station))
        est.gs_mode = ground_station.mode
        est.gs_transform_stage = ground_station.transform_stage
        est.force_detwist_turn_once = bool(operator.force_detwist_turn_once)       # line 175
        api.EstimatorPerchAziStep(encoders.perch_azi, bool(encoders.perch_azi_valid), ref(params.perch_azi),
                                  ref(state.perch_azi), ref(est.perch_azi))
        message = self.onshore_ground_estimate(sensors) if ground_estimate_message is None else ground_estimate_message
        api.EstimatorVesselStep(ref(message), ref(est.perch_azi), self.ground_estimator_faults, ref(state.vessel), ref(est.vessel))
        # Winch (estimator.c lines 188-192): "perched" is the flight mode the planner is in.
        proximity_fault = api.FaultMask()
        if not sensors.proximity_valid:
            proximity_fault.code = 1 << api.kFaultTypeNoUpdate
        api.EstimatorWinchStep(float(sensors.winch_position_m), ref(self.no_fault), bool(sensors.proximity), ref(proximity_fault),
                               self.flight_mode == api.kFlightModePerched, self.system.gs_model, ref(self.system.winch),
                               ref(state.winch), ref(est.winch))
        # Tether ground angles and anchor (estimator.c lines 243-252).
        api.EstimatorTetherGroundAnglesStep(initializing, self.system.gs_model, ref(ground_station), ref(est.winch), ref(est.vessel),
                                            ref(self.system.winch), ref(params.tether_ground_angles), ref(encoders),
                                            float(self.system.ground_station.gs02.detwist_elevation),
                                            ref(state.tether_ground_angles), ref(est.tether_ground_angles))
        api.EstimatorTetherAnchorPointStep(initializing, ref(est.winch), ref(est.vessel), float(est.winch.payout),
                                           ref(params.tether_anchor), ref(state.tether_anchor), ref(est.tether_anchor))
        if self.est_time < params.t_initialize:       # estimator.c line 237
            self.est_time += self.ts
        return est, encoders, ground_station

    def onshore_ground_estimate(self, sensors):
        """The perfect ``GroundEstimateMessage`` of an onshore station (see the module text)."""
        api = self.api
        message = api.GroundEstimateMessage()
        message.attitude_valid = message.position_valid = True
        dcm_v2p = api.Mat3()
        api.CalcDcmVesselToPlatform(float(sensors.platform_azimuth_rad), ctypes.byref(dcm_v2p))
        message.dcm_g2p = dcm_v2p                     # dcm_g2v = I, so dcm_g2p = dcm_v2p
        message.pqr.x, message.pqr.y, message.pqr.z = 0.0, 0.0, float(sensors.platform_azimuth_rate_rad_s)
        return message

    # ----- one control step -----------------------------------------------------------------------------------------
    def step(self, est, *, force_hover_accel=False):
        """One controller step on the estimate ``est``; returns (ran, ControlOutput)."""
        ran = self.api.OpenfaweFullControlStep(self.label, ctypes.byref(self.system), ctypes.byref(self.control), ctypes.byref(est),
                                               self.sync_in, bool(force_hover_accel), ctypes.byref(self.state),
                                               ctypes.byref(self.output))
        return bool(ran), self.output
