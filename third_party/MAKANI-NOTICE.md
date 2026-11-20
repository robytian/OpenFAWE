# Source attribution for the Makani M600 system models (stage 4)

Source: google/makani, commit `1d66923` (Makani Technologies LLC, 2020), Apache-2.0,
reproduced in LICENSE-Makani.txt. This does not license unrelated project materials
or imply endorsement by Makani/Google.

Data file included verbatim:

- cases/rotors/makani/rotor_rev4.json = database/m600/rotor_rev4.json
  (SHA-256 56a334dca21004b6cc9d0b78105779c7384ba38fc8fba5bd6c01d43228e9ed73):
  Rev4 propeller thrust/power coefficient tables, 70 angular rates x 60 freestreams.

- cases/polars/makani/{oref,ref,f8,mp6d,approx_blade_rj8}.json = analysis/aero/hover_model/airfoils/*.json,
  cases/polars/makani/naca0012_high_aoa.json = analysis/aero/airfoils/naca0012_high_aoa.json,
  cases/polars/makani/hover_model_sn02_wind_10_no_wake_model.json = database/m600/hover_model_sn02_wind_10_no_wake_model.json
  (SHA-256 values in cases/polars/makani/README.md): XFOIL section polars, the NACA 0012 full-circle
  table and the public whole-kite hover table used as comparison data (decision 0020).

Source-derived files (transcriptions with the source routine named in the file):

- src/openfawe/aero/polars.py: analysis/aero/airfoil.py Airfoil lookup (clamped bilinear in flap and alpha,
  angle wrap) and _AddPostStallData (0.1 rad linear blend to the NACA 0012 table on a 400-point grid).
- src/openfawe/aero/makani_hover_strip.py: analysis/aero/hover_model/hover_model.py strip model
  (sampling points, local apparent wind, induced-angle iteration, 2D-to-3D correction, cl_weight/fudge,
  tail downwash estimate) and analysis/aero/hover_model/wake_model.py (comparison module, off by default).

- src/openfawe/tether/lumped_line.py: sim/models/tether.cc lumped-mass tether
  (longitudinal springs/damping, three-point curvature bending, cylinder drag);
  parameters from config/m600/tether.py and config/common/sim/tether_sim.py.
- src/openfawe/rotor/table.py: sim/physics/rotor_database.cc coefficient conventions and
  sim/math/util.cc / common/c_math/util.c saturating bilinear interpolation;
  sim/models/actuators/rotor.cc torque cutoff and freestream projection.
- src/openfawe/rotor/makani_corrections.py: sim/models/actuators/rotor.cc
  CalcBlownWingForceMomentPos and config/m600/sim/rotor_sim.py parameters (comparison only).
- cases/rotors/m600_rotors_makani.json: config/m600/rotors.py positions/axes and
  config/prop/propellers.py Rev4 directions, mapped with diag(-1, 1, -1).
- paper1/crosswind/mesh fuselage mesh (display only): see CAMBER-SOURCES.md.

Changes in 2026: Python/NumPy implementations with explicit input validation and honest
failures; the actuator-disc vortex-ring wake, the free-wake coupling and the full-incidence
wing closure are OpenFAWE models, not Makani's; of Makani's empirical corrections the blown-wing
thrust deflection is kept as a switchable comparison module (rotor/makani_corrections.py). The tail
thrust-coefficient tables and a whole-kite aerodynamic table module are not implemented (corrected
2026-10-03; this notice previously said both were kept as modules); the ASWING baseline table is
stored as reference data only.

Added in release 0.2.0 of 2026-11-05 (crosswind flight of paper 1, source anchors in each file):

- src/openfawe/flight/crosswind_loop.py: the playbook construction and the nominal-loop law of the Makani
  flight controller (config/m600/control/crosswind_playbook_utils.py, control/crosswind/*.c,
  control/simple_aero.c, common/c_math/{geometry,util}.c); modifications listed in the file header.
- cases/flight/m600_crosswind_playbook_land.json: config/m600/control/crosswind_playbook.py (land set) and
  the constants of config/m600/control/crosswind.py.
- cases/aircraft/m600_makani_rigid_body.json: config/m600/wing.py (wing serial 06, crosswind).
- cases/ground/makani_gs02_parker_ranch.json: config/base_station and config/common/sim parameters of the
  GS02 ground station (drawing and tether ground point only).

## OpenFAWE Studio 1.0 (the Windows program): further Makani material

Same source (google/makani, commit `1d66923`, Apache-2.0). Modules of the program that transcribe or follow Makani
routines, each with the Makani files named in its own header (the routine and the changes are described there):

- src/openfawe/aero/handover.py: config/m600/sim/aero_sim.py, sim/physics/aero.cc
- src/openfawe/aero/makani_database.py: config/m600/sim/aero_models.py, config/m600/sim/aero_sim.py, config/m600/wing.py, sim/math/util.cc, sim/models/rigid_bodies/wing.cc, sim/physics/aero.cc, sim/physics/aero_frame.cc, sim/physics/delta_coeff_aero_database.cc, sim/physics/dvl_aero_database.cc
- src/openfawe/aero/makani_flap_increments.py: config/m600/sim/aero_sim.py, config/m600/wing.py, database/m600/m600_aswing_baseline.json, sim/models/rigid_bodies/wing.cc, sim/physics/avl_aero_database.cc, system/labels.h
- src/openfawe/aero/platform_sections.py: analysis/aero/hover_model/hover_model.py
- src/openfawe/flight/closed_loop.py: sim/physics/aero.cc
- src/openfawe/flight/crosswind_loop.py: common/c_math/util.c, common/c_math/util.h, config/m600/control/crosswind_playbook_utils.py, control/simple_aero.c
- src/openfawe/flight/makani_full.py: config/base_station/ground_frame.py, control/avionics/avionics_interface.c, control/control_system.c, control/estimator/estimator.c, sim/models/sensors/ground_station_v2_sensors.cc
- src/openfawe/flight/makani_interface.py: config/m600/servos.py, config/m600/sim/servos_sim.py, control/estimator/estimator_types.h, control/ground_frame.c, control/sensor_util.c, sim/models/actuators/servo.cc, sim/models/power_sys.cc
- src/openfawe/flight/residual.py: sim/models/actuators/rotor.cc
- src/openfawe/ground/gs02.py: avionics/common/plc_messages.h, common/c_math/util.h, control/sensor_util.c, control/vessel_frame.c, sim/models/ground_station_util.cc, sim/models/model.cc, sim/physics/reference_frame.cc, sim/state.h
- src/openfawe/ground/iec_wind.py: config/m600/sim/iec_sim.py, sim/models/environment.cc, sim/models/model.cc, sim/physics/wind.cc, sim/physics/wind_frame.c
- src/openfawe/ground/makani_wind.py: config/common/sim/phys_sim.py, sim/physics/wind.cc
- src/openfawe/ground/perch.py: common/c_math/vec2.c, config/base_station/sim/gs02_sim.py, config/m600/sim/contact_sim.py, sim/math/util.cc, sim/models/ground_station_util.cc
- src/openfawe/ground/turbulent_wind.py: sim/models/environment.cc, sim/models/full_system.cc, sim/physics/wind.cc, sim/physics/wind_frame.c
- src/openfawe/ground/turnkey.py: config/base_station/ground_frame.py, config/common/sim/joystick_sim.py, config/m600/sim/rotor_sim.py, control/avionics/avionics_interface.c, control/control_output.c, sim/kite_sim.cc, sim/models/actuators/rotor.cc
- src/openfawe/ground/turnkey_scoring.py: analysis/crosswind_batch_sims/batch_sim_params.py, lib/python/batch_sim/scoring_functions/__init__.py
- src/openfawe/ground/wind_shear.py: config/base_station/wind_sensor.py, config/common/sim/phys_sim.py, sim/models/environment.cc, sim/physics/wind_frame.c
- src/openfawe/power/makani_motors.py: common/c_math/util.c, config/powertrain/motors.py, sim/models/actuators/rotor.h, sim/models/stacked_power_sys.cc, sim/physics/motors.cc, sim/sim_types.h
- src/openfawe/power/makani_power_sys.py: avionics/common/motor_util.c, common/c_math/filter.c, common/c_math/util.c, control/avionics/avionics_interface.c, sim/models/model.cc, sim/models/power_sys.cc, sim/models/stacked_power_sys.cc, sim/sim_types.h
- src/openfawe/tether/bridle.py: sim/models/rigid_bodies/wing.cc
- src/openfawe/tether/variable_line.py: config/common/sim/tether_sim.py, sim/physics/ground_frame.cc

Compiled Makani source:

- third_party/makani-controller-win-nogsl: Makani's M600 flight controller (the C sources listed in its build receipt)
  compiled unchanged into a library, linked against OpenFAWE's own small linear-algebra subset in place of the GNU Scientific Library; the ctypes binding is generated from
  Makani's headers.

Data files taken or derived from Makani's `config/` and `database/` (the source is named in each file):
cases/aircraft/m600_makani_rigid_body*.json, cases/flight/m600_crosswind_playbook_*.json, cases/ground/makani_gs02_*.json,
cases/powertrain/makani_m600_*.json, cases/rotors/makani/, cases/polars/makani/.
