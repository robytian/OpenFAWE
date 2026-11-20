## Appendix A Record keys

[User guide](user-guide.md) · **English** · [中文](record-keys.zh.md)

Each line of `records.jsonl` is one period. These are all the keys of a whole flight with the power system (generated from the records of an actual flight). Frames: [section 8 of the user guide](user-guide.md#8-output-files).

### Time and modes

| Key | Unit | Meaning |
|---|---|---|
| `k` | - | control period index (one period is 10 ms) |
| `t` | s | time at the start of the period |
| `ran` | - | whether the controller ran in this period |
| `flight_mode` | - | the planner's flight mode after this period's step (the mode of the next period); names in `spec.json` `mode_names` |
| `controller_flight_mode` | - | the flight mode the controller ran in during this period |
| `gates` | - | the controller's flight-mode gates, one integer per mode: a bit mask of the conditions that hold up the change into that mode (0: none) |
| `gain_ramp_scale` | - | the controller's hover gain ramp, 0 to 1 (its fall in HoverDescend ends the flight) |
| `non_finite` | - | names of controller outputs that were not finite (empty when all are) |

### The kite

| Key | Unit | Meaning |
|---|---|---|
| `flaps_acting_rad` | rad | acting deflections of the eight control surfaces (A1, A2, A4, A5, A7, A8, elevator, rudder) |
| `flaps_command_rad` | rad | commanded deflections, same order (they act from the next period) |
| `position_cm_m` | m | centre of mass in the plant's world frame (x north, y west, z up) |
| `velocity_cm_m_s` | m/s | velocity of the centre of mass, world frame |
| `rotation` | - | attitude: the 3 x 3 matrix from body axes (x towards the tail, y to starboard, z up) to world axes |
| `angular_velocity_body_rad_s` | rad/s | angular velocity in body axes |
| `Xg` | m | origin of the body axes in Makani's ground frame g (x north, y east, z down) |
| `origin_height_m` | m | height of the body origin above the ground |
| `contacts` | - | whether each of the three perch contactors touches the perch panel |
| `airspeed_m_s` | m/s | airspeed at the body origin (the database's lookup) |
| `alpha_rad` | rad | angle of attack (the database's lookup) |
| `beta_rad` | rad | sideslip angle (the database's lookup) |
| `blend_weight` | - | weight of Makani's high-incidence table in the blend with the low-incidence one (0 to 1) |

### Loads and rotors

| Key | Unit | Meaning |
|---|---|---|
| `rotor_omega_rad_s` | rad/s | speed of each rotor |
| `rotor_command_rad_s` | rad/s | the controller's upper speed limit for each rotor |
| `run_motors` | - | the controller's motor run flag |
| `stop_motors` | - | the controller's motor stop flag |
| `thrust_coeff_used` | - | mean thrust coefficient of the rotors used in this period's lookup |
| `held_force_body_N` | N | total force held over the period (aerodynamics, rotors, blown wing), body axes |
| `held_moment_cm_body_Nm` | N m | its moment about the centre of mass, body axes |
| `aero_force_body_N` | N | aerodynamic force of the database term, body axes |
| `aero_moment_origin_body_Nm` | N m | its moment about the body origin, body axes |
| `rotor_thrust_N` | N | thrust of each rotor |
| `rotor_power_W` | W | aerodynamic shaft power of each rotor (negative while the motor drives it, positive while it generates) |
| `rotor_axial_freestream_m_s` | m/s | axial inflow of each rotor |
| `rotor_table_saturated` | - | whether each rotor's operating point left its table |
| `rotor_force_body_N` | N | sum of the eight rotors' forces, body axes |
| `blown_wing_force_body_N` | N | force of the blown-wing term, body axes (zero when that term is off) |

### Tether

| Key | Unit | Meaning |
|---|---|---|
| `free_length` | m | free tether length at the period's start |
| `start_index` | - | index of the first airborne tether node at the period's start |
| `tension_bridle_N` | N | tether force at the bridle point |
| `tension_anchor_N` | N | tether force at the ground anchor |
| `tether_elevation` | rad | elevation of the tether's first segment seen from the ground station |
| `tether_release` | - | the controller's tether-release command |
| `min_node_height_m` | m | height of the lowest airborne tether node above the ground (empty when none is airborne) |
| `slack_segments` | - | number of them without tension |
| `segments` | - | number of tether segments in the air |
| `tether_nodes_Xg_m` | m | positions of the airborne tether nodes in the frame of `Xg` (only on the periods of the wake and tether frames) |
| `tether_ends_Xg_m` | m | the ground anchor and the bridle point, same frame and periods |
| `start_index_end` | - | the same at the period's end |
| `free_length_end` | m | free tether length at the period's end |

### Ground station

| Key | Unit | Meaning |
|---|---|---|
| `gs_mode` | - | mode of the GS02 ground station (Makani's numbering) |
| `transform_stage` | - | stage of the transform between reel and high tension |
| `gs_mode_request` | - | the mode the controller requests |
| `winch_velocity_cmd` | m/s | the controller's winch velocity command |
| `drum_angle` | rad | winch drum angle (negative when reeled in) |
| `drum_omega` | rad/s | winch drum rate |
| `platform_azi` | rad | azimuth of the ground station's platform |
| `detwist_angle` | rad | angle of the detwist mechanism |
| `levelwind_engaged` | - | whether the level-wind is engaged |
| `proximity` | - | the perch proximity sensor |
| `operator_flags` | - | the six operator inputs (tether released, force high tension, force reel, unpause transform, detwist turn once, joystick release); all false in these flights |
| `payout_m` | m | the controller's estimate of the paid-out tether |
| `joystick_throttle` | - | the joystick throttle the program gives the controller (0 to 1) |

### Controller telemetry

| Key | Unit | Meaning |
|---|---|---|
| `tension_cmd_N` | N | the hover tension command |
| `wing_pos_g_cmd` | m | the hover position command in g |
| `thrust_moment` | N, N m | commanded total thrust and the three moments |
| `thrust_moment_avail` | N, N m | the same after the rotors' limits |
| `est_airspeed_m_s` | m/s | the controller's estimate of the airspeed (from the true state) |
| `est_alpha_rad` | rad | its estimate of the angle of attack |
| `est_tension_N` | N | its estimate of the tether tension |

### Power system

| Key | Unit | Meaning |
|---|---|---|
| `power_state_start` | - | the power system's 50 states at the period's start |
| `power_block_voltage_V` | V | voltage of each of the four stacked blocks |
| `power_wing_voltage_V` | V | voltage at the wing at the period's start |
| `power_ground_voltage_V` | V | voltage at the ground |
| `power_tether_current_A` | A | current in the tether |
| `power_filtered_tether_current_A` | A | the filtered tether current of the power system's state |
| `power_motor_torque_cmd_Nm` | N m | torque command of each motor |
| `power_torque_upper_limit_Nm` | N m | upper torque limit of each motor |
| `power_torque_lower_limit_Nm` | N m | lower torque limit of each motor |
| `power_motor_constraint` | - | which limit constrains each motor (Makani's code) |
| `power_speed_controller_state` | - | state of each motor's speed controller (Makani's code) |
| `power_block_power_W` | W | power of each block |
| `power_motor_current_A` | A | current of each motor |
| `power_speed_cmd_upper_rad_s` | rad/s | upper speed command in effect |
| `power_speed_cmd_lower_rad_s` | rad/s | lower speed command in effect |
| `power_rate_limited_upper_rad_s` | rad/s | upper speed command after the rate limit |
| `power_rate_limited_lower_rad_s` | rad/s | lower speed command after the rate limit |
| `power_torque_cmd_Nm` | N m | the controller's torque commands in effect |
| `power_message_for_rate` | - | the controller message the speeds of this period come from |
| `power_message_for_torque` | - | the controller message the torques come from |
| `power_freestream_m_s` | m/s | axial inflow of each rotor used by the power system |
| `power_aero_torque_mean_Nm` | N m | period-mean aerodynamic torque of each rotor |
| `power_ground_energy_J` | J | electrical energy delivered to the ground in the period (positive: generated) |
| `power_tether_loss_J` | J | resistive loss in the tether's conductors |
| `power_storage_integral_J` | J | integral of the stored-energy rate over the period |
| `power_stored_energy_start_J` | J | energy stored in the rotors and capacitors at the period's start |
| `power_stored_energy_end_J` | J | the same at the period's end |
| `power_electrical_J` | J | electrical energy of each motor |
| `power_mechanical_J` | J | mechanical energy of each motor |
| `power_aero_applied_J` | J | aerodynamic energy at each rotor shaft |
| `power_resistive_J` | J | motor resistive loss, summed over the motors |
| `power_speed_loss_J` | J | speed-dependent motor loss, summed |
| `power_hysteresis_J` | J | hysteresis loss, summed |
| `power_controller_J` | J | motor controller loss, summed |
| `power_conduction_J` | J | conduction loss, summed |
| `power_peak_current_sq_A2s` | A² s | integral of the squared motor currents, summed |
| `power_e3a_worst` | - | largest relative residual of the energy balance inside the period's integration |
| `power_stiff_stages` | - | number of stiff stages in the integration |
| `power_low_voltage_stages` | - | number of stages with a low block voltage |
| `power_min_block_voltage_V` | V | lowest block voltage in the period |
| `power_rerun` | - | whether the period's integration was repeated with a smaller sub-step |
| `power_sub_step_s` | s | integration sub-step of the power system |
| `power_low_current_stages` | - | number of stages with a low current |
| `power_saturated_stages` | - | number of stages with a saturated motor |
| `power_first_attempt_stiff_stages` | - | stiff stages of the first attempt (before a rerun) |
| `power_first_attempt_low_current_stages` | - | low-current stages of the first attempt |
| `power_first_attempt_saturated_stages` | - | saturated stages of the first attempt |
| `power_resistive_each_J` | J | resistive loss of each motor |
| `power_speed_loss_each_J` | J | speed-dependent loss of each motor |
| `power_hysteresis_each_J` | J | hysteresis loss of each motor |
| `power_controller_each_J` | J | controller loss of each motor |
| `power_conduction_each_J` | J | conduction loss of each motor |
| `power_peak_current_sq_each_A2s` | A² s | the same for each motor |
| `power_stored_rotor_start_J` | J | its rotor part, start |
| `power_stored_capacitor_start_J` | J | its capacitor part, start |
| `power_stored_rotor_end_J` | J | its rotor part, end |
| `power_stored_capacitor_end_J` | J | its capacitor part, end |
| `power_cw_loop_angle` | rad | the controller's crosswind loop angle |
| `power_state_end` | - | the same at the period's end |
| `power_controller_lower_rad_s` | rad/s | the controller's lower speed limit of this period |
| `power_controller_torque_Nm` | N m | the controller's torque output of this period |
| `power_message_pushed` | - | the message this period passes on |

### Derived by the studio

| Key | Unit | Meaning |
|---|---|---|
| `studio_ground_power_kW` | kW | electrical power at the ground in this period (derived for the plots) |
| `studio_motor_power_kW` | kW | electrical power of the eight motors (derived) |
| `studio_ground_energy_MJ` | MJ | ground energy summed since the start (derived) |

