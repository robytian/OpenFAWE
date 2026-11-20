## 附录 A 记录里的键

[帮助文档](user-guide.zh.md) · [English](record-keys.md) · **中文**

`records.jsonl` 每行一个周期，下面是带电力系统的整次飞行里的全部键（由一次实际飞行的记录生成）。坐标系见[帮助文档第 8 节](user-guide.zh.md#8-输出文件)。

### 时间与模式

| 键 | 单位 | 含义 |
|---|---|---|
| `k` | - | 控制周期序号（一个周期 10 ms） |
| `t` | s | 这个周期开始的时刻 |
| `ran` | - | 这个周期控制器是否运行 |
| `flight_mode` | - | 这一步之后规划器的飞行模式（即下一周期的模式）；名称见 `spec.json` 的 `mode_names` |
| `controller_flight_mode` | - | 这个周期里控制器所在的飞行模式 |
| `gates` | - | 控制器的飞行模式闸门，每个模式一个整数：挡住进入该模式的条件的位掩码（0 表示没有） |
| `gain_ramp_scale` | - | 控制器悬停增益的爬升系数，0 到 1（它在 HoverDescend 里降下来，飞行就结束） |
| `non_finite` | - | 控制器输出中不是有限值的项名（都正常时为空） |

### 风筝

| 键 | 单位 | 含义 |
|---|---|---|
| `flaps_acting_rad` | rad | 八个舵面的实际偏角（A1、A2、A4、A5、A7、A8、升降舵、方向舵） |
| `flaps_command_rad` | rad | 指令偏角，同上顺序（下一周期起作用） |
| `position_cm_m` | m | 质心在飞行模型的世界坐标系里的位置（x 向北，y 向西，z 向上） |
| `velocity_cm_m_s` | m/s | 质心速度，世界坐标系 |
| `rotation` | - | 姿态：机体轴（x 向尾，y 向右翼，z 向上）到世界轴的 3 × 3 矩阵 |
| `angular_velocity_body_rad_s` | rad/s | 机体轴下的角速度 |
| `Xg` | m | 机体轴原点在 Makani 地面坐标系 g 里的位置（x 向北，y 向东，z 向下） |
| `origin_height_m` | m | 机体原点离地高度 |
| `contacts` | - | 三个栖架触点各自是否碰到栖架面板 |
| `airspeed_m_s` | m/s | 机体原点处的空速（查表所用） |
| `alpha_rad` | rad | 迎角（查表所用） |
| `beta_rad` | rad | 侧滑角（查表所用） |
| `blend_weight` | - | Makani 大迎角表与小迎角表混合时大迎角表的权重（0 到 1） |

### 载荷与旋翼

| 键 | 单位 | 含义 |
|---|---|---|
| `rotor_omega_rad_s` | rad/s | 每个旋翼的转速 |
| `rotor_command_rad_s` | rad/s | 控制器给每个旋翼的转速上限 |
| `run_motors` | - | 控制器的电机运行标志 |
| `stop_motors` | - | 控制器的电机停止标志 |
| `thrust_coeff_used` | - | 这个周期查表所用的旋翼平均推力系数 |
| `held_force_body_N` | N | 这个周期里保持不变的合力（气动、旋翼、吹翼），机体轴 |
| `held_moment_cm_body_Nm` | N m | 它对质心的力矩，机体轴 |
| `aero_force_body_N` | N | 气动数据库项的气动力，机体轴 |
| `aero_moment_origin_body_Nm` | N m | 它对机体原点的力矩，机体轴 |
| `rotor_thrust_N` | N | 每个旋翼的推力 |
| `rotor_power_W` | W | 每个旋翼轴上的气动功率（电机驱动时为负，发电时为正） |
| `rotor_axial_freestream_m_s` | m/s | 每个旋翼的轴向来流 |
| `rotor_table_saturated` | - | 每个旋翼的工作点是否超出表的范围 |
| `rotor_force_body_N` | N | 八个旋翼力之和，机体轴 |
| `blown_wing_force_body_N` | N | 吹翼经验项的力，机体轴（关闭时为零） |

### 系绳

| 键 | 单位 | 含义 |
|---|---|---|
| `free_length` | m | 周期开始时系绳的自由长度 |
| `start_index` | - | 周期开始时第一个在空中的系绳节点序号 |
| `tension_bridle_N` | N | 吊索点处的系绳力 |
| `tension_anchor_N` | N | 地面锚点处的系绳力 |
| `tether_elevation` | rad | 从地面站看，系绳第一段的仰角 |
| `tether_release` | - | 控制器的系绳释放指令 |
| `min_node_height_m` | m | 最低的空中系绳节点离地高度（没有空中节点时为空） |
| `slack_segments` | - | 其中不受拉的段数 |
| `segments` | - | 空中的系绳段数 |
| `tether_nodes_Xg_m` | m | 空中系绳节点在 `Xg` 的坐标系里的位置（只在存帧的周期） |
| `tether_ends_Xg_m` | m | 地面锚点与吊索点，同上坐标系与周期 |
| `start_index_end` | - | 周期结束时的同一量 |
| `free_length_end` | m | 周期结束时系绳的自由长度 |

### 地面站

| 键 | 单位 | 含义 |
|---|---|---|
| `gs_mode` | - | GS02 地面站的模式（Makani 的编号） |
| `transform_stage` | - | 卷绕与高张力之间变换的阶段 |
| `gs_mode_request` | - | 控制器请求的模式 |
| `winch_velocity_cmd` | m/s | 控制器的绞车速度指令 |
| `drum_angle` | rad | 绞车卷筒转角（收绳时为负） |
| `drum_omega` | rad/s | 卷筒角速度 |
| `platform_azi` | rad | 地面站平台的方位角 |
| `detwist_angle` | rad | 解扭机构的转角 |
| `levelwind_engaged` | - | 排绳器是否啮合 |
| `proximity` | - | 栖架接近传感器 |
| `operator_flags` | - | 六个操作员输入（已释放系绳、强制高张力、强制卷绕、继续变换、解扭转一圈、摇杆释放）；这些飞行里都为假 |
| `payout_m` | m | 控制器估计的放出系绳长度 |
| `joystick_throttle` | - | 程序交给控制器的摇杆油门（0 到 1） |

### 控制器遥测

| 键 | 单位 | 含义 |
|---|---|---|
| `tension_cmd_N` | N | 悬停的张力指令 |
| `wing_pos_g_cmd` | m | 悬停位置指令，g 系 |
| `thrust_moment` | N, N m | 指令的总推力与三个力矩 |
| `thrust_moment_avail` | N, N m | 考虑旋翼限制之后的同一量 |
| `est_airspeed_m_s` | m/s | 控制器估计的空速（由真实状态给出） |
| `est_alpha_rad` | rad | 它估计的迎角 |
| `est_tension_N` | N | 它估计的系绳张力 |

### 电力系统

| 键 | 单位 | 含义 |
|---|---|---|
| `power_state_start` | - | 周期开始时电力系统的 50 个状态 |
| `power_block_voltage_V` | V | 四个串联块各自的电压 |
| `power_wing_voltage_V` | V | 周期开始时机上的电压 |
| `power_ground_voltage_V` | V | 地面端电压 |
| `power_tether_current_A` | A | 系绳电流 |
| `power_filtered_tether_current_A` | A | 电力系统状态里滤波后的系绳电流 |
| `power_motor_torque_cmd_Nm` | N m | 每台电机的转矩指令 |
| `power_torque_upper_limit_Nm` | N m | 每台电机的转矩上限 |
| `power_torque_lower_limit_Nm` | N m | 每台电机的转矩下限 |
| `power_motor_constraint` | - | 每台电机受哪个限制约束（Makani 的编号） |
| `power_speed_controller_state` | - | 每台电机速度控制器的状态（Makani 的编号） |
| `power_block_power_W` | W | 每个块的功率 |
| `power_motor_current_A` | A | 每台电机的电流 |
| `power_speed_cmd_upper_rad_s` | rad/s | 生效的转速指令上限 |
| `power_speed_cmd_lower_rad_s` | rad/s | 生效的转速指令下限 |
| `power_rate_limited_upper_rad_s` | rad/s | 限速之后的转速指令上限 |
| `power_rate_limited_lower_rad_s` | rad/s | 限速之后的转速指令下限 |
| `power_torque_cmd_Nm` | N m | 生效的控制器转矩指令 |
| `power_message_for_rate` | - | 这个周期的转速来自哪一条控制器消息 |
| `power_message_for_torque` | - | 转矩来自哪一条控制器消息 |
| `power_freestream_m_s` | m/s | 电力系统所用每个旋翼的轴向来流 |
| `power_aero_torque_mean_Nm` | N m | 每个旋翼周期平均的气动转矩 |
| `power_ground_energy_J` | J | 这个周期送到地面的电能（正为发电） |
| `power_tether_loss_J` | J | 系绳导体的电阻损耗 |
| `power_storage_integral_J` | J | 储能变化率在周期上的积分 |
| `power_stored_energy_start_J` | J | 周期开始时旋翼与电容储存的能量 |
| `power_stored_energy_end_J` | J | 周期结束时的同一量 |
| `power_electrical_J` | J | 每台电机的电能 |
| `power_mechanical_J` | J | 每台电机的机械能 |
| `power_aero_applied_J` | J | 每个旋翼轴上的气动能量 |
| `power_resistive_J` | J | 电机电阻损耗，各电机之和 |
| `power_speed_loss_J` | J | 与转速有关的电机损耗，之和 |
| `power_hysteresis_J` | J | 磁滞损耗，之和 |
| `power_controller_J` | J | 电机控制器损耗，之和 |
| `power_conduction_J` | J | 导通损耗，之和 |
| `power_peak_current_sq_A2s` | A² s | 电机电流平方的积分，之和 |
| `power_e3a_worst` | - | 周期内积分时能量平衡的最大相对残差 |
| `power_stiff_stages` | - | 积分里刚性阶段的个数 |
| `power_low_voltage_stages` | - | 块电压过低的阶段个数 |
| `power_min_block_voltage_V` | V | 周期内最低的块电压 |
| `power_rerun` | - | 这个周期是否用更小子步重算过 |
| `power_sub_step_s` | s | 电力系统积分的子步长 |
| `power_low_current_stages` | - | 电流过低的阶段个数 |
| `power_saturated_stages` | - | 电机饱和的阶段个数 |
| `power_first_attempt_stiff_stages` | - | 第一次尝试（重算之前）的刚性阶段个数 |
| `power_first_attempt_low_current_stages` | - | 第一次尝试的低电流阶段个数 |
| `power_first_attempt_saturated_stages` | - | 第一次尝试的饱和阶段个数 |
| `power_resistive_each_J` | J | 每台电机的电阻损耗 |
| `power_speed_loss_each_J` | J | 每台电机与转速有关的损耗 |
| `power_hysteresis_each_J` | J | 每台电机的磁滞损耗 |
| `power_controller_each_J` | J | 每台电机控制器的损耗 |
| `power_conduction_each_J` | J | 每台电机的导通损耗 |
| `power_peak_current_sq_each_A2s` | A² s | 每台电机的同一量 |
| `power_stored_rotor_start_J` | J | 其中旋翼的部分，开始 |
| `power_stored_capacitor_start_J` | J | 其中电容的部分，开始 |
| `power_stored_rotor_end_J` | J | 其中旋翼的部分，结束 |
| `power_stored_capacitor_end_J` | J | 其中电容的部分，结束 |
| `power_cw_loop_angle` | rad | 控制器的交叉风圈角 |
| `power_state_end` | - | 周期结束时的同一量 |
| `power_controller_lower_rad_s` | rad/s | 这个周期控制器给的转速下限 |
| `power_controller_torque_Nm` | N m | 这个周期控制器给的转矩 |
| `power_message_pushed` | - | 这个周期往后传的消息 |

### Studio 推得的量

| 键 | 单位 | 含义 |
|---|---|---|
| `studio_ground_power_kW` | kW | 这个周期地面端的电功率（为曲线推得） |
| `studio_motor_power_kW` | kW | 八台电机的电功率（推得） |
| `studio_ground_energy_MJ` | MJ | 从开始累计的地面电能（推得） |

