To design physical experiments validating the SAC-trained vibration control model for the rotating space habitat, focus on replicating key simulation dynamics at a 1:20 scaled-down level. The model minimizes transverse angular velocities (wobble/nutation), energy consumption, jerky actions, and tank imbalances while maintaining stability under disturbances. Physical validation requires a benchtop prototype capturing relative motions, though gravity scaling differs (dynamics faster due to smaller size/mass).

### 1. **Model Objectives and Key Metrics**
- **Objectives**: Suppress vibrations from mass imbalances and external forces using water tank redistribution; maintain balanced tank levels; minimize control effort.
- **Key Metrics**: Total episode reward (penalty-based on wobble, energy, smoothness, imbalance); nutation angle (degrees); center-of-mass (CM) offset magnitude; episode duration; angular velocities (ω_x, ω_y, ω_z).

### 2. **Scaling Factors for Physical Prototype (1:20 Scaling)**
- **Size Scaling**: Simulation cylinder (R=10m, L=20m) scaled to 1:20 (R=0.5m, L=1m) for benchtop feasibility; volume scales 1:8000.
- **Mass Scaling**: Simulation total mass ~26,000kg (structural + water) scaled to ~3-5kg (maintains inertia ratios for dynamics; e.g., I_physical / I_sim ≈ 1:32,500, but relative torques preserved).
- **Rotation Speed**: Maintain similar angular velocities (1-10 rad/s) for comparable dynamics; centrifugal stress negligible at this scale.
- **Time Scaling**: Episode duration (60s sim) kept at 60s physical (dynamics faster, but match for validation); control loop at 10-50Hz.
- **Force Scaling**: Control forces (tank water movement) scaled proportionally; disturbances (e.g., 50-200kg sim imbalance) to 6-25g physical (but use 20-100g for measurability and effect).

### 3. **Physical Habitat Prototype Design**
- **Geometry**: Cylindrical shell with rim for tanks; PVC or acrylic tubing for structure.
- **Materials**: PVC pipe (R=0.5m, L=1m, wall thickness 2-3mm); aluminum end caps; total mass 3-4kg empty + 0.5kg water.
- **Rotation Mechanism**: Brushless DC motor (200-500W) with encoder; mounted on precision bearings for low-friction spin; variable speed control (0-10 rad/s).
- **Structural Integrity**: Reinforce rim for tank mounting; balance to <0.1° wobble at max speed.

### 4. **Disturbance Scenarios**
- **Mass Imbalances**: Attach removable weights (20-100g) to 12 angular sectors (simulating scaled crew/cargo); apply randomly or progressively (e.g., 1 sector at 25% episode, 3 at 50%).
- **External Forces**: Impulse from solenoid or air jet (scaled micro-impacts); stochastic timing.
- **Motor Spin-Up**: Ramp rotation from 0 to target speed during episode to test stability under acceleration.

### 5. **Sensors Selection and Integration**
- **Angular Velocity (Gyro)**: High-precision 6-axis IMU (e.g., MPU-6050 or better) at geometric center for ω_x, ω_y, ω_z; sample at 100-200Hz.
- **Acceleration**: Same IMU for accelerometer data (simulating habitat sensors).
- **Mass Trackers**: Strain gauges or load cells under each of 36 tanks for water mass measurement; noisy readings to match sim.
- **Additional**: Encoder on motor shaft for spin rate (ω_z); optical sensors for position if needed.

### 6. **Actuators Implementation**
- **Water Tanks**: 36 syringes or small pumps (20-50mL capacity each, scaled from 100kg sim) arranged in rim; manifolds for redistribution.
- **Control Mechanism**: Pumps driven by PWM signals from microcontroller; simulate valve actions with bidirectional flow.
- **Alternative (Simplified)**: Linear actuators moving masses radially instead of water, if fluid dynamics are hard to replicate.

### 7. **Data Acquisition and Control System**
- **Hardware**: Raspberry Pi 4 or Arduino Mega for control; ADC shields for sensors; motor driver (e.g., ESC); more powerful than 1:100 scale for handling larger prototype.
- **Software**: Python with GPIO/serial for sensor reading; implement SAC inference (export model to ONNX/TFLite for real-time); log data to CSV/SQLite.
- **Real-Time Loop**: 10-50Hz control loop; collect ω, CM offset, tank masses; compute actions; apply to actuators.
- **Telemetry**: Stream data wirelessly (WiFi/Bluetooth) for monitoring; store episodes for analysis.

### 8. **Experimental Protocols**
- **Episode Structure**: 60s duration; start spinning; apply disturbances at set times; run control; measure stability.
- **Test Sequences**:
  - Baseline: No control, measure natural decay.
  - Controlled: Apply trained SAC actions; compare to sim.
  - Curriculum: Progressive disturbances (no imbalance → small → large).
  - Replicates: 10 episodes per condition; average metrics.
- **Data Collection**: Log time-series of ω, nutation, CM offset, actions, rewards; post-process for curves.

### 9. **Validation Criteria**
- **Similarity Thresholds**: Physical nutation < 2× sim value; CM offset < 1.5× sim; reward within 20% (accounting for scaling/noise).
- **Error Tolerances**: Sensor noise < 5% of signal; actuator precision < 10% of command.
- **Success Metrics**: Control reduces wobble by >50% vs. uncontrolled; matches sim trends (e.g., curriculum phases).
- **Statistical Validation**: t-tests on means; correlation >0.8 for key variables vs. sim.

### 10. **Safety and Operational Procedures**
- **Mechanical Safety**: Enclose prototype in sturdy cage (larger size increases kinetic energy); limit speed to <5 rad/s initially; emergency stop button; secure mounting to prevent tipping.
- **Electrical Safety**: Low-voltage (<24V); fuse-protected; ground all electronics.
- **Operational Checks**: Balance check before spin; sensor calibration; dry runs without disturbances; monitor for excessive vibrations.
- **Supervision**: Always attended; have shutdown protocols; larger scale requires more space and caution to avoid injury from moving parts.

This 1:20 scaling provides a more robust prototype (easier assembly, higher fidelity sensors/actuators) while remaining benchtop. Implications include faster dynamics (shorter effective time constants), lower mass requiring precise balancing, and higher build cost (~$1000-2000). Validate iteratively, starting with no disturbances.
