# Level 1: Rigid Body Dynamics for Rotating Space Habitat

## 1. Coordinate Frames

We define three coordinate frames:

### 1.1 Inertial Frame (I)
- Origin at the habitat's nominal centre of mass
- Axes fixed in inertial space: **X_I**, **Y_I**, **Z_I**
- Z_I aligned with the nominal spin axis at t = 0

### 1.2 Body-Fixed Frame (B)
- Origin at the geometric centre of the habitat
- **z_B** along the structural symmetry axis (the intended spin axis)
- **x_B**, **y_B** in the plane of the habitat cross-section
- Rotates with the structure

### 1.3 Relationship
The orientation of B relative to I is described by a rotation matrix **R(t)** parameterised by quaternions **q(t) = [q₀, q₁, q₂, q₃]** to avoid gimbal lock. Euler angles (precession ψ, nutation θ, spin φ) are derived from the quaternion for visualization and analysis but are not used in the integration.

The kinematic equation for the quaternion:

```
dq/dt = ½ Ω(ω) · q
```

where Ω(ω) is the 4×4 skew-symmetric matrix formed from the body-frame angular velocity **ω = [ω_x, ω_y, ω_z]ᵀ**:

```
        ┌  0   -ω_x  -ω_y  -ω_z ┐
Ω(ω) = │ ω_x    0    ω_z  -ω_y  │
        │ ω_y  -ω_z    0    ω_x  │
        └ ω_z   ω_y  -ω_x    0   ┘
```

---

## 2. Habitat Geometry & Structural Inertia

The habitat structural mass distribution is computed analytically for each geometry class.

### 2.1 Cylinder

Parameters: radius R, length L, wall thickness t, wall material density ρ_w, end plate thickness t_e, end plate density ρ_e.

**Shell mass:**
```
m_shell = ρ_w · 2πR · t · L
```

**End plate mass (each):**
```
m_end = ρ_e · π R² · t_e
```

**Structural inertia tensor** (body frame, about geometric centre):

For a thin-walled cylinder:
```
I_xx = I_yy = m_shell · (R²/2 + L²/12) + 2 · m_end · (R²/4 + L²/4)
I_zz = m_shell · R² + 2 · m_end · R²/2
I_xy = I_xz = I_yz = 0   (by symmetry)
```

The second term in I_xx/I_yy for the end plates uses the parallel axis theorem to shift from the end plate centroid to the habitat centre.

### 2.2 Ring (Open Cylinder — no end plates)

Same as cylinder with m_end = 0. This is a short, open-ended cylinder.

### 2.3 Toroid

Parameters: major radius R_maj (centre of tube to axis), minor radius r (tube radius), wall thickness t, density ρ_w.

**Shell mass:**
```
m_torus = ρ_w · 4π² R_maj · r · t
```

**Inertia tensor:**
```
I_xx = I_yy = m_torus · (R_maj² /2 + 5r²/8)
I_zz = m_torus · (R_maj² + 3r²/4)
```

(Thin-walled approximation; exact expressions use inner/outer radii if t/r is not small.)

---

## 3. Sector Decomposition & Moving Masses

### 3.1 Sector Definition

The habitat interior is divided into **N_θ = 12** angular sectors and **N_z = 3** axial slices (for cylinders), giving **36 total sectors**. For toroids, N_z = 1 (single axial slice), giving 12 sectors. Each sector is identified by indices (i, j) where i ∈ {1..12} is the angular index and j ∈ {1..3} is the axial index.

Sector angular span:
```
Δθ = 2π / 12 = 30°  (0.5236 rad)
```

Sector axial span (cylinder):
```
Δz = L / 3
```

Sector centroid in body frame:
```
r_ij = [ R · cos(θ_i),  R · sin(θ_i),  z_j ]ᵀ
```

where θ_i = (i - ½)Δθ and z_j = -L/2 + (j - ½)Δz.

### 3.2 Mass in Each Sector

At any time t, each sector (i,j) contains a lumped mass m_ij(t) representing crew and cargo. These masses move according to prescribed schedules or stochastic models (see Section 8).

### 3.3 Total Mass and Centre of Mass

```
M_total = m_struct + Σ_ij m_ij(t) + Σ_ij m_tank_ij(t) + Σ_j m_manifold_j(t)

r_cm(t) = (1/M_total) · [ m_struct · r_struct_cm
                          + Σ_ij m_ij(t) · r_ij
                          + Σ_ij m_tank_ij(t) · r_tank(i,j)
                          + Σ_j m_manifold_j(t) · r_manifold_j ]
```

For symmetric structures, r_struct_cm = **0**. The manifold positions r_manifold_j are at [0, 0, z_j]ᵀ (on-axis at each station), so manifold water only affects the z-component of CM — circumferential correction comes entirely from the tanks.

### 3.4 Total Inertia Tensor

The total inertia tensor about the geometric centre is:

```
I_total(t) = I_struct
            + Σ_ij m_ij(t) · [ |r_ij|² 𝟏₃ - r_ij r_ijᵀ ]              (crew/cargo)
            + Σ_ij m_tank_ij(t) · [ |r_tank(i,j)|² 𝟏₃ - r_tank(i,j) r_tank(i,j)ᵀ ]  (tanks)
            + Σ_j m_manifold_j(t) · [ |r_manifold_j|² 𝟏₃ - r_manifold_j r_manifold_jᵀ ]  (manifolds)
```

where 𝟏₃ is the 3×3 identity. This is the standard point-mass contribution via the parallel axis theorem.

**Important:** Because the dynamics are formulated about the geometric centre (not the CM), we need the inertia tensor about this fixed body-frame point. This avoids having to track a moving reference point, at the cost of introducing CM-offset coupling terms in the equations of motion.

### 3.5 Inertia Tensor Rate of Change

When masses move between sectors, the inertia tensor changes:

```
dI/dt = Σ_ij dm_ij/dt · [ |r_ij|² 𝟏₃ - r_ij r_ijᵀ ]
       + Σ_ij m_ij(t) · [ 2(r_ij · dr_ij/dt) 𝟏₃ - (dr_ij/dt · r_ijᵀ + r_ij · dr_ij/dtᵀ) ]
```

For the discrete-sector model where masses jump between sectors at discrete times, we can treat dI/dt as piecewise constant between transition events. Alternatively, masses can be interpolated smoothly between sectors to avoid discontinuities in the integrator.

---

## 4. Equations of Motion — Euler's Equations with Variable Inertia

### 4.1 Angular Momentum

The angular momentum about the geometric centre in the body frame:

```
H = I(t) · ω + M_total · r_cm × (ω × r_cm)    ... (cross-coupling from CM offset)
```

However, if we formulate about the **instantaneous CM** (which is standard), the angular momentum is simply:

```
H_cm = I_cm(t) · ω
```

where I_cm is the inertia tensor about the CM, obtained via the parallel axis theorem from the geometric-centre tensor:

```
I_cm = I_total - M_total · [ |r_cm|² 𝟏₃ - r_cm r_cmᵀ ]
```

### 4.2 Euler's Equations

In the body frame, conservation of angular momentum gives:

```
dH/dt |_body + ω × H = τ_ext
```

Expanding:

```
I(t) · dω/dt + dI/dt · ω + ω × (I(t) · ω) = τ_ext
```

Rearranging for the angular acceleration:

```
dω/dt = I(t)⁻¹ · [ τ_ext - dI/dt · ω - ω × (I(t) · ω) ]
```

### 4.3 External Torques

**τ_ext** includes:
- **Spin motor torque** τ_motor(t): Applied about z_B during spin-up/spin-down. Profile is user-defined (constant, ramp, trapezoidal, S-curve).
- **Counterweight actuator torques** (Level 1 placeholder): From control system moving counterweights. Modelled as reaction torques from redistributing mass.
- **No gravity gradient** (microgravity, free-flying habitat).
- **No atmospheric drag** (deep space or high orbit).

### 4.4 Translational Dynamics

The CM offset means the geometric centre orbits the true CM. For a free-flying body:

```
d²r_cm_inertial/dt² = F_ext / M_total
```

With no external forces, the CM is fixed in inertial space. The geometric centre traces out a path around it. This matters for crew comfort (generates a parasitic acceleration at the habitat wall) and for docking/proximity operations.

The parasitic acceleration at a point **r** in the body frame:

```
a_parasitic(r) = ω × (ω × (r - r_cm)) + dω/dt × (r - r_cm)
```

This is what the accelerometers measure (minus the centripetal acceleration from nominal spin, which is the desired artificial gravity).

---

## 5. State Vector and Integration

### 5.1 State Vector

The full state for Level 1:

```
x = [ q₀, q₁, q₂, q₃,  ω_x, ω_y, ω_z,                    (7 rigid body states)
      m_tank_11, ..., m_tank_12,3,                            (36 tank fill levels)
      m_manifold_1, m_manifold_2, m_manifold_3 ]              (3 manifold levels)
```

**Total state dimension: 46**

The quaternion must be renormalised at each step. Tank and manifold masses evolve according to the valve commands and axial transfer dynamics (Section 9.5). Water mass conservation provides a verification constraint at every step.

### 5.2 Equations Summary

```
Rigid body:
  dq/dt  = ½ Ω(ω) · q
  dω/dt  = I(t)⁻¹ · [ τ_ext - dI/dt · ω - ω × (I(t) · ω) ]

Tank dynamics (per tank):
  dm_tank_ij/dt = v_ij · q_circ_max                    (clipped at tank limits)

Manifold dynamics (per station):
  dm_manifold_j/dt = -Σ_i (v_ij · q_circ_max) + q_axial_net_j

Axial transfer (automatic equalisation):
  q_axial_net_j = f(m_manifold_1, m_manifold_2, m_manifold_3, q_axial_max)
```

### 5.3 Integrator Choice

Recommended: **RK4** (classical 4th-order Runge-Kutta) for initial development, with quaternion renormalisation after each full step. For production, a symplectic integrator or implicit method may be needed if energy drift is unacceptable over long simulations.

Time step sizing: The fastest dynamics are at the spin frequency Ω_spin. A rule of thumb is Δt ≤ T_spin / 50 for RK4. At 2 rpm (T_spin = 30 s), this gives Δt ≤ 0.6 s. For accurate nutation capture, use Δt ≤ T_nutation / 20, where T_nutation is typically shorter than T_spin.

### 5.4 Quaternion Normalisation

After each integration step:

```
q ← q / |q|
```

This corrects numerical drift that would otherwise cause the rotation matrix to become non-orthogonal.

---

## 6. Precession and Nutation Analysis

### 6.1 Steady-State Spin with Imbalance

For an axisymmetric body spinning at rate Ω about z_B, small perturbations produce:

**Nutation frequency:**
```
ω_nut = Ω · (I_zz - I_xx) / I_xx     (for I_xx = I_yy, axisymmetric case)
```

This is the free nutation (torque-free precession) frequency. For a thin-walled cylinder where I_zz ≈ 2·I_xx, ω_nut ≈ Ω. For a toroid, the ratio depends on geometry.

**Precession rate** (under a constant transverse torque τ_perp):
```
ω_prec = τ_perp / (I_zz · Ω)
```

### 6.2 Imbalance-Driven Behaviour

A static mass imbalance (CM offset) at spin rate Ω creates a rotating unbalanced centrifugal force. In the body frame this appears as a constant radial force. The resulting motion is a **conical whirl** — the spin axis traces a cone in inertial space.

A dynamic mass imbalance (product of inertia I_xz or I_yz ≠ 0) causes the principal axis to be misaligned from the geometric axis, producing steady-state nutation.

The simulation naturally captures both effects through the full Euler equations without linearisation.

### 6.3 What to Monitor

- **Nutation angle** θ_nut = arccos(z_B · Z_I) — angle between body spin axis and inertial spin axis
- **Angular momentum vector** H in inertial frame (should be conserved when τ_ext = 0)
- **Kinetic energy** T = ½ ωᵀ I ω (should be conserved when τ_ext = 0 and I is constant)
- **CM offset magnitude** |r_cm| — primary disturbance metric

---

## 7. Spin-Up and Spin-Down Modelling

### 7.1 Motor Torque Profiles

The spin motor applies torque about z_B. Available profiles:

| Profile | Description | Parameters |
|---------|-------------|------------|
| Constant | τ(t) = τ₀ | τ₀ |
| Linear ramp | τ ramps from 0 to τ_max | τ_max, t_ramp |
| Trapezoidal | Ramp up, hold, ramp down | τ_max, t_ramp, t_hold |
| S-curve | Smooth jerk-limited profile | τ_max, t_ramp |
| Custom | User-supplied τ(t) lookup table | Table data |

### 7.2 Spin-Up Dynamics

During spin-up from rest:
- ω_z increases from 0 to target Ω
- With mass imbalance present, ω_x and ω_y will be excited as the cross-coupling terms ω × (Iω) become significant
- Passing through resonance conditions where ω_z equals natural frequencies (relevant for Level 2+) requires careful profiling

The time to reach target spin:
```
t_spinup ≈ I_zz · Ω / τ_avg
```

### 7.3 Spin-Down

Reverse of spin-up. The motor applies braking torque. Same profiles available. During spin-down, nutation effects can grow as ω_z decreases (nutation angle scales inversely with spin rate for a given angular momentum perturbation).

### 7.4 Acceleration Profiles at the Rim

During spin-up, the artificial gravity at radius R is:
```
g_art(t) = ω_z(t)² · R
```

The tangential acceleration from spin-up torque:
```
a_tang = dω_z/dt · R = τ_motor / I_zz · R
```

Both must be within crew comfort limits, typically g_art < 1g target and a_tang < 0.01g.

---

## 8. Mass Movement Models

### 8.1 Prescribed Schedules

Crew and cargo movements can be specified as time-scheduled events:

```python
# Example: crew member moves from sector (3,1) to sector (7,1) at t=100s
event = MassTransfer(time=100.0, mass=80.0, from_sector=(3,1), to_sector=(7,1), duration=30.0)
```

The `duration` parameter controls how quickly the mass transitions, allowing smooth interpolation of the inertia tensor.

### 8.2 Stochastic Models

For Monte Carlo analysis, crew movement can follow probabilistic models:
- **Random walk**: Each crew member moves to an adjacent sector with some probability per time step
- **Activity-based**: Crew members transition between zones (sleep, work, exercise, galley) on schedules with random perturbations
- **Correlated movement**: Shift changes where multiple crew move simultaneously (worst-case scenario for imbalance)

### 8.3 Cargo Operations

Large mass movements (resupply, waste management, manufacturing) can be scheduled deterministically. These are typically the largest single-event disturbances.

---

## 9. Counterweight Model (36 Rim Water Tanks — Hybrid Manifold)

### 9.1 Tank Layout

The control system uses **36 rim water tanks**: 12 tanks at each of the 3 axial stations, co-located with the 12 angular sectors. Each tank sits at the rim radius R at angular position θ_i and axial position z_j.

**Tank position in body frame:**
```
r_tank(i,j) = [ R · cos(θ_i),  R · sin(θ_i),  z_j ]ᵀ
```

where i ∈ {1..12} is the angular index and j ∈ {1..3} is the axial station index, using the same angular and axial positions as the sector decomposition (Section 3.1).

**Effect on inertia tensor:** Each tank contributes as a point mass at its body-frame position. The total counterweight contribution to the inertia tensor:

```
I_tanks(t) = Σ_i Σ_j m_tank_ij(t) · [ |r_tank(i,j)|² 𝟏₃ - r_tank(i,j) r_tank(i,j)ᵀ ]
```

This is added to I_struct and the crew/cargo sector contributions to form I_total.

### 9.2 Hybrid Manifold Topology

The plumbing uses a **hybrid manifold** architecture with two timescales of actuation:

**Fast channel — circumferential redistribution:**
Each axial station has its own local manifold (reservoir). The 12 tanks at that station are connected to the local manifold via individual valves/pumps. Water can be rapidly redistributed among the 12 tanks within a single station.

```
Station j manifold mass: m_manifold_j(t)
Conservation per station (when axial transfers inactive):
  m_manifold_j + Σ_i m_tank_ij = m_water_station_j = const
```

**Slow channel — axial redistribution:**
The 3 local manifolds are connected by axial transfer lines. Water can be pumped between stations, but at a lower flow rate (longer pipe runs, smaller cross-section).

```
Total water conservation:
  Σ_j [ m_manifold_j + Σ_i m_tank_ij ] = m_water_total = const
```

**Flow rate limits:**

| Channel | Rate limit | Typical value |
|---------|-----------|---------------|
| Tank ↔ local manifold (circumferential) | q_circ_max (kg/s per valve) | Fast — primary disturbance rejection |
| Manifold ↔ manifold (axial) | q_axial_max (kg/s per line) | Slow — steady-state trim, q_axial_max << q_circ_max |

The ratio q_circ_max / q_axial_max is a design parameter. A ratio of 5–10× is physically reasonable and creates a natural timescale separation.

### 9.3 Correction Authority

The 36-tank layout provides full 5-DOF correction:

| Correction | Method | Tanks involved |
|------------|--------|----------------|
| CM offset x_B | Differential angular redistribution at all stations | All 36 |
| CM offset y_B | Differential angular redistribution at all stations | All 36 |
| CM offset z_B | Axial transfer: shift total water mass between stations | Axial lines |
| Product I_xz | Opposing angular patterns at different stations (e.g. +x at fwd, -x at aft) | Cross-station coordination |
| Product I_yz | Opposing angular patterns at different stations (e.g. +y at fwd, -y at aft) | Cross-station coordination |

Static imbalance (CM offset) correction uses the fast circumferential channel. Dynamic imbalance (product of inertia) correction requires coordinated cross-station action using both channels.

### 9.4 Action Space for Control

The action space is **36-dimensional**, one valve command per tank:

```
a = [ v_11, v_12, ..., v_1,12,     (station 1: 12 valve commands)
      v_21, v_22, ..., v_2,12,     (station 2: 12 valve commands)
      v_31, v_32, ..., v_3,12 ]    (station 3: 12 valve commands)
```

Each v_ij ∈ [-1, +1] represents the normalised valve command for tank (i,j):
- v_ij > 0: fill tank from local manifold (rate = v_ij · q_circ_max)
- v_ij < 0: drain tank to local manifold (rate = |v_ij| · q_circ_max)
- v_ij = 0: valve closed

Additionally, 2 axial transfer commands (between the 3 manifolds):

```
a_axial = [ q_12, q_23 ]
```

where q_12 is the flow rate from manifold 1 to manifold 2, and q_23 from manifold 2 to manifold 3. These can be positive or negative.

**Total action dimension: 38** (36 valve commands + 2 axial transfers).

However, to keep the action space at 36 dimensions, the axial transfers can be derived implicitly: if the net flow into/out of a station's manifold from its 12 valves would cause the manifold to empty or overfill, the axial transfer system automatically equalises pressure between manifolds. This makes the axial channel reactive rather than directly controlled, reducing the RL agent's burden while preserving the physical capability.

**Recommended approach for RL:** Use the **36-dimensional** action space (valve commands only). Implement the axial transfer as an automatic background process that equalises manifold levels at rate q_axial_max. The agent learns circumferential control; axial rebalancing happens passively. Once the agent is trained, the axial channel can be promoted to direct control if needed.

### 9.5 Tank State Dynamics

The fill state of each tank evolves as:

```
dm_tank_ij/dt = v_ij · q_circ_max                     (valve flow)

dm_manifold_j/dt = -Σ_i (v_ij · q_circ_max)           (manifold: net of all valve flows at station j)
                   + q_axial_in_j - q_axial_out_j      (axial transfers)
```

**Constraints:**
```
0 ≤ m_tank_ij ≤ m_tank_max        (tank capacity)
0 ≤ m_manifold_j                    (manifold can't go negative)
```

When a tank hits its upper or lower limit, the valve command is clipped. When a manifold approaches empty, all drain commands at that station are throttled proportionally.

### 9.6 Reward Shaping for Tank Control

With 36 actuators and a large null space, auxiliary reward terms prevent degenerate policies:

```
r_total = r_vibration + λ_1 · r_energy + λ_2 · r_smooth + λ_3 · r_reserve
```

| Term | Purpose | Formula |
|------|---------|---------|
| r_vibration | Primary: minimise wobble | -‖ω_x, ω_y‖ or -‖a_parasitic‖ |
| r_energy | Penalise total pump activity | -Σ ‖v_ij‖ |
| r_smooth | Penalise jerky valve commands | -Σ ‖v_ij(t) - v_ij(t-1)‖ |
| r_reserve | Prefer balanced fill levels | -Σ (m_tank_ij - m_uniform)² |

The reserve term softly biases the agent toward keeping water distributed evenly, maintaining capacity for future corrections without constraining which correction strategy it finds.

---

## 10. Sensor Model (Observations)

### 10.1 Accelerometers

Each 3-axis accelerometer at body-frame position **r_s** measures:

```
a_meas = ω × (ω × r_s) + dω/dt × r_s + noise
```

The first term is centripetal (dominant, contains the spin-rate information). The second term contains the nutation/precession dynamics. In Level 1, there are no vibration components.

**Sensor noise model:** White Gaussian noise with standard deviation σ_acc per axis.

### 10.2 Sector Mass Estimates

The mass tracking system provides:
```
m̂_ij(t) = m_ij(t) + noise
```

With update rate f_mass (typically much lower than accelerometer rate, e.g. 1 Hz vs 100 Hz).

### 10.3 Observation Vector

The full observation for the RL agent (with N_s accelerometers):

```
o = [ a_meas_1, ..., a_meas_Ns,            (3 × N_s accelerometer readings)
      m̂_11, ..., m̂_12,3,                  (36 sector mass estimates)
      m_tank_11, ..., m_tank_12,3,          (36 current tank fill levels)
      m_manifold_1, m_manifold_2, m_manifold_3 ]  (3 manifold levels)
```

Total observation dimension: 3·N_s + 36 + 36 + 3 = 3·N_s + 75.

For example, with 6 accelerometers: dim(o) = 18 + 75 = **93**.

The observation naturally decomposes into:
- **Disturbance information**: sector masses + accelerometer readings (what's wrong)
- **Actuator state**: tank fill levels + manifold levels (what resources are available)

---

## 11. Conservation Laws (Verification)

### 11.1 Angular Momentum Conservation

When τ_ext = 0 (no motor torque, no external forces):

```
H_inertial = R(q) · I(t) · ω = constant
```

This must hold even as I(t) changes due to mass movement. The simulation must verify:

```
|H(t) - H(0)| / |H(0)| < ε_tol
```

at every time step. Typical tolerance: ε_tol = 1e-8 for RK4 with appropriate step size.

### 11.2 Energy

Total kinetic energy:

```
T = ½ ωᵀ I(t) ω
```

When masses move and I changes with no external torque, kinetic energy is **not** conserved — the moving masses do work against the centrifugal field. The change in kinetic energy should equal the work done by internal forces moving the masses. This is a useful consistency check.

### 11.3 Water Mass Conservation

```
Σ_j [ m_manifold_j(t) + Σ_i m_tank_ij(t) ] = m_water_total = const
```

Must hold exactly (to machine precision) at every time step. Any drift indicates a bug in the tank/manifold integration.

### 11.4 Quaternion Norm

```
|q|² = q₀² + q₁² + q₂² + q₃² = 1
```

Must hold to machine precision after normalisation.

---

## 12. Computational Notes

### 12.1 Inertia Tensor Inverse

I(t) is 3×3 symmetric positive definite. Direct inversion (Cramer's rule or Cholesky) is fine — this is not a bottleneck. Recompute at each RK4 sub-step since I depends on the current mass distribution.

### 12.2 Mass Movement Interpolation

To avoid discontinuous jumps in I(t) when a mass transfers between sectors, linearly interpolate the mass between the source and destination sector positions over the transfer duration. This gives a smooth dI/dt and avoids integrator difficulties.

### 12.3 Performance Considerations

For a single forward simulation, Level 1 is computationally cheap — the bottleneck is the 3×3 matrix operations at each step, which are trivial. The cost scales with the number of RL training episodes, not the physics complexity. At 100 Hz simulation rate for a 1-hour scenario, that's 360,000 steps per episode — straightforward in Python, and the main candidate for C++ acceleration later is the batch simulation of many episodes in parallel.

### 12.4 Units Convention

| Quantity | Unit |
|----------|------|
| Length | metres (m) |
| Mass | kilograms (kg) |
| Time | seconds (s) |
| Angle | radians (rad) |
| Angular velocity | rad/s |
| Torque | N·m |
| Acceleration | m/s² |
| Inertia | kg·m² |

Spin rates may be displayed in rpm for user convenience but are always stored and computed in rad/s.
