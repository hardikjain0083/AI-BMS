# ebms-sim: Physics-Based Simulator & Synthetic-Data Generator for E-Rickshaw AI BMS

> [!WARNING]
> **MODEL VALIDATION DISCLAIMER**:
> This package is a first-principles **physics-based mathematical simulator and synthetic data generator**. It is **NOT** validated or calibrated against physical instrumented vehicle telemetry or laboratory cell cycler hardware. All default constants are derived from engineering literature and manufacturer datasheet placeholders. Before deploying predictive models trained on this synthetic data to an embedded microcontroller (e.g., STM32/ESP32) or a production vehicle, parameters **MUST** be calibrated against real-world test data using `config.yaml`.

---

## 1. Project Overview & Architectural Principles

`ebms-sim` provides first-principles forward simulation of longitudinal vehicle dynamics, battery electrochemical state, pack thermal evolution, and smart charging for Indian electric three-wheelers (L5M e-rickshaws).

### Core Architecture Rules
1. **Pure Physics Functions**: All core algorithms in `ebms_sim` (`vehicle.py`, `battery.py`, `thermal.py`, `charging.py`, `optimizer.py`) are pure mathematical functions: numbers in, numbers out, no global state, and zero file/network I/O. This ensures direct portability to embedded C/C++ firmware.
2. **Explicit Dataclasses & Units**: All parameters are strongly typed dataclasses with physical units embedded in variable names (e.g., `mass_kg`, `crr`, `cda_m2`, `eta_drive`, `aux_w`, `r0_ohm`).
3. **Chemistry Presets**: Built-in support for:
   - `lead_acid_48V_100Ah`: 4x 12V tubular blocks in series (48V nominal, 100 Ah @ C20, Peukert $k=1.20$).
   - `lfp_51V_60Ah`: 16S LiFePO4 cells (51.2V nominal, 60 Ah, flat OCV plateau, $k=1.00$).
4. **Grouped Route Isolation**: Generates both trip-level and 30-second window tables with explicit `route_id` tags for `GroupKFold` cross-validation to prevent spatial terrain data leakage.

---

## 2. Mathematical Equations & Physics Foundations

### 2.1 Vehicle Dynamics (`ebms_sim/vehicle.py`)
Longitudinal tractive wheel forces are computed by summing rolling resistance, aerodynamic drag, road slope gravity, and drivetrain rotational inertia:

$$\theta = \arctan(\text{grade})$$

1. **Rolling Resistance Force**:
   $$F_{\text{roll}} = m \cdot g \cdot C_{rr} \cdot \cos(\theta)$$
   *(Opposes vehicle direction; zero when vehicle is stationary).*

2. **Aerodynamic Drag Force**:
   $$v_{\text{rel}} = v + v_{\text{wind}}$$
   $$F_{\text{aero}} = \frac{1}{2} \rho_{\text{air}}(T) \cdot C_d A \cdot v_{\text{rel}}^2 \cdot \text{sgn}(v_{\text{rel}})$$
   Where air density $\rho_{\text{air}}(T) = \frac{P_{\text{atm}}}{R_{\text{spec}} \cdot (T_{\text{ambient}} + 273.15)}$.

3. **Grade Resistance Force**:
   $$F_{\text{grade}} = m \cdot g \cdot \sin(\theta) = m \cdot g \cdot \frac{\text{grade}}{\sqrt{1 + \text{grade}^2}}$$

4. **Inertial Force**:
   $$F_{\text{inertia}} = m \cdot (1 + \lambda_{\text{rot}}) \cdot a$$

5. **Wheel Power**:
   $$P_{\text{wheel}} = (F_{\text{roll}} + F_{\text{aero}} + F_{\text{grade}} + F_{\text{inertia}}) \cdot v$$

6. **Battery Electrical Power Demand**:
   - In **constant efficiency mode** (`eta_model="constant"`):
     $$P_{\text{batt}} = \begin{cases} \frac{P_{\text{wheel}}}{\eta_{\text{drive}}} + P_{\text{aux}}, & \text{if } P_{\text{wheel}} \ge 0 \text{ (propulsion)} \\ P_{\text{wheel}} \cdot \eta_{\text{regen}} + P_{\text{aux}}, & \text{if } P_{\text{wheel}} < 0 \text{ (braking / descent)} \end{cases}$$
   - In **loss-based mode** (`eta_model="loss_based"`):
     $$P_{\text{loss}} = p_{0} + c_1 \cdot P_{\text{wheel}} + c_2 \cdot P_{\text{wheel}}^2$$
     $$P_{\text{batt}} = P_{\text{wheel}} + P_{\text{loss}} + P_{\text{aux}} \quad (P_{\text{wheel}} \ge 0)$$
     *(Calibrated so overall efficiency $\approx 0.82$ at typical 700 W cruise, falling at low power/speed).*
   *(Default regen efficiency $\eta_{\text{regen}} = 0.0$, yielding strictly $P_{\text{aux}}$ on steep downhills).*

---

### 2.2 Battery Electrochemical Model (`ebms_sim/battery.py`)

1. **Open-Circuit Voltage (OCV-SOC Table)**:
   Piecewise linear interpolation over empirical lookup tables for 48V Lead-Acid and 51.2V LFP chemistries:
   $$V_{\text{ocv}} = \text{OCV}(\text{SOC})$$

2. **Internal Resistance Model**:
   Accounts for electrolyte sluggishness at cold temperatures and solid-electrolyte interphase (SEI) / sulfation aging degradation:
   $$R(T, \text{SOH}) = R_{0,\text{base}} \cdot \left[1 + \alpha_T \cdot (25 - T)\right] \cdot \left[1 + \beta_{\text{SOH}} \cdot (1 - \text{SOH})\right]$$

3. **Peukert Law (High-Rate Capacity Loss)**:
   For lead-acid chemistries ($k \approx 1.15 - 1.25$):
   $$I_{\text{eff}} = I \cdot \left(\frac{\max(I, I_{\text{ref}})}{I_{\text{ref}}}\right)^{k - 1}$$
   For LFP ($k = 1.00$), $I_{\text{eff}} = I$.

4. **Usable Capacity Temperature Derating**:
   $$C_{\text{usable}}(T) = C_{\text{nom}} \cdot f_T(T)$$
   Where $f_T(25^\circ\text{C}) = 1.00$ and $f_T(0^\circ\text{C}) \approx 0.80$.

5. **Terminal Voltage & Electrical Current Solution**:
   $$V_{\text{term}} = V_{\text{ocv}} - I \cdot R$$
   Solving from power $P_{\text{batt}} = V_{\text{term}} \cdot I = V_{\text{ocv}} \cdot I - R \cdot I^2$:
   $$I = \frac{V_{\text{ocv}} - \sqrt{V_{\text{ocv}}^2 - 4 R P_{\text{batt}}}}{2 R}$$

6. **Coulomb Counting SOC Integration**:
   $$\Delta \text{Ah} = \frac{I_{\text{eff}} \cdot \Delta t}{3600}$$
   $$\text{SOC}_{t+1} = \text{SOC}_t - \frac{\Delta \text{Ah}}{C_{\text{usable}}(T)}$$

---

### 2.3 Lumped Pack Thermal Model (`ebms_sim/thermal.py`)

1. **Joule Heat Generation**:
   $$Q_{\text{gen}} = I^2 \cdot R(T, \text{SOH})$$

2. **Convective Dissipation to Ambient with Airflow**:
   $$R_{\text{th\_eff}} = \frac{R_{\text{th}}}{1 + \gamma_{\text{cooling}} \cdot v_{\text{vehicle}}}$$
   $$Q_{\text{loss}} = \frac{T_{\text{pack}} - T_{\text{ambient}}}{R_{\text{th\_eff}}}$$

3. **Exact Analytic Temperature Integration**:
   Solving $C_{\text{th}} \frac{dT}{dt} = Q_{\text{gen}} - \frac{T - T_{\text{amb}}}{R_{\text{th\_eff}}}$ over step $\Delta t$:
   $$T_{\text{eq}} = T_{\text{ambient}} + Q_{\text{gen}} \cdot R_{\text{th\_eff}}$$
   $$\tau = C_{\text{th}} \cdot R_{\text{th\_eff}}$$
   $$T(t + \Delta t) = T_{\text{eq}} + \left[T(t) - T_{\text{eq}}\right] e^{-\Delta t / \tau}$$

---

### 2.4 Cruising Speed Optimizer (`ebms_sim/optimizer.py`)

Specific energy consumption (Wh/km) exhibits a convex **U-shaped curve** across speed $v \in [10, 25]$ km/h:
$$\text{Wh/km}(v) = \frac{P_{\text{batt}}(v)}{v} \approx \underbrace{\frac{P_{\text{aux}}}{v}}_{\text{High loss at low speed}} + \underbrace{\frac{m g C_{rr}}{\eta_{\text{drive}}}}_{\text{Rolling friction}} + \underbrace{\frac{1}{2 \eta_{\text{drive}}} \rho C_d A v^2}_{\text{Aero drag dominates at high speed}}$$

**Characteristics of the Optimum**:
- The exact energy-optimal speed is not universal: it depends on vehicle payload, road slope, auxiliary power, and the chosen drivetrain efficiency model.
- Because the consumption curve is shallow across the 12–18 km/h range (varying by less than 1–2 Wh/km), maintaining any steady speed within this window yields near-optimal efficiency.
- **Computed Optima on Flat Road (2 Pax / 130 kg, 25°C)**:
  * **Lead-Acid 48V 100Ah (`constant`)**: Optimal speed = **11.9 km/h** (39.1 Wh/km)
  * **Lead-Acid 48V 100Ah (`loss_based`)**: Optimal speed = **14.4 km/h** (40.1 Wh/km)
  * **LFP 51.2V 60Ah (`constant`)**: Optimal speed = **12.0 km/h** (33.1 Wh/km)
  * **LFP 51.2V 60Ah (`loss_based`)**: Optimal speed = **14.6 km/h** (35.4 Wh/km)

For multi-segment routes subject to trip time constraint $\sum t_i \le T_{\text{max}}$, dynamic programming / dual Lagrangian bisection solves:
$$\min_{\{v_i \in [10, 25]\}} \sum_{i} \left[E_i(v_i) + \lambda \cdot t_i(v_i)\right]$$

---

### 2.5 Smart Charging & Thermal Limits (`ebms_sim/charging.py`)

> [!NOTE]
> **Approximation Notice**: The Constant Voltage (CV) taper phase is modeled as a linear current taper approximation from `taper_soc` to 1.0, rather than a true dynamic differential equation of cell electrochemical diffusion overpotential.

#### Lithium / LFP Charging
LFP charging current limit follows a piecewise bell-shaped thermal derating profile:
$$f_{\text{thermal}}(T) = \begin{cases} 0.0, & T \le 0^\circ\text{C} \text{ (Freezing shutoff / plating prevention)} \\ 0.20 \cdot \frac{T}{5}, & 0^\circ\text{C} < T < 5^\circ\text{C} \text{ (Trickle rate)} \\ 0.20 + 0.80 \cdot \frac{T - 5}{10}, & 5^\circ\text{C} \le T < 15^\circ\text{C} \text{ (Ramped charging)} \\ 1.00, & 15^\circ\text{C} \le T \le 35^\circ\text{C} \text{ (Full CC charge rate)} \\ 1.00 - \frac{T - 35}{10}, & 35^\circ\text{C} < T < 45^\circ\text{C} \text{ (Thermal derating)} \\ 0.0, & T \ge 45^\circ\text{C} \text{ (High-temperature runaway shutoff)} \end{cases}$$

Combined with linear CC-CV taper approximation when $\text{SOC} > \text{taper\_soc}$:
$$I_{\text{charge}}(\text{SOC}, T) = I_{\text{min}} + (I_{\text{max}}(T) - I_{\text{min}}) \cdot \frac{1 - \text{SOC}}{1 - \text{taper\_soc}}$$

#### Lead-Acid Charging
Lead-acid batteries do **not** use the lithium bell curve. Instead, charging uses:
1. **Temperature-Compensated Charging Voltage**:
   $$V_{\text{cell}}(T) = V_{\text{ref}} + (-0.003) \cdot (T - 25.0) \quad [\text{V/cell}]$$
   - Absorption target: $V_{\text{ref}} = 2.40\text{ V/cell}$ ($57.6\text{ V}$ pack at 25°C across 24 cells).
   - Float target: $V_{\text{ref}} = 2.25\text{ V/cell}$ ($54.0\text{ V}$ pack at 25°C across 24 cells).
2. **Current Limit**:
   - Continuous CC charging limit between $0.10\text{ C}$ and $0.20\text{ C}$ (default 0.15C = 15 A for 100 Ah).
   - Linear taper above absorption `taper_soc` (0.80) down to `min_charge_current_a`.
   - **Temperature Safety Window**: Charging current is 0.0 A below 0°C and shuts off above 50°C.

---

## 3. Parameter Sources & Literature Baseline

| Parameter | Default Value | Unit | Engineering Source / Literature Placeholder |
|---|---|---|---|
| E-Rickshaw Curb Mass | 340.0 | kg | ARAI / ICAT L5M homologation specs |
| Driver Mass | 65.0 | kg | Standard passenger allowance |
| Passenger Mass | 65.0 | kg | Standard adult anthropometric average |
| Tire $C_{rr}$ (Rolling) | 0.018 | - | Typical e-rickshaw 3.75-12 / 90/90-12 tube-type tires |
| Drag Area $C_d A$ | 0.85 | $\text{m}^2$ | Frontal area $\approx 1.5\text{ m}^2$, $C_d \approx 0.55-0.60$ (bluff body canopy) |
| Drive Efficiency $\eta_{\text{drive}}$| 0.82 | - | BLDC motor (88%) + planetary gearbox (95%) + controller (98%) |
| Aux Power $P_{\text{aux}}$ | 50.0 | W | Halogen/LED headlights, taillights, meter, BMS quiescent load |
| Lead-Acid Pack $R_0$ | 0.065 | $\Omega$ | 4x 12V 100Ah flooded tubular deep-cycle blocks in series |
| LFP Pack $R_0$ | 0.025 | $\Omega$ | 16S prismatic LiFePO4 cells with busbars |
| Peukert Exponent ($k$) | 1.20 | - | Standard lead-acid discharge exponent at C5-C20 |
| Pack Thermal Mass ($C_{\text{th}}$) | 90,000 | J/K | $100\text{ kg} \times 900\text{ J/(kg}\cdot\text{K)}$ |

---

## 4. Usage Guide

### 4.1 Running the Verification Test Suite
All physics rules, monotonicity constraints, and boundary conditions are validated via pytest:
```bash
python -m pytest tests/
```

### 4.2 Generating Demo Visualizations
Run the standalone demonstration script:
```bash
python demo.py
```
This produces:
- `demo_trip_plot.png`: Multi-panel 10-minute trip trace (speed, elevation, power, V, I, SOC, temp).
- `demo_speed_sweep_curve.png`: Wh/km vs speed U-curve.
- `demo_charging_curve.png`: Charging bell curve & CC-CV session.

Interactive Jupyter notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

### 4.3 Generating Synthetic Datasets (CLI)
Generate 100,000 trips across multicore CPUs with realistic noise and parameter randomization:
```bash
python -m ebms_sim.datagen --n 100000 --out data/trips.parquet --seed 42 --workers 8
```
This generates:
1. `data/trips.parquet`: Trip-level dataset.
2. `data/trips_windows.parquet`: 30-second sliding window dataset.

### 4.4 Training Machine Learning Models (CLI)
Train XGBoost models for Wh/km and range with `GroupKFold` on `route_id` and evaluate on held-out temperature bands:
```bash
python -m ebms_sim.train --data data/trips.parquet --out-dir models/ --folds 5 --held-out-temp 38.0
```

### 4.5 Telemetry Recalibration via `config.yaml`
When vehicle CAN-bus data or IoT gateway logs become available:
1. Extract empirical rolling resistance from coast-down deceleration tests.
2. Record internal resistance $R_0$ from voltage drop pulses ($\Delta V / \Delta I$).
3. Update `config.yaml` with the calibrated numbers.

### 4.6 Running the Streamlit Demonstration App
Install the dashboard dependencies and launch the interactive professor demonstration:
```bash
python -m pip install -r requirements-app.txt
streamlit run app/app.py
```

The app opens with a permanent validation warning. Use the sidebar to select a
battery preset, vehicle and route parameters, then inspect the nine tabs for
trip traces, range and speed advice, hills and regeneration, smart charging,
scenario comparison, AI approximation, generated data, equations, and live
health checks. The Guided Demo walks through the seven recommended teaching
steps with Next and Previous controls.

### 4.7 Five-Minute Demo Script
1. Start with the lead-acid preset at 2 passengers, flat route, 20 km/h, and 25°C. Show the trip metrics and live health checks.
2. Add passengers and luggage. Point out the change in Wh/km and remaining range.
3. Switch to a hill-climb route and compare the flat and hilly energy bars. Increase regen to show the recovery effect.
4. Open Range & Speed Advisor and switch between `constant` and `loss_based` efficiency. Explain that the optimum depends on payload, auxiliary load, grade, and the efficiency model; the curve is shallow.
5. Open Smart Charging and compare 5°C, 25°C, and 40°C. Explain LFP bell-curve derating, separate lead-acid voltage compensation, and the linear CC-CV approximation.
6. Freeze Scenario A, change one parameter, and use Scenario Comparison to discuss deltas.
7. Finish with AI vs Physics. The XGBoost model is presented as a fast approximator of this simulator, not as an improvement or physical validation.

### 4.8 Screenshots
Screenshots placeholder: add dashboard captures here after the local Streamlit
demo has been run with both chemistry presets and the Guided Demo enabled.

### 4.9 Known Limitations
- No real-vehicle or laboratory validation has been performed.
- Pack temperature is represented by a lumped thermal node.
- The CC-CV taper is a linear approximation, not a true constant-voltage electrochemical phase.
- Lead-acid capacity uses a simplified Peukert correction.
- The machine-learning model learns synthetic simulator outputs and is only a fast approximator of the physics engine.

