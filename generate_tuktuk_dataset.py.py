import math
import os
import time
import numpy as np
import pandas as pd

# =========================================================================
# Indian Electric 3-Wheeler (E-Rickshaw / Tuk-Tuk / E-Auto / E-Cargo) Specs
# =========================================================================
RICKSHAW_CLASSES = {
    "l3_erickshaw_passenger": {
        "chemistry_probs": [("LFP", 0.70), ("Lead-Acid", 0.30)],
        "cap_kwh_range": (4.0, 6.2),
        "v_nom_lfp": 51.2,      # 16S LFP
        "v_nom_la": 48.0,       # 4x 12V Lead-Acid
        "v_nom_nmc": 51.8,      # 14S NMC
        "curb_mass_range": (290.0, 360.0),
        "payload_range": (70.0, 460.0),
        "cd_range": (0.62, 0.78),
        "area_range": (1.85, 2.15),
        "r_nom_lfp": 0.022,
        "r_nom_la": 0.048,
        "r_nom_nmc": 0.020,
        "c_thermal": 45000.0,
        "max_power_w": 2400.0,
        "max_charge_a": 25.0,
        "max_discharge_a": 65.0,
        "v_max_mps": 7.5,       # ~27 km/h
        "eta_drive": 0.80,
        "eta_regen": 0.35,
    },
    "l5m_eauto_passenger": {
        "chemistry_probs": [("LFP", 0.80), ("NMC", 0.20)],
        "cap_kwh_range": (7.5, 10.5),
        "v_nom_lfp": 51.2,
        "v_nom_nmc": 51.8,
        "v_nom_la": 48.0,
        "curb_mass_range": (380.0, 470.0),
        "payload_range": (70.0, 390.0),
        "cd_range": (0.55, 0.68),
        "area_range": (2.05, 2.35),
        "r_nom_lfp": 0.014,
        "r_nom_nmc": 0.012,
        "r_nom_la": 0.040,
        "c_thermal": 74000.0,
        "max_power_w": 11000.0,
        "max_charge_a": 65.0,
        "max_discharge_a": 220.0,
        "v_max_mps": 16.5,      # ~59 km/h
        "eta_drive": 0.86,
        "eta_regen": 0.65,
    },
    "l5n_ecargo_loader": {
        "chemistry_probs": [("LFP", 0.85), ("NMC", 0.15)],
        "cap_kwh_range": (8.3, 13.0),
        "v_nom_lfp": 51.2,
        "v_nom_nmc": 55.5,
        "v_nom_la": 48.0,
        "curb_mass_range": (440.0, 560.0),
        "payload_range": (100.0, 720.0),
        "cd_range": (0.68, 0.88),
        "area_range": (2.40, 2.85),
        "r_nom_lfp": 0.013,
        "r_nom_nmc": 0.011,
        "r_nom_la": 0.038,
        "c_thermal": 88000.0,
        "max_power_w": 12000.0,
        "max_charge_a": 75.0,
        "max_discharge_a": 240.0,
        "v_max_mps": 15.0,      # ~54 km/h
        "eta_drive": 0.85,
        "eta_regen": 0.62,
    },
    "l5m_swappable_eauto": {
        "chemistry_probs": [("NMC", 0.55), ("LFP", 0.45)],
        "cap_kwh_range": (4.5, 6.8),
        "v_nom_lfp": 51.2,
        "v_nom_nmc": 51.8,
        "v_nom_la": 48.0,
        "curb_mass_range": (350.0, 420.0),
        "payload_range": (70.0, 360.0),
        "cd_range": (0.56, 0.68),
        "area_range": (2.00, 2.25),
        "r_nom_lfp": 0.018,
        "r_nom_nmc": 0.015,
        "r_nom_la": 0.042,
        "c_thermal": 52000.0,
        "max_power_w": 9000.0,
        "max_charge_a": 50.0,
        "max_discharge_a": 185.0,
        "v_max_mps": 14.5,      # ~52 km/h
        "eta_drive": 0.85,
        "eta_regen": 0.60,
    },
}

# =========================================================================
# 8 Indian 3-Wheeler / Tuk-Tuk Operating Regimes
# =========================================================================
TUKTUK_REGIMES = {
    "old_city_bazaar_stopgo": dict(
        elev0_range=(180, 550), t_amb_range=(28, 42), rh_range=(35, 75),
        surface="congested_asphalt_broken", crr_base=0.0155,
        v_frac_range=(0.35, 0.55), wind_range=(-1.5, 3.5),
        grade_mode="flat_narrow", stop_prob=0.52,
        solar_heat_w=110.0,
    ),
    "metro_feeder_flyover_ramps": dict(
        elev0_range=(200, 920), t_amb_range=(25, 39), rh_range=(35, 70),
        surface="urban_asphalt", crr_base=0.0130,
        v_frac_range=(0.55, 0.82), wind_range=(-2.0, 5.5),
        grade_mode="flyover_ramps", stop_prob=0.32,
        solar_heat_w=100.0,
    ),
    "delhi_up_summer_heatwave": dict(
        elev0_range=(190, 260), t_amb_range=(41.5, 48.5), rh_range=(15, 38),
        surface="urban_asphalt", crr_base=0.0135,
        v_frac_range=(0.50, 0.78), wind_range=(1.5, 8.0),
        grade_mode="flat_urban", stop_prob=0.35,
        solar_heat_w=210.0,
    ),
    "monsoon_waterlogged_roads": dict(
        elev0_range=(5, 120), t_amb_range=(25, 31), rh_range=(82, 98),
        surface="wet_waterlogged_potholes", crr_base=0.0210,
        v_frac_range=(0.38, 0.65), wind_range=(3.0, 12.0),
        grade_mode="gentle_urban", stop_prob=0.30,
        solar_heat_w=35.0,
    ),
    "hill_town_steep_climb": dict(
        elev0_range=(600, 1800), t_amb_range=(14, 29), rh_range=(55, 88),
        surface="hilly_rough_asphalt", crr_base=0.0150,
        v_frac_range=(0.45, 0.72), wind_range=(-2.0, 6.5),
        grade_mode="hill_climb", stop_prob=0.18,
        solar_heat_w=85.0,
    ),
    "hill_town_steep_descent": dict(
        elev0_range=(1100, 2100), t_amb_range=(13, 27), rh_range=(55, 85),
        surface="hilly_rough_asphalt", crr_base=0.0145,
        v_frac_range=(0.50, 0.75), wind_range=(-3.0, 4.5),
        grade_mode="hill_descent", stop_prob=0.14,
        solar_heat_w=75.0,
    ),
    "tier2_rural_potholed_mandi": dict(
        elev0_range=(250, 580), t_amb_range=(28, 44), rh_range=(25, 65),
        surface="rural_mud_gravel_potholes", crr_base=0.0225,
        v_frac_range=(0.48, 0.78), wind_range=(-2.0, 7.5),
        grade_mode="rolling_rural", stop_prob=0.20,
        solar_heat_w=160.0,
    ),
    "north_india_winter_morning": dict(
        elev0_range=(180, 280), t_amb_range=(3.0, 11.5), rh_range=(75, 96),
        surface="urban_asphalt", crr_base=0.0135,
        v_frac_range=(0.45, 0.72), wind_range=(-1.0, 4.0),
        grade_mode="flat_urban", stop_prob=0.28,
        solar_heat_w=25.0,
    ),
}


def ocv_3w(soc: float, v_nom: float, chemistry: str) -> float:
    """Open Circuit Voltage (V) curve by chemistry."""
    soc = max(0.0, min(1.0, soc))
    if chemistry == "LFP":
        scale = v_nom / 3.20
        cell_v = 3.22 + 0.14 * soc + 0.22 * (soc ** 8) - 0.58 * math.exp(-18.0 * soc)
        return cell_v * scale
    elif chemistry == "NMC":
        scale = v_nom / 3.65
        cell_v = 3.08 + 0.90 * soc + 0.20 * (soc ** 3) - 0.20 * math.exp(-14.0 * soc)
        return cell_v * scale
    else:
        # 48V Lead-Acid (4x 12V)
        return 43.2 + 8.4 * soc - 1.2 * math.exp(-12.0 * soc)


def simulate_tuktuk_trip(
    trip_id: int,
    regime_name: str,
    vclass_name: str,
    duration_s: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    reg = TUKTUK_REGIMES[regime_name]
    vc = RICKSHAW_CLASSES[vclass_name]

    chems, probs = zip(*vc["chemistry_probs"])
    chemistry = str(rng.choice(chems, p=probs))

    cap_kwh = float(rng.uniform(*vc["cap_kwh_range"]))
    if chemistry == "LFP":
        v_nom = vc["v_nom_lfp"]
        r_nom_base = vc["r_nom_lfp"]
    elif chemistry == "NMC":
        v_nom = vc["v_nom_nmc"]
        r_nom_base = vc["r_nom_nmc"]
    else:
        v_nom = vc["v_nom_la"]
        r_nom_base = vc["r_nom_la"]

    soh = float(rng.uniform(0.75, 1.00))
    pack_cap_ah_nom = (cap_kwh * 1000.0) / v_nom

    curb_mass = float(rng.uniform(*vc["curb_mass_range"]))
    is_cargo = vclass_name == "l5n_ecargo_loader"
    if is_cargo:
        passenger_count = 1
        payload_init = float(rng.uniform(*vc["payload_range"]))
    else:
        max_pax = 5 if vclass_name == "l3_erickshaw_passenger" else 4
        passenger_count = int(rng.integers(1, max_pax + 2))
        luggage_kg = float(rng.uniform(5.0, 55.0))
        payload_init = min(
            vc["payload_range"][1],
            passenger_count * float(rng.uniform(58.0, 76.0)) + luggage_kg,
        )

    rain_flaps_or_roof_rack = (
        1 if (regime_name == "monsoon_waterlogged_roads" or rng.random() < 0.30) else 0
    )
    aero_mult = rng.uniform(1.06, 1.18) if rain_flaps_or_roof_rack else 1.0
    cd = float(rng.uniform(*vc["cd_range"])) * aero_mult
    area = float(rng.uniform(*vc["area_range"]))
    cd_a = cd * area

    tire_pressure_factor = float(rng.uniform(0.92, 1.24))
    crr_trip = reg["crr_base"] * tire_pressure_factor
    surface = reg["surface"]

    elev = float(rng.uniform(*reg["elev0_range"]))
    elev0 = elev
    t_amb0 = float(rng.uniform(*reg["t_amb_range"]))
    rh = float(rng.uniform(*reg["rh_range"]))
    wind_base = float(rng.uniform(*reg["wind_range"]))

    soc = (
        float(rng.uniform(0.55, 0.96))
        if regime_name == "hill_town_steep_descent"
        else float(rng.uniform(0.35, 0.98))
    )

    if t_amb0 < 12.0:
        t_cell = t_amb0 + float(rng.uniform(0.5, 4.0))
    elif t_amb0 > 38.0:
        t_cell = t_amb0 + float(rng.uniform(1.5, 6.5))
    else:
        t_cell = t_amb0 + float(rng.uniform(1.5, 6.0))

    # Synthetic Indian 3W speed profile
    t = np.arange(duration_s, dtype=np.float64)
    v_max = vc["v_max_mps"]
    v_base = v_max * float(rng.uniform(*reg["v_frac_range"]))

    p1 = rng.uniform(130, 290)
    p2 = rng.uniform(35, 90)
    p3 = rng.uniform(14, 32)
    v_raw = (
        v_base
        + 0.30 * v_base * np.sin(2.0 * np.pi * t / p1 + rng.uniform(0, 6.28))
        + 0.24 * v_base * np.sin(2.0 * np.pi * t / p2 + rng.uniform(0, 6.28))
        + 0.12 * v_base * np.cos(2.0 * np.pi * t / p3 + rng.uniform(0, 6.28))
    )

    # Traffic stops & dynamic passenger boarding/alighting
    stop_prob = reg["stop_prob"]
    n_stops = max(3, int(duration_s * stop_prob / 55.0))
    stop_indices = []
    for _ in range(n_stops):
        start = int(rng.integers(25, duration_s - 45))
        dur = int(rng.integers(8, 38 if stop_prob > 0.35 else 24))
        v_raw[start : start + dur] = 0.0
        stop_indices.append(start + dur // 2)

    payload_prof = np.full(duration_s, payload_init, dtype=np.float64)
    pax_prof = np.full(duration_s, passenger_count, dtype=np.int8)
    if not is_cargo and stop_indices:
        cur_pax = passenger_count
        cur_pay = payload_init
        max_pax = 5 if vclass_name == "l3_erickshaw_passenger" else 4
        for s_idx in sorted(set(stop_indices)):
            if rng.random() < 0.45:
                delta_pax = int(rng.choice([-2, -1, 1, 2]))
                new_pax = int(np.clip(cur_pax + delta_pax, 1, max_pax + 1))
                cur_pay = np.clip(
                    cur_pay + (new_pax - cur_pax) * float(rng.uniform(58.0, 74.0)),
                    65.0,
                    vc["payload_range"][1],
                )
                cur_pax = new_pax
                payload_prof[s_idx:] = cur_pay
                pax_prof[s_idx:] = cur_pax

    # Speed breakers & pothole slowdowns
    n_bumps = int(rng.integers(10, 28))
    for _ in range(n_bumps):
        idx = int(rng.integers(12, duration_s - 15))
        v_raw[idx : idx + 6] *= rng.uniform(0.20, 0.50)

    v_clipped = np.clip(v_raw, 0.0, v_max)
    pad = 3
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    v_prof = np.convolve(np.pad(v_clipped, pad, mode="edge"), kernel, mode="valid")
    v_prof[v_prof < 0.25] = 0.0
    a_prof = np.clip(np.gradient(v_prof), -2.5, 2.0)

    # Road grade profile
    gmode = reg["grade_mode"]
    g_p1 = rng.uniform(220, 440)
    g_p2 = rng.uniform(65, 140)
    ph1, ph2 = rng.uniform(0, 6.28), rng.uniform(0, 6.28)

    if gmode == "hill_climb":
        max_g = 0.085 if vclass_name == "l3_erickshaw_passenger" else 0.135
        grade_prof = np.clip(
            0.048 + 0.035 * np.sin(2 * np.pi * t / g_p1 + ph1) + 0.020 * np.cos(2 * np.pi * t / g_p2 + ph2),
            -0.02,
            max_g,
        )
    elif gmode == "hill_descent":
        min_g = -0.085 if vclass_name == "l3_erickshaw_passenger" else -0.125
        grade_prof = np.clip(
            -0.048 + 0.030 * np.sin(2 * np.pi * t / g_p1 + ph1) + 0.018 * np.cos(2 * np.pi * t / g_p2 + ph2),
            min_g,
            0.02,
        )
    elif gmode == "flyover_ramps":
        grade_prof = np.clip(
            0.042 * np.sin(2 * np.pi * t / 150.0 + ph1) + 0.022 * np.sin(2 * np.pi * t / 55.0 + ph2),
            -0.068,
            0.068,
        )
    elif gmode == "rolling_rural":
        grade_prof = np.clip(
            0.028 * np.sin(2 * np.pi * t / g_p1 + ph1) + 0.016 * np.cos(2 * np.pi * t / g_p2 + ph2),
            -0.055,
            0.055,
        )
    elif gmode == "gentle_urban":
        grade_prof = np.clip(
            0.016 * np.sin(2 * np.pi * t / g_p1 + ph1) + 0.010 * np.cos(2 * np.pi * t / g_p2 + ph2),
            -0.038,
            0.038,
        )
    else:
        grade_prof = np.clip(
            0.010 * np.sin(2 * np.pi * t / g_p1 + ph1) + 0.012 * np.cos(2 * np.pi * t / g_p2 + ph2),
            -0.032,
            0.032,
        )

    # Pre-generate sensor noise
    n_elev = rng.normal(0.0, 0.50, duration_s)
    n_grade = rng.normal(0.0, 0.14, duration_s)
    n_tamb = rng.normal(0.0, 0.18, duration_s)
    n_tcell = rng.normal(0.0, 0.15, duration_s)
    n_curr = rng.normal(0.0, 0.25, duration_s)
    n_volt = rng.normal(0.0, 0.06, duration_s)
    wind_gust = 1.6 * np.sin(2.0 * np.pi * t / rng.uniform(45, 95) + rng.uniform(0, 6.28)) + rng.normal(0.0, 0.4, duration_s)
    crr_jitter = rng.normal(0.0, 0.0009, duration_s)

    # Pre-allocate output arrays
    out_dist = np.empty(duration_s, dtype=np.float64)
    out_elev = np.empty(duration_s, dtype=np.float64)
    out_grade = np.empty(duration_s, dtype=np.float64)
    out_cum_gain = np.empty(duration_s, dtype=np.float64)
    out_crr = np.empty(duration_s, dtype=np.float64)
    out_wind = np.empty(duration_s, dtype=np.float64)
    out_rho = np.empty(duration_s, dtype=np.float64)
    out_tamb = np.empty(duration_s, dtype=np.float64)
    out_tcell = np.empty(duration_s, dtype=np.float64)
    out_aux_w = np.empty(duration_s, dtype=np.float64)
    out_rint = np.empty(duration_s, dtype=np.float64)
    out_volt = np.empty(duration_s, dtype=np.float64)
    out_curr = np.empty(duration_s, dtype=np.float64)
    out_pow_kw = np.empty(duration_s, dtype=np.float64)
    out_wh_km = np.empty(duration_s, dtype=np.float64)
    out_soc = np.empty(duration_s, dtype=np.float64)
    out_derate = np.empty(duration_s, dtype=np.int8)

    g = 9.81
    rot_inertia = 1.06
    eta_drive = vc["eta_drive"]
    eta_regen = vc["eta_regen"]
    r_nom_aged = r_nom_base * (1.0 + 2.0 * (1.0 - soh))
    c_thermal = vc["c_thermal"]
    max_p_w = vc["max_power_w"]
    max_chg_a = vc["max_charge_a"]
    max_dis_a = vc["max_discharge_a"]
    solar_w = reg["solar_heat_w"]

    cum_elev_gain = 0.0
    cum_dist_m = 0.0

    for step in range(duration_s):
        v = float(v_prof[step])
        a = float(a_prof[step])
        grade = float(grade_prof[step])
        total_mass = curb_mass + float(payload_prof[step])

        theta = math.atan(grade)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)

        crr_now = max(0.009, crr_trip + float(crr_jitter[step]))
        t_amb = t_amb0 - 0.0065 * (elev - elev0) + 0.5 * math.sin(2.0 * math.pi * step / 3600.0)
        v_wind = wind_base + float(wind_gust[step])
        pressure = 101325.0 * math.exp(-elev / 8500.0)
        rho = pressure / (287.05 * (t_amb + 273.15))

        f_grade = total_mass * g * sin_t
        f_roll = (crr_now * total_mass * g * cos_t) if v > 0.1 else 0.0
        v_rel = v + v_wind
        f_aero = 0.5 * rho * cd_a * v_rel * abs(v_rel)
        f_inert = total_mass * rot_inertia * a
        p_wheel = (f_grade + f_roll + f_aero + f_inert) * v

        # Motor controller power clipping on steep climbs
        if p_wheel > max_p_w * eta_drive and v > 1.5:
            scale_v = (max_p_w * eta_drive) / max(p_wheel, 1.0)
            v = max(1.5, v * (0.5 + 0.5 * scale_v))
            v_prof[step] = v
            f_aero = 0.5 * rho * cd_a * (v + v_wind) * abs(v + v_wind)
            p_wheel = min(max_p_w * eta_drive, (f_grade + f_roll + f_aero + 0.3 * f_inert) * v)

        cum_dist_m += v
        dh = v * sin_t
        elev = max(2.0, elev + dh)
        if dh > 0.0:
            cum_elev_gain += dh

        # 3W 12V auxiliary load (no cabin AC)
        p_aux = 55.0
        if t_amb > 34.0:
            p_aux += 35.0
        if surface == "wet_waterlogged_potholes" or regime_name == "north_india_winter_morning":
            p_aux += 65.0

        # Battery power demand with regen throttling
        if p_wheel >= 0.0:
            p_batt = (p_wheel / eta_drive) + p_aux
        else:
            regen_cap = 1.0
            if soc > 0.92:
                regen_cap *= max(0.0, (1.0 - soc) / 0.08)
            if t_cell < 6.0:
                regen_cap *= max(0.10, (t_cell + 10.0) / 16.0)
            if chemistry == "Lead-Acid":
                regen_cap *= 0.25
            p_batt = (p_wheel * eta_regen * regen_cap) + p_aux

        # Temperature & SoC dependent internal resistance
        t_coeff = 0.031 if chemistry == "LFP" else (0.026 if chemistry == "NMC" else 0.036)
        r_temp = math.exp(t_coeff * (25.0 - t_cell))
        r_soc = 1.0 + 0.40 * math.exp(-9.0 * soc) + 0.12 * math.exp(-10.0 * (1.0 - soc))
        r_int = r_nom_aged * r_temp * r_soc

        v_ocv = ocv_3w(soc, v_nom, chemistry)

        # BMS thermal / low-SoC derating
        derate = 0
        p_limit = min(max_p_w / eta_drive, 0.85 * (v_ocv * v_ocv) / (4.0 * r_int))
        t_derate_thresh = 55.0 if chemistry == "LFP" else (50.0 if chemistry == "NMC" else 48.0)

        if t_cell > t_derate_thresh:
            derate = 1
            p_limit *= max(0.35, 1.0 - 0.08 * (t_cell - t_derate_thresh))
        if soc < 0.10:
            derate = 1
            p_limit *= max(0.30, soc / 0.10)
        if p_batt > p_limit:
            p_batt = p_limit
            derate = 1

        disc = v_ocv * v_ocv - 4.0 * r_int * p_batt
        if disc < 0.0:
            disc = 0.0
            derate = 1
        curr = (v_ocv - math.sqrt(disc)) / (2.0 * r_int)

        if curr > max_dis_a:
            curr = max_dis_a
            derate = 1
        elif curr < -max_chg_a:
            curr = -max_chg_a

        v_pack = v_ocv - curr * r_int
        p_actual = v_pack * curr

        # SoC integration (with Peukert effect for Lead-Acid)
        cold_cap_factor = 1.0 - (0.25 if chemistry == "Lead-Acid" else 0.20) * max(0.0, (15.0 - t_cell) / 25.0)
        peukert_mult = (curr / 15.0) ** 0.16 if (chemistry == "Lead-Acid" and curr > 15.0) else 1.0
        eff_cap_ah = pack_cap_ah_nom * soh * cold_cap_factor
        soc -= (curr * peukert_mult) / (eff_cap_ah * 3600.0)
        if soc < 0.02:
            soc = 0.02
            derate = 1
        elif soc > 1.0:
            soc = 1.0

        # Passive air-cooled thermal balance
        q_joule = curr * curr * r_int
        h_passive = (4.8 + 0.60 * v) * (rho / 1.18)
        q_passive = h_passive * (t_cell - t_amb)
        q_solar = solar_w * (0.25 if v < 3.0 else 0.12)
        t_cell += (q_joule + q_solar - q_passive) / c_thermal

        wh_km = (p_actual / (v * 3.6)) if v > 0.8 else 0.0

        out_dist[step] = cum_dist_m / 1000.0
        out_elev[step] = elev + float(n_elev[step])
        out_grade[step] = grade * 100.0 + float(n_grade[step])
        out_cum_gain[step] = cum_elev_gain
        out_crr[step] = crr_now
        out_wind[step] = v_wind
        out_rho[step] = rho
        out_tamb[step] = t_amb + float(n_tamb[step])
        out_tcell[step] = t_cell + float(n_tcell[step])
        out_aux_w[step] = p_aux
        out_rint[step] = r_int * 1000.0
        out_volt[step] = v_pack + float(n_volt[step])
        out_curr[step] = curr + float(n_curr[step])
        out_pow_kw[step] = p_actual / 1000.0
        out_wh_km[step] = np.clip(wh_km, -150.0, 350.0)
        out_soc[step] = soc * 100.0
        out_derate[step] = derate

    return pd.DataFrame({
        "trip_id": np.full(duration_s, trip_id, dtype=np.int16),
        "time_s": np.arange(duration_s, dtype=np.int16),
        "distance_km": np.round(out_dist, 3),
        "road_regime": regime_name,
        "rickshaw_class": vclass_name,
        "battery_chemistry": chemistry,
        "nominal_voltage_v": v_nom,
        "battery_capacity_kwh": round(cap_kwh, 2),
        "soh_pct": round(soh * 100.0, 2),
        "road_surface": surface,
        "elevation_m": np.round(out_elev, 2),
        "road_grade_pct": np.round(out_grade, 2),
        "cumulative_elev_gain_m": np.round(out_cum_gain, 1),
        "curb_weight_kg": round(curb_mass, 1),
        "passenger_count": pax_prof,
        "payload_weight_kg": np.round(payload_prof, 1),
        "total_vehicle_mass_kg": np.round(curb_mass + payload_prof, 1),
        "rain_flaps_or_rack": np.full(duration_s, rain_flaps_or_roof_rack, dtype=np.int8),
        "drag_coeff_cd": round(cd, 3),
        "frontal_area_m2": round(area, 2),
        "drag_area_cda_m2": round(cd_a, 3),
        "rolling_resistance_crr": np.round(out_crr, 4),
        "headwind_mps": np.round(out_wind, 2),
        "air_density_kgm3": np.round(out_rho, 3),
        "speed_kmh": np.round(v_prof * 3.6, 2),
        "accel_mps2": np.round(a_prof, 3),
        "ambient_temp_c": np.round(out_tamb, 2),
        "relative_humidity_pct": round(rh, 1),
        "battery_temp_c": np.round(out_tcell, 2),
        "aux_load_w": np.round(out_aux_w, 1),
        "internal_resistance_mohm": np.round(out_rint, 2),
        "pack_voltage_v": np.round(out_volt, 2),
        "pack_current_a": np.round(out_curr, 2),
        "pack_power_kw": np.round(out_pow_kw, 3),
        "energy_rate_wh_per_km": np.round(out_wh_km, 1),
        "soc_pct": np.round(out_soc, 3),
        "power_derating_flag": out_derate,
    })


if __name__ == "__main__":
    t0 = time.time()
    rng = np.random.default_rng(20263)

    TRIPS_PER_COMBO = 4   # 8 regimes x 4 classes x 4 trips = 128 trips (256,000 rows)
    DURATION_S = 2000     # Change TRIPS_PER_COMBO to 16 for ~1M rows

    frames = []
    trip_id = 1
    for reg_name in TUKTUK_REGIMES:
        for vc_name in RICKSHAW_CLASSES:
            for _ in range(TRIPS_PER_COMBO):
                frames.append(simulate_tuktuk_trip(trip_id, reg_name, vc_name, DURATION_S, rng))
                trip_id += 1

    df = pd.concat(frames, ignore_index=True)
    output_csv = "indian_ev_rickshaw_tuktuk_bms_dataset.csv"
    df.to_csv(output_csv, index=False)

    size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print(f"Saved {output_csv} ({size_mb:.2f} MB) | {len(df):,} rows in {time.time() - t0:.2f}s")