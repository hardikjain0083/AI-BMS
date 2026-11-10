"""Demonstration script for ebms-sim.

Generates three visual demonstration artifacts:
1. One complete multi-panel trip simulation plot (speed, elevation, power, V, I, SOC, temp).
2. Wh/km vs. Speed U-curve showing optimal cruising speed.
3. Smart charging bell-curve profile and thermal evolution.
"""

from __future__ import annotations
import os
import tempfile

# Set non-interactive backend and writable cache for matplotlib
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ebms_sim.config import PRESETS
from ebms_sim.profiles import generate_speed_profile, generate_gradient_profile
from ebms_sim.trip import simulate_trip
from ebms_sim.optimizer import sweep_speed_wh_per_km
from ebms_sim.charging import (
    simulate_charge,
    temperature_charge_multiplier,
    max_charge_current_a,
)


def plot_trip_demo(save_path: str = "demo_trip_plot.png"):
    """Simulate and plot a realistic 10-minute mixed-route e-rickshaw trip."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    duration_s = 600

    speed_profile = generate_speed_profile(
        duration_s=duration_s,
        speed_cap_kmh=24.0,
        stop_interval_s=60.0,
        stop_duration_s=10.0,
        aggressiveness=1.0,
        seed=42,
    )
    gradient_profile = generate_gradient_profile(
        length_steps=duration_s,
        route_type="mixed",
        seed=42,
    )

    res = simulate_trip(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        thermal_params=preset["thermal"],
        speed_profile_ms=speed_profile,
        gradient_profile=gradient_profile,
        payload_mass_kg=195.0,  # 3 passengers
        t_ambient_c=32.0,
        headwind_ms=1.5,
        start_soc=0.85,
        dt_s=1.0,
    )

    df = res.time_series
    time_min = df["time_s"] / 60.0

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    fig.suptitle(
        f"ebms-sim: 10-Minute E-Rickshaw Trip Simulation\n"
        f"Distance: {res.summary.distance_km:.2f} km | Energy: {res.summary.total_energy_wh:.1f} Wh | "
        f"Consumption: {res.summary.wh_per_km:.1f} Wh/km | Est. Range: {res.summary.range_remaining_km:.1f} km",
        fontsize=12,
        fontweight="bold",
    )

    # 1. Speed and Elevation
    ax1 = axes[0]
    ax1.plot(time_min, df["speed_kmh"], color="tab:blue", label="Speed (km/h)", lw=1.5)
    ax1.set_ylabel("Speed [km/h]", color="tab:blue", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_min, df["elevation_m"], color="tab:brown", linestyle=":", label="Elevation (m)", lw=2.0)
    ax1_twin.set_ylabel("Elevation [m]", color="tab:brown", fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="tab:brown")

    # 2. Wheel and Battery Power
    ax2 = axes[1]
    ax2.plot(time_min, df["wheel_power_w"], color="tab:cyan", label="Wheel Power", lw=1.2, alpha=0.8)
    ax2.plot(time_min, df["battery_power_w"], color="tab:red", label="Battery Power", lw=1.5)
    ax2.axhline(0.0, color="black", linestyle="--", lw=0.8)
    ax2.set_ylabel("Power [W]", fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # 3. Terminal Voltage and Battery Current
    ax3 = axes[2]
    ax3.plot(time_min, df["voltage_v"], color="tab:purple", label="Terminal Voltage (V)", lw=1.5)
    ax3.set_ylabel("Voltage [V]", color="tab:purple", fontweight="bold")
    ax3.tick_params(axis="y", labelcolor="tab:purple")
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(time_min, df["current_a"], color="tab:orange", label="Current (A)", lw=1.5)
    ax3_twin.set_ylabel("Current [A]", color="tab:orange", fontweight="bold")
    ax3_twin.tick_params(axis="y", labelcolor="tab:orange")

    # 4. State of Charge and Pack Temperature
    ax4 = axes[3]
    ax4.plot(time_min, df["soc"] * 100.0, color="tab:green", label="SOC (%)", lw=2.0)
    ax4.set_ylabel("SOC [%]", color="tab:green", fontweight="bold")
    ax4.tick_params(axis="y", labelcolor="tab:green")
    ax4.set_xlabel("Trip Time [minutes]", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(time_min, df["temp_c"], color="tab:red", linestyle="-.", label="Pack Temp (C)", lw=1.8)
    ax4_twin.set_ylabel("Pack Temp [°C]", color="tab:red", fontweight="bold")
    ax4_twin.tick_params(axis="y", labelcolor="tab:red")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved Trip Plot: {save_path}")


def plot_speed_sweep_demo(save_path: str = "demo_speed_sweep_curve.png"):
    """Plot Wh/km versus speed U-curve illustrating energy-optimal cruising speed."""
    preset = PRESETS["lead_acid_48V_100Ah"]
    speeds = np.linspace(10.0, 25.0, 31)

    spds, wh_flat, best_flat, min_flat = sweep_speed_wh_per_km(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        payload_mass_kg=130.0,
        grade_fraction=0.0,
        headwind_ms=0.0,
        t_ambient_c=25.0,
        speeds_kmh=speeds,
    )

    _, wh_uphill, best_uphill, min_uphill = sweep_speed_wh_per_km(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        payload_mass_kg=130.0,
        grade_fraction=0.03,  # 3% uphill
        headwind_ms=0.0,
        t_ambient_c=25.0,
        speeds_kmh=speeds,
    )

    _, wh_headwind, best_wind, min_wind = sweep_speed_wh_per_km(
        vehicle_params=preset["vehicle"],
        battery_params=preset["battery"],
        payload_mass_kg=130.0,
        grade_fraction=0.0,
        headwind_ms=3.0,  # 3 m/s headwind
        t_ambient_c=25.0,
        speeds_kmh=speeds,
    )

    plt.figure(figsize=(9, 5.5))
    plt.plot(spds, wh_flat, color="tab:blue", lw=2.5, label="Flat Road (Grade 0%)")
    plt.scatter([best_flat], [min_flat], color="tab:blue", s=90, zorder=5,
                label=f"Optimum Flat: {best_flat:.1f} km/h ({min_flat:.1f} Wh/km)")

    plt.plot(spds, wh_uphill, color="tab:red", lw=2.2, linestyle="--", label="Uphill (Grade +3%)")
    plt.scatter([best_uphill], [min_uphill], color="tab:red", s=90, zorder=5,
                label=f"Optimum Uphill: {best_uphill:.1f} km/h ({min_uphill:.1f} Wh/km)")

    plt.plot(spds, wh_headwind, color="tab:green", lw=2.2, linestyle="-.", label="Headwind (+3 m/s)")
    plt.scatter([best_wind], [min_wind], color="tab:green", s=90, zorder=5,
                label=f"Optimum Headwind: {best_wind:.1f} km/h ({min_wind:.1f} Wh/km)")

    plt.title("ebms-sim: Specific Energy Consumption vs Cruising Speed (Wh/km)\n"
              "U-Shaped Curve: High Aux Loss at Low Speed + Aero Drag at High Speed",
              fontsize=11, fontweight="bold")
    plt.xlabel("Cruising Speed [km/h] (Legal cap: 25 km/h)", fontweight="bold")
    plt.ylabel("Energy Consumption [Wh/km]", fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved Speed Sweep Plot: {save_path}")


def plot_charging_demo(save_path: str = "demo_charging_curve.png"):
    """Plot thermal derating bell curve and simulated charge time-series."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Bell-curve thermal derating multiplier
    temp_range = np.linspace(-5.0, 50.0, 200)
    multipliers = [temperature_charge_multiplier(t) for t in temp_range]

    ax1.plot(temp_range, multipliers, color="tab:red", lw=2.5)
    ax1.axvspan(-5.0, 0.0, color="blue", alpha=0.15, label="No Charge (<= 0°C)")
    ax1.axvspan(0.0, 5.0, color="cyan", alpha=0.15, label="Trickle (0-5°C)")
    ax1.axvspan(5.0, 15.0, color="yellow", alpha=0.15, label="Reduced (5-15°C)")
    ax1.axvspan(15.0, 35.0, color="green", alpha=0.15, label="Full Rate (15-35°C)")
    ax1.axvspan(35.0, 45.0, color="orange", alpha=0.15, label="Derated (35-45°C)")
    ax1.axvspan(45.0, 50.0, color="red", alpha=0.15, label="Shutoff (>= 45°C)")

    ax1.set_title("Charge Current Thermal Derating (Bell Curve)", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Cell / Pack Temperature [°C]", fontweight="bold")
    ax1.set_ylabel("Charge Current Multiplier [0..1]", fontweight="bold")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # Right: Charge Session Simulation Trace (LFP battery)
    lfp_preset = PRESETS["lfp_51V_60Ah"]
    charge_sim = simulate_charge(
        battery_params=lfp_preset["battery"],
        thermal_params=lfp_preset["thermal"],
        charging_params=lfp_preset["charging"],
        start_soc=0.15,
        target_soc=0.98,
        t_ambient_c=28.0,
        dt_s=30.0,
    )

    c_df = charge_sim.time_series
    ax2.plot(c_df["time_hours"], c_df["soc"] * 100.0, color="tab:green", lw=2.2, label="SOC [%]")
    ax2.plot(c_df["time_hours"], c_df["charge_current_a"], color="tab:orange", lw=2.0, label="Current [A]")
    ax2.plot(c_df["time_hours"], c_df["temp_c"], color="tab:red", linestyle="--", lw=1.8, label="Pack Temp [°C]")

    ax2.set_title(
        f"Simulated Smart Charge Session (LFP 51.2V 60Ah)\n"
        f"Time to Full: {charge_sim.summary.time_to_full_hours:.2f} hrs | Energy: {charge_sim.summary.energy_charged_wh:.0f} Wh",
        fontweight="bold",
        fontsize=11,
    )
    ax2.set_xlabel("Elapsed Time [hours]", fontweight="bold")
    ax2.set_ylabel("Parameters", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="center right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved Charging Demo Plot: {save_path}")


def main():
    print("Generating demo artifacts...")
    plot_trip_demo("demo_trip_plot.png")
    plot_speed_sweep_demo("demo_speed_sweep_curve.png")
    plot_charging_demo("demo_charging_curve.png")
    print("All demo artifacts successfully created.")


if __name__ == "__main__":
    main()

