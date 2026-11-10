"""Pure Plotly figure builders for the ebms-sim Streamlit application.

All functions are pure: data in, plotly Figure out. Zero Streamlit calls or I/O.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_trip_simulation_figure(df: pd.DataFrame, title_suffix: str = "") -> go.Figure:
    """Build a 4-row synchronized interactive dashboard of the trip time-series."""
    time_min = df["time_s"] / 60.0

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Speed & Road Elevation",
            "Wheel Mechanical Power & Battery Electrical Power",
            "Terminal Voltage & Battery Current",
            "State of Charge (SOC) & Pack Temperature",
        ),
        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    # 1. Speed & Elevation
    fig.add_trace(
        go.Scatter(x=time_min, y=df["speed_kmh"], name="Speed (km/h)", line=dict(color="#1f77b4", width=2)),
        row=1, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=time_min, y=df["elevation_m"], name="Elevation (m)", line=dict(color="#8c564b", width=2, dash="dot")),
        row=1, col=1, secondary_y=True,
    )

    # 2. Power
    fig.add_trace(
        go.Scatter(x=time_min, y=df["wheel_power_w"], name="Wheel Power (W)", line=dict(color="#17becf", width=1.5)),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_min, y=df["battery_power_w"], name="Battery Power (W)", line=dict(color="#d62728", width=2)),
        row=2, col=1,
    )

    # 3. Voltage & Current
    fig.add_trace(
        go.Scatter(x=time_min, y=df["voltage_v"], name="Voltage (V)", line=dict(color="#9467bd", width=2)),
        row=3, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=time_min, y=df["current_a"], name="Current (A)", line=dict(color="#ff7f0e", width=2)),
        row=3, col=1, secondary_y=True,
    )

    # 4. SOC & Temperature
    fig.add_trace(
        go.Scatter(x=time_min, y=df["soc"] * 100.0, name="SOC (%)", line=dict(color="#2ca02c", width=2.5)),
        row=4, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=time_min, y=df["temp_c"], name="Pack Temp (°C)", line=dict(color="#e377c2", width=2, dash="dash")),
        row=4, col=1, secondary_y=True,
    )

    # Layout enhancements
    fig.update_xaxes(title_text="Trip Elapsed Time [minutes]", row=4, col=1)
    fig.update_yaxes(title_text="Speed [km/h]", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Elevation [m]", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Power [W]", row=2, col=1)
    fig.update_yaxes(title_text="Voltage [V]", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Current [A]", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="SOC [%]", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Temp [°C]", row=4, col=1, secondary_y=True)

    fig.update_layout(
        height=850,
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template="plotly_white",
    )
    return fig


def build_speed_sweep_figure(
    speeds: List[float],
    wh_per_km: List[float],
    best_speed: float,
    min_wh: float,
    constrained_speed: Optional[float] = None,
) -> go.Figure:
    """Build Plotly figure for Wh/km vs cruising speed curve."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=speeds,
        y=wh_per_km,
        mode="lines",
        name="Consumption (Wh/km)",
        line=dict(color="#1f77b4", width=3),
    ))

    # Mark unconstrained optimum
    fig.add_trace(go.Scatter(
        x=[best_speed],
        y=[min_wh],
        mode="markers+text",
        name=f"Unconstrained Best: {best_speed:.1f} km/h",
        text=[f"  Optimum: {best_speed:.1f} km/h ({min_wh:.1f} Wh/km)"],
        textposition="top right",
        marker=dict(color="#2ca02c", size=12, symbol="star"),
    ))

    if constrained_speed is not None and abs(constrained_speed - best_speed) > 0.5:
        # Mark constrained speed
        c_wh = np.interp(constrained_speed, speeds, wh_per_km)
        fig.add_trace(go.Scatter(
            x=[constrained_speed],
            y=[c_wh],
            mode="markers+text",
            name=f"Time-Constrained: {constrained_speed:.1f} km/h",
            text=[f"  Target: {constrained_speed:.1f} km/h ({c_wh:.1f} Wh/km)"],
            textposition="bottom right",
            marker=dict(color="#d62728", size=11, symbol="diamond"),
        ))

    fig.update_layout(
        title="Energy Consumption vs Cruising Speed (10 - 25 km/h)",
        xaxis_title="Cruising Speed [km/h] (Legal cap: 25 km/h)",
        yaxis_title="Specific Energy Consumption [Wh/km]",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig


def build_range_uncertainty_gauge(
    nominal_km: float,
    min_km: float,
    max_km: float,
) -> go.Figure:
    """Build bullet/range bar displaying nominal range with uncertainty interval."""
    fig = go.Figure()

    # Uncertainty span bar
    fig.add_trace(go.Bar(
        x=[max_km - min_km],
        y=["Estimated Range"],
        base=[min_km],
        orientation="h",
        name="Uncertainty Range",
        marker=dict(color="#a1d99b"),
        hovertemplate="Span: %{base:.1f} km to %{x:.1f} km<extra></extra>",
    ))

    # Nominal point marker
    fig.add_trace(go.Scatter(
        x=[nominal_km],
        y=["Estimated Range"],
        mode="markers+text",
        text=[f" <b>{nominal_km:.1f} km</b>"],
        textposition="top center",
        marker=dict(color="#006d2c", size=16, symbol="line-ns", line=dict(width=4)),
        name="Nominal Prediction",
    ))

    fig.update_layout(
        title="Predicted Usable Range with Dynamic Uncertainty Band",
        xaxis_title="Driving Range [km]",
        height=220,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=30),
        showlegend=True,
    )
    return fig


def build_hills_comparison_bar(
    flat_wh: float,
    hilly_no_regen_wh: float,
    hilly_regen_wh: float,
    regen_savings_wh: float,
) -> go.Figure:
    """Build comparison bar chart of flat vs hilly with/without regen."""
    categories = ["Flat Road", "Hilly (No Regen)", "Hilly (With Regen)"]
    values = [flat_wh, hilly_no_regen_wh, hilly_regen_wh]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f} Wh/km" for v in values],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title="Topography & Regenerative Braking Energy Impact",
        yaxis_title="Specific Energy Consumption [Wh/km]",
        template="plotly_white",
        height=380,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig


def build_smart_charging_dashboard(
    temp_multiplier_df: pd.DataFrame,
    charge_session_df: pd.DataFrame,
    operating_temp_c: float,
    chemistry: str,
) -> go.Figure:
    """Build 2-column charging analysis figure."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Thermal Current Derating Curve ({chemistry.upper()})",
            "Simulated Charge Session Trace",
        ),
    )

    # 1. Thermal derating
    fig.add_trace(
        go.Scatter(
            x=temp_multiplier_df["temp_c"],
            y=temp_multiplier_df["multiplier"],
            mode="lines",
            name="Current Limit Multiplier",
            line=dict(color="#d62728", width=2.5),
        ),
        row=1, col=1,
    )
    # Current operating point
    op_mult = float(np.interp(operating_temp_c, temp_multiplier_df["temp_c"], temp_multiplier_df["multiplier"]))
    fig.add_trace(
        go.Scatter(
            x=[operating_temp_c],
            y=[op_mult],
            mode="markers+text",
            text=[f"  Operating: {operating_temp_c:.1f}°C ({op_mult:.2f}x)"],
            textposition="top center",
            marker=dict(color="#1f77b4", size=12, symbol="circle"),
            name="Current Operating Temp",
        ),
        row=1, col=1,
    )

    # 2. Charge session trace
    fig.add_trace(
        go.Scatter(
            x=charge_session_df["time_hours"],
            y=charge_session_df["soc"] * 100.0,
            mode="lines",
            name="SOC (%)",
            line=dict(color="#2ca02c", width=2.5),
        ),
        row=1, col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=charge_session_df["time_hours"],
            y=charge_session_df["charge_current_a"],
            mode="lines",
            name="Charge Current (A)",
            line=dict(color="#ff7f0e", width=2),
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Pack Temperature [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Current Multiplier [0..1]", row=1, col=1)
    fig.update_xaxes(title_text="Charging Time [hours]", row=1, col=2)
    fig.update_yaxes(title_text="SOC [%] / Current [A]", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0.5, xanchor="center"),
    )
    return fig


def build_scenario_comparison_figure(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str = "Scenario A",
    label_b: str = "Scenario B",
) -> go.Figure:
    """Overlay Scenario A and B on speed, power, and SOC."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Speed Comparison [km/h]",
            "Battery Electrical Power [W]",
            "State of Charge (SOC) [%]",
        ),
    )

    t_a = df_a["time_s"] / 60.0
    t_b = df_b["time_s"] / 60.0

    # 1. Speed
    fig.add_trace(go.Scatter(x=t_a, y=df_a["speed_kmh"], name=f"{label_a} Speed", line=dict(color="#1f77b4", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_b, y=df_b["speed_kmh"], name=f"{label_b} Speed", line=dict(color="#ff7f0e", width=2, dash="dash")), row=1, col=1)

    # 2. Power
    fig.add_trace(go.Scatter(x=t_a, y=df_a["battery_power_w"], name=f"{label_a} Power", line=dict(color="#1f77b4", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_b, y=df_b["battery_power_w"], name=f"{label_b} Power", line=dict(color="#ff7f0e", width=1.5, dash="dash")), row=2, col=1)

    # 3. SOC
    fig.add_trace(go.Scatter(x=t_a, y=df_a["soc"] * 100.0, name=f"{label_a} SOC", line=dict(color="#1f77b4", width=2.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=t_b, y=df_b["soc"] * 100.0, name=f"{label_b} SOC", line=dict(color="#ff7f0e", width=2.5, dash="dash")), row=3, col=1)

    fig.update_xaxes(title_text="Time [minutes]", row=3, col=1)
    fig.update_layout(
        template="plotly_white",
        height=650,
        margin=dict(l=30, r=30, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0.5, xanchor="center"),
    )
    return fig


def build_ai_scatter_parity_figure(
    true_wh: np.ndarray,
    pred_wh: np.ndarray,
    mape: float,
) -> go.Figure:
    """Build 45-degree parity plot comparing XGBoost surrogate vs First-Principles Physics."""
    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=true_wh,
        y=pred_wh,
        mode="markers",
        name="1,000 Random Scenarios",
        marker=dict(color="#1f77b4", size=5, opacity=0.6),
    ))

    # 45-degree parity line
    min_v = float(min(np.min(true_wh), np.min(pred_wh)))
    max_v = float(max(np.max(true_wh), np.max(pred_wh)))
    fig.add_trace(go.Scatter(
        x=[min_v, max_v],
        y=[min_v, max_v],
        mode="lines",
        name="Perfect Agreement (y = x)",
        line=dict(color="#d62728", width=2, dash="dash"),
    ))

    fig.update_layout(
        title=f"AI Approximator Parity Plot: XGBoost vs Physics Engine (MAPE: {mape:.2f}%)",
        xaxis_title="Ground-Truth Physics Consumption [Wh/km]",
        yaxis_title="XGBoost Surrogate Prediction [Wh/km]",
        template="plotly_white",
        height=480,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    return fig

