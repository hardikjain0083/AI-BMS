# Changelog

## Phase 1

- Added opt-in load-dependent `loss_based` drivetrain efficiency while keeping constant efficiency as the default.
- Added low-power loss behavior and calibration checks around 700 W wheel power.
- Added chemistry-specific lead-acid charging with temperature-compensated absorption voltage, 24-cell pack support, 0.10-0.20 C limits, and a 50°C cutoff.
- Kept the LFP bell-curve charging model separate from lead-acid charging.
- Documented the CC-CV taper as a linear approximation.
- Added and verified five pure sanity checks covering energy, monotonicity, regeneration, range, and charging temperature limits.
- Corrected speed-optimum documentation and removed misleading `high-fidelity` wording.

## Phase 2

- Completed the Streamlit dashboard layout with sidebar controls, guided demo steps, nine analysis tabs, Plotly charts, scenario comparison, AI fallback messaging, and live health checks.
- Preserved full vehicle and battery preset fields when UI overrides are applied.
- Made model and dataset discovery relative to the repository rather than the process working directory.
- Preserved all vehicle parameters when comparing regen and no-regen routes.
- Added AppTest-compatible imports for both direct Streamlit execution and file-based smoke tests.
- Added README instructions for launching the app, a screenshot placeholder, and a five-minute teaching script.