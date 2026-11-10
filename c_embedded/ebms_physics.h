/**
 * @file ebms_physics.h
 * @brief Zero-dependency C99 embedded port of ebms-sim physics equations
 *        designed for 32-bit MCUs (ESP32 / STM32F4 / Cortex-M4/M7).
 *
 * All functions are pure, re-entrant, thread-safe, and allocate no dynamic memory.
 */

#ifndef EBMS_PHYSICS_H
#define EBMS_PHYSICS_H

#ifdef __cplusplus
extern "C" {
#endif

#define EBMS_GRAVITY_MS2          9.80665f
#define EBMS_AIR_DENSITY_STD      1.2041f

typedef enum {
    EBMS_CHEM_LEAD_ACID = 0,
    EBMS_CHEM_LFP       = 1
} ebms_chemistry_t;

/**
 * @brief Vehicle parameters struct (all values in SI units).
 */
typedef struct {
    float curb_mass_kg;
    float driver_mass_kg;
    float crr;
    float cda_m2;
    float eta_drive;
    float eta_regen;
    float aux_w;
    float rotational_inertia_factor;
    float max_speed_kmh;
} ebms_vehicle_params_t;

/**
 * @brief Battery parameters struct.
 */
typedef struct {
    ebms_chemistry_t chemistry;
    float nominal_voltage_v;
    float nominal_capacity_ah;
    float r0_ohm;
    float peukert_exp;
    float peukert_i_ref_a;
    float temp_coeff_r;
    float soh_coeff_r;
    float cutoff_soc;
    float max_continuous_current_a;
} ebms_battery_params_t;

/* --- Vehicle Dynamics --- */

float ebms_rolling_resistance_force_n(float mass_kg, float crr, float grade_fraction);
float ebms_aero_drag_force_n(float v_ms, float headwind_ms, float cda_m2, float air_density_kg_m3);
float ebms_grade_force_n(float mass_kg, float grade_fraction);
float ebms_inertial_force_n(float mass_kg, float acceleration_ms2, float rotational_inertia_factor);
float ebms_wheel_power_w(float total_force_n, float v_ms);
float ebms_battery_power_w(float wheel_pwr_w, float eta_drive, float eta_regen, float aux_w);

/* --- Battery Equivalent Circuit Model --- */

float ebms_ocv_from_soc(float soc, ebms_chemistry_t chemistry);
float ebms_usable_capacity_factor(float temp_c);
float ebms_internal_resistance_ohm(float r0_base_ohm, float temp_c, float soh, float temp_coeff_r, float soh_coeff_r);
float ebms_peukert_effective_current_a(float current_a, float peukert_exp, float peukert_i_ref_a);
float ebms_terminal_voltage_v(float ocv_v, float current_a, float r_ohm);
void  ebms_solve_current_and_voltage(float power_w, float ocv_v, float r_ohm, float *out_curr_a, float *out_volt_v);
float ebms_update_soc_coulomb_counting(float soc, float current_a, float dt_s, float nominal_capacity_ah, float temp_c, float peukert_exp, float peukert_i_ref_a, float charge_efficiency);
float ebms_estimate_remaining_range_km(float current_soc, float cutoff_soc, float nominal_capacity_ah, float nominal_voltage_v, float temp_c, float wh_per_km);

/* --- Lumped Thermal Model --- */

float ebms_joule_heat_generation_w(float current_a, float r_ohm);
float ebms_effective_thermal_resistance(float r_th_base, float v_ms, float cooling_factor);
float ebms_step_pack_temperature_c(float t_pack_c, float current_a, float r_ohm, float t_ambient_c, float c_th_j_per_k, float r_th_k_per_w, float dt_s, float v_ms, float cooling_factor);

/* --- Smart Charging --- */

float ebms_temperature_charge_multiplier(float temp_c);
float ebms_max_charge_current_a(float temp_c, float soc, float nominal_capacity_ah, float max_charge_c_rate, float taper_soc, float min_charge_current_a);

#ifdef __cplusplus
}
#endif

#endif /* EBMS_PHYSICS_H */

