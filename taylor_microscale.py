# %% [markdown]
# # Taylor microscale and integral length scale from hot-wire time series
#
# Uses **``mcflow_plotting.hotwire``** for IO and scale estimates.

# %%
from __future__ import annotations

from pathlib import Path

from mcflow_plotting import (
    NU_AIR,
    dissipation_from_time_derivatives,
    epsilon_from_welch_spectrum,
    integral_length_scale_longitudinal,
    load_calibrated_hotwire_velocity,
    taylor_microscale_from_epsilon,
    taylor_microscale_from_series,
)

# %matplotlib inline  # uncomment when running in Jupyter

# %%
DEFAULT_H5 = Path(
    "/workspace/data/hotwire/tti_no_gravity/"
    "hotwire_10kHz_TTI_no_gravity_turbulent_side_pitot_calibrated_hotwire.h5"
)

# %%
cv = load_calibrated_hotwire_velocity(DEFAULT_H5)
u, fs = cv.u, cv.fs_hz
dpath = cv.dataset_path

lambda_series, u_rms, u_mean, mean_du_dt_sq = taylor_microscale_from_series(u, fs)

mean_dudx_sq = mean_du_dt_sq / (u_mean**2)
eps_time = dissipation_from_time_derivatives(u_mean, mean_du_dt_sq, nu=NU_AIR)
lambda_eps_time = taylor_microscale_from_epsilon(u_rms, eps_time, nu=NU_AIR)

eps_spec = epsilon_from_welch_spectrum(u, fs, nu=NU_AIR)
lambda_eps_spec = taylor_microscale_from_epsilon(u_rms, eps_spec, nu=NU_AIR)

re_lambda = u_rms * lambda_series / NU_AIR

l_11, t_l, l_used_zero = integral_length_scale_longitudinal(u, fs, u_mean)

# %%
print(f"Dataset: {dpath}")
print(f"N = {u.size}, fs = {fs} Hz, U_mean = {u_mean:.4f} m/s")
print(f"u_rms = {u_rms:.6f} m/s")
print(f"⟨(du'/dt)²⟩ = {mean_du_dt_sq:.6e} (m/s)² / s²")
print(f"⟨(∂u'/∂x)²⟩ = ⟨(du'/dt)²⟩/Ū² = {mean_dudx_sq:.6e} (m/s)² / m²")
print()
print("Taylor microscale λ = u_rms / √(⟨(∂u'/∂x)²⟩)  (= Ū u_rms / √(⟨(du'/dt)²⟩) under Taylor)")
print(f"  λ (from measurements) = {lambda_series:.6e} m")
print(
    f"  λ (from ε_time, isotropic) = {lambda_eps_time:.6e} m  "
    f"(agrees with λ above; |Δ|/λ = {abs(lambda_eps_time - lambda_series) / lambda_series:.2e})"
)
print()
print("Dissipation from the same time derivatives (isotropic identity):")
print(f"  ε_time = 15 ν ⟨(∂u'/∂x)²⟩ = {eps_time:.6e} m²/s³")
print()
print("Spectrum–integral ε and λ (can diverge if ∫k²E₁₁ dk is noise-dominated at high k):")
print(f"  ε_spec = {eps_spec:.6e} m²/s³")
print(f"  λ from ε_spec = {lambda_eps_spec:.6e} m")
print()
print(f"Re_λ = u' λ / ν = {re_lambda:.2f}")
print()
print("Longitudinal integral length L₁₁ = Ū ∫ R(τ)/⟨u′²⟩ dτ  (Taylor); τ integral to first ρ(τ)=0")
print(f"  T_L (integral time scale) = {t_l:.6e} s")
print(f"  L₁₁ = Ū T_L             = {l_11:.6e} m")
if not l_used_zero:
    print("  (note: ρ(τ) did not cross zero within max lag; T_L is truncated — increase record or max_lag)")
