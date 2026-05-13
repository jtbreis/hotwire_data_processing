# %% [markdown]
# # Hot-wire calibration (TTI, no gravity, turbulent side)
#
# Load hot-wire / pitot time series, build **voltage → velocity** from a calibration HDF5,
# compare calibrated hot-wire to pitot, and write a pitot-calibrated HDF5. All numerics live
# in **``mcflow_plotting.hotwire``**; figures use **``mcflow_plotting.plots``**.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcflow_plotting import (
    collect_calibration_voltage_pitot_means,
    fit_voltage_to_velocity_poly,
    load_first_group_pitot_and_hotwire_voltage,
    write_pitot_calibrated_hotwire_h5,
)
from mcflow_plotting.plots.hotwire import (
    plot_calibration_scatter_and_fit,
    plot_pitot_vs_calibrated_hotwire,
)

# %matplotlib inline  # uncomment when running in Jupyter

# %%
# --- paths & acquisition ---
CALIBRATION_H5 = Path(
    "/workspace/data/hotwire/tti_no_gravity/raw_data/"
    "hotwire_calibration_TTI_no_gravity_non_turbulent_side.h5"
)
DATA_5K_H5 = Path(
    "/workspace/data/hotwire/tti_no_gravity/raw_data/"
    "hotwire_10kHz_TTI_no_gravity_non_turbulent_side_2.h5"
)
FS_HZ = 10000.0  # from filename

HW_V_MIN = -4.9
HW_V_MAX = 4.9

# %%
v_cal, u_cal, n_skip = collect_calibration_voltage_pitot_means(
    CALIBRATION_H5, v_min_strict=HW_V_MIN, v_max_strict=HW_V_MAX
)
if n_skip:
    print(
        f"Calibration: kept points strictly inside ({HW_V_MIN}, {HW_V_MAX}) V; "
        f"skipped {n_skip} run(s) outside that open interval."
    )
if v_cal.size == 0:
    raise ValueError(
        f"No calibration runs with {HW_V_MIN} V < mean raw V < {HW_V_MAX} V in {CALIBRATION_H5}."
    )

fit = fit_voltage_to_velocity_poly(v_cal, u_cal, max_degree=3)
cal_poly = fit.as_poly1d()
rms_cal = fit.rms_residual_at_means

# %%
v_curve = np.linspace(float(v_cal.min()), float(v_cal.max()), 300)
fig_cal, ax_cal = plot_calibration_scatter_and_fit(
    v_mean=v_cal,
    u_mean=u_cal,
    v_curve=v_curve,
    u_curve=cal_poly(v_curve),
    poly_degree=fit.poly_degree,
    rms_cal=rms_cal,
    v_min=HW_V_MIN,
    v_max=HW_V_MAX,
)
plt.show()

# %%
run_name, u_pitot, v_raw = load_first_group_pitot_and_hotwire_voltage(DATA_5K_H5)
u_hotwire = fit.apply_voltages(v_raw, clip_low=HW_V_MIN, clip_high=HW_V_MAX)
t = np.arange(u_pitot.size, dtype=float) / FS_HZ

max_plot_points = 12_000
step = max(1, len(t) // max_plot_points)
fig_ts, ax_ts = plot_pitot_vs_calibrated_hotwire(
    t_sub=t[::step],
    u_pitot_sub=u_pitot[::step],
    u_hotwire_sub=u_hotwire[::step],
    poly_degree=fit.poly_degree,
    v_clip_low=HW_V_MIN,
    v_clip_high=HW_V_MAX,
    run_name=run_name,
)
plt.show()

# %%
print(f"Calibration RMS residual (mean points): {rms_cal:.4f} m/s")
print(
    f"Time-series mean |pitot - hotwire_cal|: {np.mean(np.abs(u_pitot - u_hotwire)):.4f} m/s"
)
print(f"Polynomial coefficients (highest degree first): {fit.coeffs_high_first}")

# %%
OUTPUT_H5 = DATA_5K_H5.parent / f"{DATA_5K_H5.stem}_pitot_calibrated_hotwire.h5"
write_pitot_calibrated_hotwire_h5(
    OUTPUT_H5,
    run_name=run_name,
    u_pitot=u_pitot,
    u_hotwire_calibrated=u_hotwire,
    fs_hz=FS_HZ,
    source_h5=DATA_5K_H5,
    calibration_h5=CALIBRATION_H5,
    poly_degree=fit.poly_degree,
    poly_coeffs_high_first=fit.coeffs_high_first,
    v_clip_inclusive=(HW_V_MIN, HW_V_MAX),
    calibration_v_mean_strict_range=f"{HW_V_MIN} V < V_mean < {HW_V_MAX} V",
)
print(f"Wrote: {OUTPUT_H5}")
