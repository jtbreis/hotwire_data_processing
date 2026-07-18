# %% [markdown]
# # Hot-wire calibration (TNTI aligned with gravity)
#
# Load hot-wire / pitot time series, build **voltage → velocity** from calibration HDF5(s),
# compare calibrated hot-wire to pitot, and write a pitot-calibrated HDF5. All numerics live
# in **``mcflow_plotting.hotwire``**; figures use **``mcflow_plotting.plots``**.
#
# File naming:
# - ``Calibration_Curve_<date>_<wall-offset>_<scan-range>[_PartN][_<rotation>].h5``
# - ``TNTI_aligned_with_g_<wall-offset>_<scan-range>[_<rotation>].h5``
#
# ``<wall-offset>`` is distance from the wall (e.g. ``6in``, ``15p25in_113mm``).
# ``<scan-range>`` is where data were collected relative to that offset (e.g. ``0-200mm``).
# Some setups include a probe rotation suffix (``0deg`` or ``90deg``).
#
# Each case maps one data file to one *or more* calibration files. A calibration curve split
# over two HDF5 files is pooled automatically (works as long as the voltages stay unique).

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcflow_plotting import (
    collect_calibration_voltage_pitot_means,
    fit_voltage_to_velocity_poly,
    load_all_groups_pitot_and_hotwire_voltage,
    write_pitot_calibrated_hotwire_h5,
)
from mcflow_plotting.plots.hotwire import (
    plot_calibration_scatter_and_fit,
    plot_pitot_vs_calibrated_hotwire,
)

# %matplotlib inline  # uncomment when running in Jupyter

# %%
# --- paths & acquisition ---
DATA_DIR = Path(
    "/workspaces/hotwire_data_processing/data/hotwire/tnti_aligned_with_gravity"
)

# One case per measurement: (data file, [calibration file(s)]).
CASES: list[tuple[Path, list[Path]]] = [
    # 6 in from wall
    (
        DATA_DIR / "TNTI_aligned_with_g_6in_0-200mm.h5",
        [DATA_DIR / "Calibration_Curve_2026-07-02_6in_0-200mm.h5"],
    ),
    (
        DATA_DIR / "TNTI_aligned_with_g_6in_200-400mm.h5",
        [
            DATA_DIR / "Calibration_Curve_2026-07-07_6in_200-400mm_Part1.h5",
            DATA_DIR / "Calibration_Curve_2026-07-07_6in_200-400mm_Part2.h5",
        ],
    ),
    (
        DATA_DIR / "TNTI_aligned_with_g_6in_400-500mm.h5",
        [DATA_DIR / "Calibration_Curve_2026-07-07_6in_400-500mm.h5"],
    ),
    # 15.25 in from wall, 113 mm scan — 0° and 90° probe rotation
    (
        DATA_DIR / "TNTI_aligned_with_g_15p25in_113mm_0deg.h5",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_113mm_0deg.h5"],
    ),
    (
        DATA_DIR / "TNTI_aligned_with_g_15p25in_113mm_90deg.h5",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_113mm_90deg.h5"],
    ),
    # 15.25 in from wall — longer scan ranges
    (
        DATA_DIR / "TNTI_aligned_with_g_15p25in_165-365mm.h5",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_165-365mm.h5"],
    ),
    (
        DATA_DIR / "TNTI_aligned_with_g_15p25in_365-615mm.h5",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_365-615mm.h5"],
    ),
]

FS_HZ = 10000.0  # from filename

HW_V_MIN = -4.9
HW_V_MAX = 4.9

MAX_PLOT_POINTS = 12_000

# %%
for data_h5, calibration_h5s in CASES:
    print(f"\n=== Calibrating {data_h5.name} ===")

    # --- build voltage → velocity fit from the calibration file(s) ---
    v_cal, u_cal, n_skip = collect_calibration_voltage_pitot_means(
        calibration_h5s, v_min_strict=HW_V_MIN, v_max_strict=HW_V_MAX
    )
    if n_skip:
        print(
            f"Calibration: kept points strictly inside ({HW_V_MIN}, {HW_V_MAX}) V; "
            f"skipped {n_skip} run(s) outside that open interval."
        )
    if v_cal.size == 0:
        raise ValueError(
            f"No calibration runs with {HW_V_MIN} V < mean raw V < {HW_V_MAX} V "
            f"in {[str(p) for p in calibration_h5s]}."
        )

    fit = fit_voltage_to_velocity_poly(v_cal, u_cal, max_degree=3)
    cal_poly = fit.as_poly1d()
    rms_cal = fit.rms_residual_at_means

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

    # --- apply the fit to every run in the data file ---
    runs_raw = load_all_groups_pitot_and_hotwire_voltage(data_h5)
    calibrated_runs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for run_name, u_pitot, v_raw in runs_raw:
        u_hotwire = fit.apply_voltages(
            v_raw, clip_low=HW_V_MIN, clip_high=HW_V_MAX
        )
        calibrated_runs[run_name] = (u_pitot, u_hotwire)
        t = np.arange(u_pitot.size, dtype=float) / FS_HZ
        step = max(1, len(t) // MAX_PLOT_POINTS)
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

    print(f"Calibration RMS residual (mean points): {rms_cal:.4f} m/s")
    print(
        f"Polynomial coefficients (highest degree first): {fit.coeffs_high_first}"
    )
    for run_name, (u_pitot, u_hotwire) in calibrated_runs.items():
        print(
            f"{run_name}: mean |pitot - hotwire_cal| = "
            f"{np.mean(np.abs(u_pitot - u_hotwire)):.4f} m/s"
        )

    # --- write pitot-calibrated hot-wire HDF5 ---
    output_h5 = data_h5.parent / f"{data_h5.stem}_pitot_calibrated_hotwire.h5"
    write_pitot_calibrated_hotwire_h5(
        output_h5,
        runs=calibrated_runs,
        fs_hz=FS_HZ,
        source_h5=data_h5,
        calibration_h5=calibration_h5s,
        poly_degree=fit.poly_degree,
        poly_coeffs_high_first=fit.coeffs_high_first,
        v_clip_inclusive=(HW_V_MIN, HW_V_MAX),
        calibration_v_mean_strict_range=f"{HW_V_MIN} V < V_mean < {HW_V_MAX} V",
    )
    print(f"Wrote {len(calibrated_runs)} run(s) to: {output_h5}")
