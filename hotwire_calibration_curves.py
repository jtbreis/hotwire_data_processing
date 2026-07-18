# %% [markdown]
# # Hot-wire / Pitot calibration curves (all cases)
#
# Plot calibration data from every ``Calibration_Curve_*.h5`` under
# ``data/hotwire/tnti_aligned_with_gravity``:
#
# - **U(V) mapping**: mean pitot velocity vs mean raw hot-wire voltage, with polynomial fit
# - **Fan sweep**: fan drive voltage vs mean pitot velocity and vs mean hot-wire voltage
# - **Fan over pitot**: fan drive voltage vs mean pitot velocity (pitot on x-axis)
#
# Calibration HDF5 layout (one top-level group per fan setting, e.g. ``Run-fans=10V``):
#
# ```
# <run_name>/
#   Velocity in ms-1/Velocity in ms-1   # pitot velocity time series (m s⁻¹)
#   hot-wire/raw_data                   # hot-wire raw voltage time series (V)
# ```
#
# Cases that span two files (e.g. Part1 + Part2) are pooled automatically.

# %%
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mcflow_plotting import (
    collect_calibration_voltage_pitot_means,
    fit_voltage_to_velocity_poly,
)
from mcflow_plotting.plots.hotwire import plot_calibration_scatter_and_fit
from mcflow_plotting.style.colors import FLOW_COLORS, OVERLAY_COLORS
from mcflow_plotting.style.figure import (
    analysis_plots_dir,
    finalize_figure,
    save_figure,
    use_lab_matplotlib_style,
)

# %matplotlib inline  # uncomment when running in Jupyter

# %%
DATA_DIR = Path(
    "/workspaces/hotwire_data_processing/data/hotwire/tnti_aligned_with_gravity"
)

# (case label, calibration file(s)) — mirrors ``hotwire_calibration.py`` CASES.
CALIBRATION_CASES: list[tuple[str, list[Path]]] = [
    ("6in_0-200mm", [DATA_DIR / "Calibration_Curve_2026-07-02_6in_0-200mm.h5"]),
    (
        "6in_200-400mm",
        [
            DATA_DIR / "Calibration_Curve_2026-07-07_6in_200-400mm_Part1.h5",
            DATA_DIR / "Calibration_Curve_2026-07-07_6in_200-400mm_Part2.h5",
        ],
    ),
    ("6in_400-500mm", [DATA_DIR / "Calibration_Curve_2026-07-07_6in_400-500mm.h5"]),
    (
        "15p25in_113mm_0deg",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_113mm_0deg.h5"],
    ),
    (
        "15p25in_113mm_90deg",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_113mm_90deg.h5"],
    ),
    (
        "15p25in_165-365mm",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_165-365mm.h5"],
    ),
    (
        "15p25in_365-615mm",
        [DATA_DIR / "Calibration_Curve_2026-07-08_15p25in_365-615mm.h5"],
    ),
]

HW_V_MIN = -4.9
HW_V_MAX = 4.9
MAX_POLY_DEGREE = 3

PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
PLOTS_DIR = analysis_plots_dir(PLOTS_ROOT, DATA_DIR, "calibration")
SHOW_PLOTS = True

_FAN_V_RE = re.compile(r"fans=(\d+(?:\.\d+)?)V", re.IGNORECASE)


@dataclass(frozen=True)
class CalibrationRunPoint:
    run_name: str
    fan_v: float
    u_mean: float
    v_mean: float


def _parse_fan_voltage(run_name: str) -> float:
    match = _FAN_V_RE.search(run_name)
    if not match:
        raise ValueError(f"Cannot parse fan voltage from run name {run_name!r}.")
    return float(match.group(1))


def load_calibration_run_points(
    calibration_h5s: list[Path],
) -> list[CalibrationRunPoint]:
    """One point per run group: fan voltage, mean pitot U, mean raw hot-wire V."""
    points: list[CalibrationRunPoint] = []
    for path in calibration_h5s:
        with h5py.File(path, "r") as f:
            for run_name in f:
                g = f[run_name]
                u = g["Velocity in ms-1"]["Velocity in ms-1"][:]
                v = g["hot-wire"]["raw_data"][:]
                points.append(
                    CalibrationRunPoint(
                        run_name=str(run_name),
                        fan_v=_parse_fan_voltage(str(run_name)),
                        u_mean=float(np.mean(u)),
                        v_mean=float(np.mean(v)),
                    )
                )
    points.sort(key=lambda p: p.fan_v)
    return points


def plot_fan_voltage_sweep(
    *,
    case_label: str,
    points: list[CalibrationRunPoint],
    figsize: tuple[float, float] = (10.0, 4.5),
) -> tuple[plt.Figure, np.ndarray]:
    """Fan drive voltage vs mean pitot velocity and mean hot-wire voltage."""
    use_lab_matplotlib_style()
    fan_v = np.array([p.fan_v for p in points], dtype=float)
    u_mean = np.array([p.u_mean for p in points], dtype=float)
    v_mean = np.array([p.v_mean for p in points], dtype=float)

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=figsize, sharex=True)

    ax_u.plot(
        fan_v,
        u_mean,
        "o-",
        ms=6,
        lw=1.4,
        color=FLOW_COLORS[0],
        markeredgecolor="0.2",
        markeredgewidth=0.5,
        label="Pitot mean velocity",
    )
    ax_u.set_ylabel("Pitot mean velocity (m s⁻¹)")
    ax_u.set_title("Pitot")
    ax_u.grid(True, alpha=0.25)
    ax_u.legend(loc="best", fontsize=8)

    ax_v.plot(
        fan_v,
        v_mean,
        "s-",
        ms=6,
        lw=1.4,
        color=FLOW_COLORS[1],
        markeredgecolor="0.2",
        markeredgewidth=0.5,
        label="Hot-wire mean raw voltage",
    )
    ax_v.axhline(HW_V_MIN, color="0.5", ls="--", lw=1.0, alpha=0.7)
    ax_v.axhline(HW_V_MAX, color="0.5", ls="--", lw=1.0, alpha=0.7)
    ax_v.set_ylabel("Hot-wire mean raw voltage (V)")
    ax_v.set_title("Hot-wire")
    ax_v.grid(True, alpha=0.25)
    ax_v.legend(loc="best", fontsize=8)

    for ax in (ax_u, ax_v):
        ax.set_xlabel("Fan drive voltage (V)")

    fig.suptitle(
        f"Calibration fan sweep — {case_label}",
        fontsize=12,
        y=1.02,
    )
    finalize_figure(fig)
    return fig, np.array([ax_u, ax_v])


def plot_fan_voltage_over_pitot_velocity(
    *,
    case_label: str,
    points: list[CalibrationRunPoint],
    figsize: tuple[float, float] = (7.0, 4.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Fan drive voltage vs mean pitot velocity (pitot on x-axis)."""
    use_lab_matplotlib_style()
    fan_v = np.array([p.fan_v for p in points], dtype=float)
    u_mean = np.array([p.u_mean for p in points], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        u_mean,
        fan_v,
        "o-",
        ms=6,
        lw=1.4,
        color=FLOW_COLORS[2],
        markeredgecolor="0.2",
        markeredgewidth=0.5,
        label="Fan drive voltage",
    )
    ax.set_xlabel("Pitot mean velocity (m s⁻¹)")
    ax.set_ylabel("Fan drive voltage (V)")
    ax.set_title(f"Fan voltage vs pitot velocity — {case_label}", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    finalize_figure(fig)
    return fig, ax


def plot_all_fan_over_pitot(
    *,
    case_points: list[tuple[str, list[CalibrationRunPoint]]],
    figsize: tuple[float, float] = (8.0, 5.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay fan voltage vs pitot velocity for every case."""
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, points) in enumerate(case_points):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        u_mean = np.array([p.u_mean for p in points], dtype=float)
        fan_v = np.array([p.fan_v for p in points], dtype=float)
        ax.plot(
            u_mean,
            fan_v,
            "o-",
            ms=5,
            lw=1.3,
            color=color,
            markeredgecolor="0.2",
            markeredgewidth=0.4,
            label=label,
        )

    ax.set_xlabel("Pitot mean velocity (m s⁻¹)")
    ax.set_ylabel("Fan drive voltage (V)")
    ax.set_title("Fan voltage vs pitot velocity — all cases", fontsize=12)
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)
    finalize_figure(fig)
    return fig, ax


def plot_all_uv_curves(
    *,
    case_results: list[tuple[str, np.ndarray, np.ndarray, np.poly1d, int, float]],
    figsize: tuple[float, float] = (8.0, 5.5),
) -> tuple[plt.Figure, plt.Axes]:
    """Overlay U(V) scatter + fit for every case."""
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (label, v_mean, u_mean, poly, deg, rms) in enumerate(case_results):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        ax.scatter(
            v_mean,
            u_mean,
            s=36,
            color=color,
            edgecolors="0.2",
            linewidths=0.5,
            zorder=3,
            label=f"{label} (deg {deg}, RMS {rms:.3f})",
        )
        v_curve = np.linspace(float(v_mean.min()), float(v_mean.max()), 200)
        ax.plot(v_curve, poly(v_curve), lw=1.3, color=color, alpha=0.85, zorder=2)

    ax.axvline(HW_V_MIN, color="0.5", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.axvline(HW_V_MAX, color="0.5", ls="--", lw=1.0, alpha=0.7, zorder=1)
    ax.set_xlabel("Hot-wire mean raw voltage (V)")
    ax.set_ylabel("Pitot mean velocity (m s⁻¹)")
    ax.set_title(
        f"All calibration U(V) curves ({HW_V_MIN} V < V̄ < {HW_V_MAX} V)",
        fontsize=12,
    )
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.25)
    finalize_figure(fig)
    return fig, ax


# %%
all_case_results: list[tuple[str, np.ndarray, np.ndarray, np.poly1d, int, float]] = []
all_case_points: list[tuple[str, list[CalibrationRunPoint]]] = []

for case_label, calibration_h5s in CALIBRATION_CASES:
    print(f"\n=== {case_label} ===")

    points = load_calibration_run_points(calibration_h5s)
    all_case_points.append((case_label, points))
    print(f"  {len(points)} fan settings from {[p.name for p in calibration_h5s]}")

    v_cal, u_cal, n_skip = collect_calibration_voltage_pitot_means(
        calibration_h5s, v_min_strict=HW_V_MIN, v_max_strict=HW_V_MAX
    )
    if v_cal.size == 0:
        raise ValueError(
            f"No calibration points inside ({HW_V_MIN}, {HW_V_MAX}) V for {case_label}."
        )
    if n_skip:
        print(f"  Skipped {n_skip} run(s) with mean V outside ({HW_V_MIN}, {HW_V_MAX}) V")

    fit = fit_voltage_to_velocity_poly(v_cal, u_cal, max_degree=MAX_POLY_DEGREE)
    poly = fit.as_poly1d()
    all_case_results.append(
        (case_label, v_cal, u_cal, poly, fit.poly_degree, fit.rms_residual_at_means)
    )
    print(
        f"  U(V) fit: degree {fit.poly_degree}, "
        f"RMS = {fit.rms_residual_at_means:.4f} m s⁻¹, "
        f"{v_cal.size} points"
    )

    # --- U(V) calibration curve ---
    v_curve = np.linspace(float(v_cal.min()), float(v_cal.max()), 300)
    fig_uv, _ = plot_calibration_scatter_and_fit(
        v_mean=v_cal,
        u_mean=u_cal,
        v_curve=v_curve,
        u_curve=poly(v_curve),
        poly_degree=fit.poly_degree,
        rms_cal=fit.rms_residual_at_means,
        v_min=HW_V_MIN,
        v_max=HW_V_MAX,
    )
    fig_uv.suptitle(f"Calibration U(V) — {case_label}", fontsize=12, y=1.02)
    finalize_figure(fig_uv)
    uv_path = PLOTS_DIR / f"uv_{case_label}.pdf"
    save_figure(fig_uv, uv_path)
    print(f"  Wrote {uv_path.name}")

    # --- Fan voltage sweep (pitot + hot-wire) ---
    fig_fan, _ = plot_fan_voltage_sweep(case_label=case_label, points=points)
    fan_path = PLOTS_DIR / f"fan-sweep_{case_label}.pdf"
    save_figure(fig_fan, fan_path)
    print(f"  Wrote {fan_path.name}")

    # --- Fan voltage vs pitot velocity ---
    fig_fop, _ = plot_fan_voltage_over_pitot_velocity(
        case_label=case_label, points=points
    )
    fop_path = PLOTS_DIR / f"fan-over-pitot_{case_label}.pdf"
    save_figure(fig_fop, fop_path)
    print(f"  Wrote {fop_path.name}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_uv)
        plt.close(fig_fan)
        plt.close(fig_fop)

# --- Combined overlay of all U(V) curves ---
fig_all, _ = plot_all_uv_curves(case_results=all_case_results)
all_path = PLOTS_DIR / "uv_all-cases.pdf"
save_figure(fig_all, all_path)
print(f"\nWrote combined overlay: {all_path.resolve()}")

fig_fop_all, _ = plot_all_fan_over_pitot(case_points=all_case_points)
fop_all_path = PLOTS_DIR / "fan-over-pitot_all-cases.pdf"
save_figure(fig_fop_all, fop_all_path)
print(f"Wrote combined fan-over-pitot: {fop_all_path.resolve()}")

if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig_all)
    plt.close(fig_fop_all)
