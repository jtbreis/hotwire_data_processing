# %% [markdown]
# # Hot-wire profiles vs wall-normal distance
#
# Load every position from pitot-calibrated HDF5 files under
# ``data/hotwire/tnti_aligned_with_gravity`` and plot vs distance from the wall:
# Kolmogorov lengthscale η (hot-wire spectrum), mean velocity (hot-wire vs pitot),
# RMS fluctuation velocity (hot-wire vs pitot), and dominant frequency (hot-wire Welch PSD).

# %%
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mcflow_plotting import NU_AIR, compute_normalized_spectrum_from_velocity
from mcflow_plotting.style.colors import FLOW_COLORS
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
CALIBRATED_GLOB = "TNTI_aligned_with_g_*_pitot_calibrated_hotwire.h5"

# Spectrum / statistics use this slice of each run (None = full record).
T_START_S: float | None = None
T_END_S: float | None = None
SPECTRUM_METHOD = os.environ.get(
    "TKE_SPECTRUM_METHOD", "welch"
).strip().lower()
NU_AIR_LOCAL = NU_AIR

PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
PLOTS_DIR = analysis_plots_dir(PLOTS_ROOT, DATA_DIR, "wall-profiles")
SHOW_PLOTS = True

INCH_TO_MM = 25.4

# Nominal local scan ranges from filenames / run labels (mm, before geometry offset).
GEOMETRY_SCAN_RANGE_LOCAL_MM: dict[str, tuple[float, float]] = {
    "6in": (0.0, 500.0),
    "15.25in": (113.0, 615.0),
}

_RUN_Y_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_RUN_GEO_RE = re.compile(r"Run-([\d.p]+in)", re.IGNORECASE)
_RUN_ANGLE_RE = re.compile(r"(\d+)\s*deg", re.IGNORECASE)
_STEM_RANGE_RE = re.compile(r"_(\d+)-(\d+)mm")
_STEM_SINGLE_Y_RE = re.compile(r"_(\d+)mm_(?:0|9)0deg")


@dataclass(frozen=True)
class WallProfilePoint:
    """One wall-normal measurement location."""

    geometry: str
    angle_deg: int | None
    y_mm: float
    run_name: str
    h5_path: Path
    u_mean_hotwire: float
    u_mean_pitot: float
    u_rms_hotwire: float
    u_rms_pitot: float
    eta_m: float
    f_dom_hz: float
    interior_score: float


@dataclass(frozen=True)
class ProfileSeries:
    """Points sharing geometry and optional probe rotation."""

    geometry: str
    angle_deg: int | None
    points: tuple[WallProfilePoint, ...]

    @property
    def label(self) -> str:
        if self.angle_deg is None:
            return self.geometry
        return f"{self.geometry} ({self.angle_deg}°)"


def _jupyter_matplotlib_inline() -> None:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
    except (ImportError, AttributeError):
        pass


_jupyter_matplotlib_inline()


def parse_y_mm_from_run_name(run_name: str) -> float:
    """Extract wall distance in mm from names like ``Run-6in + 200mm from wall``."""
    match = _RUN_Y_MM_RE.search(run_name)
    if match is None:
        raise ValueError(
            f"Could not parse wall distance from run name {run_name!r}."
        )
    return float(match.group(1))


def parse_geometry(run_name: str) -> str:
    match = _RUN_GEO_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse geometry from {run_name!r}.")
    return match.group(1)


def geometry_diameter_mm(geometry: str) -> float:
    """Convert diffuser label (e.g. ``6in``, ``15.25in``) to diameter in mm."""
    if not geometry.endswith("in"):
        raise ValueError(f"Unrecognized geometry label {geometry!r}.")
    inches = float(geometry.removesuffix("in"))
    return inches * INCH_TO_MM


def geometry_wall_offset_mm(geometry: str) -> float:
    """Shift local probe positions onto the common wall-normal axis."""
    return geometry_diameter_mm(geometry)


def wall_distance_mm(geometry: str, y_mm: float) -> float:
    """Distance from the main wall: local label + geometry offset."""
    return geometry_wall_offset_mm(geometry) + y_mm


def scan_range_wall_mm(geometry: str) -> tuple[float, float]:
    lo, hi = GEOMETRY_SCAN_RANGE_LOCAL_MM[geometry]
    off = geometry_wall_offset_mm(geometry)
    return lo + off, hi + off


def measurement_wall_y_mm_for_geometry(
    points: list[WallProfilePoint],
    geometry: str,
) -> np.ndarray:
    """Sorted unique wall-normal locations on the combined axis."""
    ys = sorted(
        {wall_distance_mm(p.geometry, p.y_mm) for p in points if p.geometry == geometry}
    )
    return np.asarray(ys, dtype=float)


def measurement_y_mm_for_geometry(
    points: list[WallProfilePoint],
    geometry: str,
) -> np.ndarray:
    """Sorted unique local measurement labels for one geometry."""
    ys = sorted({p.y_mm for p in points if p.geometry == geometry})
    return np.asarray(ys, dtype=float)


def parse_angle_deg(run_name: str) -> int | None:
    match = _RUN_ANGLE_RE.search(run_name)
    return int(match.group(1)) if match else None


def interior_score_for_point(h5_stem: str, y_mm: float) -> float:
    """
    Prefer scan files where ``y_mm`` lies in the interior of the stated range.

    Tie-break at shared scan boundaries: prefer the file whose range ends at ``y_mm``
    (same calibration session as upstream points in that scan).
    """
    stem = h5_stem.replace("TNTI_aligned_with_g_", "")
    range_match = _STEM_RANGE_RE.search(stem)
    if range_match:
        y_lo = float(range_match.group(1))
        y_hi = float(range_match.group(2))
        return min(y_mm - y_lo, y_hi - y_mm)
    single_match = _STEM_SINGLE_Y_RE.search(stem)
    if single_match:
        y0 = float(single_match.group(1))
        return 0.0 if abs(y_mm - y0) < 1e-6 else -1.0
    return 0.0


def spectrum_window_slice(
    n: int,
    fs_hz: float,
    *,
    t_start_s: float | None,
    t_end_s: float | None,
) -> slice:
    s0 = 0 if t_start_s is None else max(0, int(np.floor(t_start_s * fs_hz)))
    s1 = n if t_end_s is None else min(n, int(np.ceil(t_end_s * fs_hz)))
    if s1 <= s0:
        raise ValueError(f"Empty spectrum window: indices {s0}..{s1}.")
    return slice(s0, s1)


def dominant_frequency_hz(u: np.ndarray, fs_hz: float) -> float:
    """Peak frequency of one-sided Welch PSD of velocity fluctuations (f > 0)."""
    u = np.asarray(u, dtype=float)
    u_fluc = u - float(np.mean(u))
    nperseg = min(65_536, max(8192, u.size // 4))
    f_hz, p_uu = signal.welch(
        u_fluc,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
        average="mean",
    )
    # Skip DC only; low-frequency energy is allowed to be the dominant peak.
    band = f_hz > 0.0
    if not np.any(band):
        raise ValueError("No positive-frequency PSD bins for dominant-frequency estimate.")
    i_peak = int(np.argmax(p_uu[band]))
    return float(f_hz[band][i_peak])


def load_calibrated_runs(path: Path) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
    """Return ``(run_name, u_pitot, u_hotwire, fs_hz)`` for each top-level group."""
    runs: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    with h5py.File(path, "r") as f:
        fs_hz = float(f.attrs.get("sample_rate_Hz", 10_000.0))
        for run_name in f:
            grp = f[run_name]
            u_pitot = grp["Pitot"]["Velocity Pitot in ms-1"][:].astype(float)
            u_hotwire = grp["Velocity_calibrated_hotwire_ms"][:].astype(float)
            runs.append((str(run_name), u_pitot, u_hotwire, fs_hz))
    return runs


def collect_wall_profile_points(
    data_dir: Path,
    *,
    spectrum_method: str,
    nu: float,
    t_start_s: float | None,
    t_end_s: float | None,
) -> list[WallProfilePoint]:
    h5_paths = sorted(data_dir.glob(CALIBRATED_GLOB))
    if not h5_paths:
        raise FileNotFoundError(
            f"No files matching {CALIBRATED_GLOB!r} found in {data_dir}."
        )

    points: list[WallProfilePoint] = []
    for h5_path in h5_paths:
        for run_name, u_pitot_all, u_hotwire_all, fs_hz in load_calibrated_runs(
            h5_path
        ):
            y_mm = parse_y_mm_from_run_name(run_name)
            geometry = parse_geometry(run_name)
            angle_deg = parse_angle_deg(run_name)
            sl = spectrum_window_slice(
                u_hotwire_all.size,
                fs_hz,
                t_start_s=t_start_s,
                t_end_s=t_end_s,
            )
            u_pitot = u_pitot_all[sl]
            u_hotwire = u_hotwire_all[sl]

            spec = compute_normalized_spectrum_from_velocity(
                u_hotwire,
                fs_hz,
                run_name,
                spectrum_method=spectrum_method,
                nu=nu,
                path=str(h5_path),
                dpath=f"{run_name}/Velocity_calibrated_hotwire_ms",
            )

            points.append(
                WallProfilePoint(
                    geometry=geometry,
                    angle_deg=angle_deg,
                    y_mm=y_mm,
                    run_name=run_name,
                    h5_path=h5_path,
                    u_mean_hotwire=float(np.mean(u_hotwire)),
                    u_mean_pitot=float(np.mean(u_pitot)),
                    u_rms_hotwire=float(np.std(u_hotwire, ddof=0)),
                    u_rms_pitot=float(np.std(u_pitot, ddof=0)),
                    eta_m=float(spec["eta"]),
                    f_dom_hz=dominant_frequency_hz(u_hotwire, fs_hz),
                    interior_score=interior_score_for_point(h5_path.stem, y_mm),
                )
            )
    points.sort(
        key=lambda p: (p.geometry, p.angle_deg or -1, p.y_mm, p.h5_path.name)
    )
    return points


def dedupe_wall_profile_points(
    points: list[WallProfilePoint],
) -> list[WallProfilePoint]:
    """Keep one point per (geometry, angle, y), preferring interior scan positions."""
    best: dict[tuple[str, int | None, float], WallProfilePoint] = {}
    for point in points:
        key = (point.geometry, point.angle_deg, point.y_mm)
        current = best.get(key)
        if current is None:
            best[key] = point
            continue
        if point.interior_score > current.interior_score:
            best[key] = point
            continue
        if point.interior_score == current.interior_score:
            # Shared scan boundary: prefer the file whose range ends at y_mm.
            y_hi_current = _scan_y_hi(current.h5_path.stem)
            y_hi_new = _scan_y_hi(point.h5_path.stem)
            if y_hi_new is not None and abs(y_hi_new - point.y_mm) < 1e-6:
                best[key] = point
    return sorted(
        best.values(),
        key=lambda p: (p.geometry, p.angle_deg or -1, p.y_mm),
    )


def drop_angled_probe_points(
    points: list[WallProfilePoint],
) -> list[WallProfilePoint]:
    """Remove 0° / 90° probe-rotation runs; keep unlabelled (in-plane) scans."""
    return [p for p in points if p.angle_deg is None]


def merge_wall_profile_preferring_geometry(
    points: list[WallProfilePoint],
    *,
    preferred_geometry: str = "6in",
) -> list[WallProfilePoint]:
    """
    Build one continuous wall-normal profile.

    In the overlapping wall-distance range, keep ``preferred_geometry`` and drop
    the other geometry's points at or below that coverage.
    """
    preferred = [p for p in points if p.geometry == preferred_geometry]
    if not preferred:
        return sorted(
            points,
            key=lambda p: wall_distance_mm(p.geometry, p.y_mm),
        )

    y_pref_max = max(wall_distance_mm(p.geometry, p.y_mm) for p in preferred)
    others = [
        p
        for p in points
        if p.geometry != preferred_geometry
        and wall_distance_mm(p.geometry, p.y_mm) > y_pref_max + 1e-6
    ]
    return sorted(
        preferred + others,
        key=lambda p: wall_distance_mm(p.geometry, p.y_mm),
    )


def _scan_y_lo(h5_stem: str) -> float | None:
    stem = h5_stem.replace("TNTI_aligned_with_g_", "")
    match = _STEM_RANGE_RE.search(stem)
    return float(match.group(1)) if match else None


def _scan_y_hi(h5_stem: str) -> float | None:
    stem = h5_stem.replace("TNTI_aligned_with_g_", "")
    match = _STEM_RANGE_RE.search(stem)
    return float(match.group(2)) if match else None


def group_profile_series(points: list[WallProfilePoint]) -> list[ProfileSeries]:
    """Single combined series (same profile; one color)."""
    if not points:
        return []
    return [
        ProfileSeries(
            geometry="combined",
            angle_deg=None,
            points=tuple(
                sorted(points, key=lambda p: wall_distance_mm(p.geometry, p.y_mm))
            ),
        )
    ]


def _profile_pdf_path(stem: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    return PLOTS_DIR / f"{stem}.pdf"


def annotate_combined_axis(
    ax: plt.Axes,
    *,
    points: list[WallProfilePoint],
) -> None:
    """Set wall-normal limits from the merged profile."""
    if not points:
        return
    all_wall_y = np.asarray(
        [wall_distance_mm(p.geometry, p.y_mm) for p in points],
        dtype=float,
    )
    y_hi = float(np.max(all_wall_y))
    pad = 0.02 * y_hi
    ax.set_ylim(-pad, y_hi + pad)


def _metric_values(
    points: tuple[WallProfilePoint, ...],
    *,
    metric_key: str,
    sensor: str,
) -> np.ndarray:
    if metric_key == "eta_mm":
        return np.asarray([p.eta_m * 1e3 for p in points])
    if metric_key == "mean":
        return np.asarray(
            [p.u_mean_hotwire if sensor == "hotwire" else p.u_mean_pitot for p in points]
        )
    if metric_key == "rms":
        return np.asarray(
            [p.u_rms_hotwire if sensor == "hotwire" else p.u_rms_pitot for p in points]
        )
    if metric_key == "ti":
        # Turbulent intensity I = u_rms / ⟨U⟩.
        u_mean = np.asarray(
            [p.u_mean_hotwire if sensor == "hotwire" else p.u_mean_pitot for p in points],
            dtype=float,
        )
        u_rms = np.asarray(
            [p.u_rms_hotwire if sensor == "hotwire" else p.u_rms_pitot for p in points],
            dtype=float,
        )
        out = np.full_like(u_mean, np.nan, dtype=float)
        ok = np.abs(u_mean) > 1e-12
        out[ok] = u_rms[ok] / u_mean[ok]
        return out
    return np.asarray([p.f_dom_hz for p in points])


def _plot_series_on_axis(
    ax: plt.Axes,
    series_list: list[ProfileSeries],
    *,
    metric_key: str,
    sensor: str,
) -> None:
    color = FLOW_COLORS[0]
    for series in series_list:
        y_wall = np.asarray(
            [wall_distance_mm(p.geometry, p.y_mm) for p in series.points],
            dtype=float,
        )
        vals = _metric_values(series.points, metric_key=metric_key, sensor=sensor)
        style = "--s" if metric_key == "rms" else "-o"
        ax.plot(
            vals,
            y_wall,
            style,
            markersize=4,
            lw=1.6,
            color=color,
        )


def plot_sensor_wall_profiles(
    series_list: list[ProfileSeries],
    *,
    points: list[WallProfilePoint],
    sensor: str,
) -> tuple[plt.Figure, Path]:
    """
    Interface-style wall profiles (style guide §3): mean, RMS, and turbulent
    intensity vs wall-normal distance. Hot-wire also includes η and f_dom.
    """
    use_lab_matplotlib_style()

    ti_label = r"$u_\mathrm{rms}/\langle U \rangle$"
    if sensor == "hotwire":
        row_specs = (
            ("mean", r"$\langle U \rangle$ in $\mathrm{m/s}$", r"Mean velocity"),
            ("rms", r"$u_\mathrm{rms}$ in $\mathrm{m/s}$", r"RMS fluctuation"),
            ("ti", ti_label, r"Turbulent intensity"),
            ("eta_mm", r"$\eta$ in $\mathrm{mm}$", r"Kolmogorov $\eta$"),
            ("f_dom", r"$f_\mathrm{dom}$ in $\mathrm{Hz}$", r"PSD peak"),
        )
        fig, axes = plt.subplots(2, 3, figsize=(12.0, 10.0), sharey=True)
        axes_flat = list(axes.ravel())
        axes_flat[-1].set_visible(False)
        axes_flat = axes_flat[:-1]
    elif sensor == "pitot":
        row_specs = (
            ("mean", r"$\langle U \rangle$ in $\mathrm{m/s}$", r"Mean velocity"),
            ("rms", r"$u_\mathrm{rms}$ in $\mathrm{m/s}$", r"RMS fluctuation"),
            ("ti", ti_label, r"Turbulent intensity"),
        )
        fig, axes_flat = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True)
        axes_flat = list(axes_flat)
    else:
        raise ValueError(f"Unknown sensor {sensor!r}; expected 'hotwire' or 'pitot'.")

    for ax, (metric_key, metric_label, title) in zip(axes_flat, row_specs):
        _plot_series_on_axis(
            ax, series_list, metric_key=metric_key, sensor=sensor
        )
        annotate_combined_axis(ax, points=points)
        ax.set_xlabel(metric_label)
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

    axes_flat[0].set_ylabel(r"$y$ in $\mathrm{mm}$")

    sensor_title = "Hot-wire" if sensor == "hotwire" else "Pitot"
    fig.suptitle(
        rf"TNTI $\downarrow$ — {sensor_title} wall profiles",
        fontsize=16,
        y=1.01,
    )
    finalize_figure(fig, pad=0.6)

    pdf_path = _profile_pdf_path(f"{sensor}_{SPECTRUM_METHOD}")
    save_figure(fig, pdf_path)
    return fig, pdf_path


def plot_wall_profiles(
    series_list: list[ProfileSeries],
    *,
    points: list[WallProfilePoint],
) -> tuple[plt.Figure, plt.Figure, Path, Path]:
    """Return separate hot-wire and pitot profile figures."""
    fig_hw, pdf_hw = plot_sensor_wall_profiles(
        series_list, points=points, sensor="hotwire"
    )
    fig_pt, pdf_pt = plot_sensor_wall_profiles(
        series_list, points=points, sensor="pitot"
    )
    return fig_hw, fig_pt, pdf_hw, pdf_pt


def print_profile_table(points: list[WallProfilePoint]) -> None:
    print(
        "| geometry | angle | y_local (mm) | y_wall (mm) | run | ⟨U⟩ HW | ⟨U⟩ pitot | "
        "u_rms HW | u_rms pitot | I HW | I pitot | η (mm) | f_dom (Hz) |"
    )
    print(
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for p in points:
        angle = "" if p.angle_deg is None else str(p.angle_deg)
        y_wall = wall_distance_mm(p.geometry, p.y_mm)
        i_hw = p.u_rms_hotwire / p.u_mean_hotwire if abs(p.u_mean_hotwire) > 1e-12 else float("nan")
        i_pt = p.u_rms_pitot / p.u_mean_pitot if abs(p.u_mean_pitot) > 1e-12 else float("nan")
        print(
            f"| {p.geometry} | {angle} | {p.y_mm:.0f} | {y_wall:.1f} | {p.run_name} | "
            f"{p.u_mean_hotwire:.4f} | {p.u_mean_pitot:.4f} | "
            f"{p.u_rms_hotwire:.4f} | {p.u_rms_pitot:.4f} | "
            f"{i_hw:.4f} | {i_pt:.4f} | "
            f"{p.eta_m * 1e3:.4f} | {p.f_dom_hz:.2f} |"
        )


# %%
all_points = collect_wall_profile_points(
    DATA_DIR,
    spectrum_method=SPECTRUM_METHOD,
    nu=NU_AIR_LOCAL,
    t_start_s=T_START_S,
    t_end_s=T_END_S,
)
points = dedupe_wall_profile_points(all_points)
points = drop_angled_probe_points(points)
points = merge_wall_profile_preferring_geometry(points, preferred_geometry="6in")
series_list = group_profile_series(points)

print(
    f"Loaded {len(all_points)} raw position(s) -> {len(points)} merged "
    f"(no angled probes; 6in preferred in overlap) "
    f"from {len(set(p.h5_path for p in all_points))} HDF5 files."
)
print(
    "Merged wall profile: "
    + ", ".join(
        f"{wall_distance_mm(p.geometry, p.y_mm):.0f} mm ({p.geometry})"
        for p in points
    )
)
print_profile_table(points)

fig_hw, fig_pt, pdf_hw, pdf_pt = plot_wall_profiles(series_list, points=points)
print(f"Wrote {pdf_hw.resolve()}")
print(f"Wrote {pdf_pt.resolve()}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig_hw)
    plt.close(fig_pt)
