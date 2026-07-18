# %% [markdown]
# # Turbulent-burst regions on stacked hot-wire time signals
#
# Load pitot-calibrated HDF5 runs the same way as ``hotwire_wall_profiles.py``
# (``data/hotwire/tnti_aligned_with_gravity``). Apply a **Butterworth high-pass**
# (default cutoff **100 Hz**) to remove slow mean drift, then on a **10 s** window
# identify regions where fluctuations are **persistent over a short period**:
#
# 1. High-pass filter ``u``, then ``u' = u_hp - ⟨u_hp⟩``, ``σ = std(u_hp)``.
# 2. Short-time fluctuation amplitude: rolling RMS of ``u'`` over ``SHORT_PERIOD_S``.
# 3. Burst candidate when that rolling RMS exceeds **``N_SIGMA · σ``**
#    (default: local RMS > 1σ of the window — i.e. the velocity is fluctuating by
#    more than one standard deviation over that short interval).
# 4. Bridge brief gaps (``MERGE_GAP_S``) and keep only runs ≥ ``MIN_BURST_S``.
#
# Wall-normal positions are stacked (velocity offsets) with burst intervals shaded.

# %%
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from mcflow_plotting.style.colors import OVERLAY_COLORS
from mcflow_plotting.style.figure import (
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

# Analysis window (seconds into each record).
T_START_S: float = 0.0
T_END_S: float = 10.0

# High-pass before burst detection (None / <=0 disables).
HIGHPASS_CUTOFF_HZ: float | None = None
HIGHPASS_ORDER: int = 4

# Burst criterion: rolling RMS(u') over SHORT_PERIOD_S > N_SIGMA · σ_window.
N_SIGMA: float = 1.0
SHORT_PERIOD_S: float = 0.05  # short persistence / averaging window
MIN_BURST_S: float = 0.05  # drop shorter burst runs
MERGE_GAP_S: float = 0.02  # bridge brief false gaps (0 = off)

# How many wall-normal traces to stack (evenly subsampled from the merged profile).
MAX_TRACES: int = 8
# Vertical offset between traces in units of each trace's σ (visual separation).
TRACE_OFFSET_SIGMA: float = 6.0
MAX_PLOT_POINTS: int = 25_000

PLOTS_DIR = (
    Path(__file__).resolve().parent / "plots" / "hotwire_data_processing" / "burst_detection"
)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = True
FIGSIZE = (10.0, 8.0)

_RUN_Y_MM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mm", re.IGNORECASE)
_RUN_GEO_RE = re.compile(r"Run-([\d.p]+in)", re.IGNORECASE)
_RUN_ANGLE_RE = re.compile(r"(\d+)\s*deg", re.IGNORECASE)
_STEM_RANGE_RE = re.compile(r"_(\d+)-(\d+)mm")
_STEM_SINGLE_Y_RE = re.compile(r"_(\d+)mm_(?:0|9)0deg")
INCH_TO_MM = 25.4


def _jupyter_matplotlib_inline() -> None:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
    except (ImportError, AttributeError):
        pass


_jupyter_matplotlib_inline()


# %%
@dataclass(frozen=True)
class WallPosition:
    geometry: str
    angle_deg: int | None
    y_mm: float
    run_name: str
    h5_path: Path
    interior_score: float


@dataclass(frozen=True)
class BurstTrace:
    point: WallPosition
    t_s: np.ndarray
    u: np.ndarray
    u_mean: float
    u_std: float
    burst: np.ndarray
    intermittency: float


def parse_y_mm_from_run_name(run_name: str) -> float:
    match = _RUN_Y_MM_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse wall distance from run name {run_name!r}.")
    return float(match.group(1))


def parse_geometry(run_name: str) -> str:
    match = _RUN_GEO_RE.search(run_name)
    if match is None:
        raise ValueError(f"Could not parse geometry from {run_name!r}.")
    return match.group(1)


def parse_angle_deg(run_name: str) -> int | None:
    match = _RUN_ANGLE_RE.search(run_name)
    return int(match.group(1)) if match else None


def geometry_diameter_mm(geometry: str) -> float:
    if not geometry.endswith("in"):
        raise ValueError(f"Unrecognized geometry label {geometry!r}.")
    return float(geometry.removesuffix("in")) * INCH_TO_MM


def wall_distance_mm(geometry: str, y_mm: float) -> float:
    """
    Common axis: geometry diameter in mm + measurement point from the run label.

    ``6in`` → \(6\times 25.4\) mm, ``15.25in`` → \(15.25\times 25.4\) mm;
    ``y_mm`` from e.g. ``Run-6in + 200mm from wall``.
    """
    return geometry_diameter_mm(geometry) + float(y_mm)


def interior_score_for_point(h5_stem: str, y_mm: float) -> float:
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


def _scan_y_hi(h5_stem: str) -> float | None:
    stem = h5_stem.replace("TNTI_aligned_with_g_", "")
    match = _STEM_RANGE_RE.search(stem)
    return float(match.group(2)) if match else None


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


def load_calibrated_runs(path: Path) -> list[tuple[str, np.ndarray, float]]:
    """Return ``(run_name, u_hotwire, fs_hz)`` for each top-level group."""
    runs: list[tuple[str, np.ndarray, float]] = []
    with h5py.File(path, "r") as f:
        fs_hz = float(f.attrs.get("sample_rate_Hz", 10_000.0))
        for run_name in f:
            u_hotwire = f[run_name]["Velocity_calibrated_hotwire_ms"][:].astype(float)
            runs.append((str(run_name), u_hotwire, fs_hz))
    return runs


def collect_wall_positions(data_dir: Path) -> list[WallPosition]:
    h5_paths = sorted(data_dir.glob(CALIBRATED_GLOB))
    if not h5_paths:
        raise FileNotFoundError(
            f"No files matching {CALIBRATED_GLOB!r} found in {data_dir}."
        )
    points: list[WallPosition] = []
    for h5_path in h5_paths:
        for run_name, _u, _fs in load_calibrated_runs(h5_path):
            y_mm = parse_y_mm_from_run_name(run_name)
            points.append(
                WallPosition(
                    geometry=parse_geometry(run_name),
                    angle_deg=parse_angle_deg(run_name),
                    y_mm=y_mm,
                    run_name=run_name,
                    h5_path=h5_path,
                    interior_score=interior_score_for_point(h5_path.stem, y_mm),
                )
            )
    points.sort(key=lambda p: (p.geometry, p.angle_deg or -1, p.y_mm, p.h5_path.name))
    return points


def dedupe_wall_positions(points: list[WallPosition]) -> list[WallPosition]:
    best: dict[tuple[str, int | None, float], WallPosition] = {}
    for point in points:
        key = (point.geometry, point.angle_deg, point.y_mm)
        current = best.get(key)
        if current is None or point.interior_score > current.interior_score:
            best[key] = point
            continue
        if point.interior_score == current.interior_score:
            y_hi_current = _scan_y_hi(current.h5_path.stem)
            y_hi_new = _scan_y_hi(point.h5_path.stem)
            if y_hi_new is not None and abs(y_hi_new - point.y_mm) < 1e-6:
                best[key] = point
    return sorted(best.values(), key=lambda p: (p.geometry, p.angle_deg or -1, p.y_mm))


def drop_angled_probe_points(points: list[WallPosition]) -> list[WallPosition]:
    return [p for p in points if p.angle_deg is None]


def merge_preferring_geometry(
    points: list[WallPosition],
    *,
    preferred_geometry: str = "6in",
) -> list[WallPosition]:
    """Merge geometries on ``y = D_geo[mm] + y_meas``."""
    preferred = [p for p in points if p.geometry == preferred_geometry]
    if not preferred:
        return sorted(points, key=lambda p: wall_distance_mm(p.geometry, p.y_mm))
    y_pref_max = max(wall_distance_mm(p.geometry, p.y_mm) for p in preferred)
    others = [
        p
        for p in points
        if p.geometry != preferred_geometry
        and wall_distance_mm(p.geometry, p.y_mm) > y_pref_max + 1e-6
    ]
    return sorted(
        preferred + others, key=lambda p: wall_distance_mm(p.geometry, p.y_mm)
    )


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    n = int(m.size)
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def fill_short_false_gaps_between_true(mask: np.ndarray, max_gap_len: int) -> np.ndarray:
    if max_gap_len <= 0:
        return np.asarray(mask, dtype=bool)
    m = np.asarray(mask, dtype=bool).copy()
    n = int(m.size)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < n:
            if m[i]:
                i += 1
                continue
            j = i
            while j < n and not m[j]:
                j += 1
            gap = j - i
            left_t = i > 0 and m[i - 1]
            right_t = j < n and m[j]
            if left_t and right_t and gap <= max_gap_len:
                m[i:j] = True
                changed = True
            i = j
    return m


def highpass_filter(
    u: np.ndarray,
    fs_hz: float,
    *,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth high-pass (``filtfilt``)."""
    u = np.asarray(u, dtype=float)
    nyquist = 0.5 * float(fs_hz)
    if cutoff_hz <= 0.0 or cutoff_hz >= nyquist:
        raise ValueError(
            f"High-pass cutoff {cutoff_hz} Hz must lie in (0, {nyquist}) Hz."
        )
    sos = signal.butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, u)


def detect_burst_mask(
    u: np.ndarray,
    fs_hz: float,
    *,
    n_sigma: float,
    short_period_s: float,
    min_burst_s: float,
    merge_gap_s: float,
) -> tuple[np.ndarray, float, float]:
    """
    Burst where short-time RMS of ``u'`` exceeds ``n_sigma · σ`` of the window.

    That is the operational form of “velocity fluctuates by more than one
    standard deviation over a short period.”
    """
    u = np.asarray(u, dtype=float)
    if u.size < 2:
        raise ValueError("Need at least two samples for burst detection.")

    u_mean = float(np.mean(u))
    u_std = float(np.std(u, ddof=0))
    if u_std <= 0.0:
        return np.zeros(u.size, dtype=bool), u_mean, u_std

    u_prime = u - u_mean
    win = max(1, int(round(short_period_s * fs_hz)))
    # Rolling mean of (u')^2 → local RMS over the short period.
    rolling_ms = uniform_filter1d(u_prime * u_prime, size=win, mode="nearest")
    rolling_rms = np.sqrt(np.maximum(rolling_ms, 0.0))
    persistent = rolling_rms > (n_sigma * u_std)

    merge_gap = max(0, int(round(merge_gap_s * fs_hz)))
    if merge_gap > 0:
        persistent = fill_short_false_gaps_between_true(persistent, merge_gap)

    min_len = max(1, int(round(min_burst_s * fs_hz)))
    burst = np.zeros(u.size, dtype=bool)
    for i0, i1 in contiguous_runs(persistent):
        if i1 - i0 >= min_len:
            burst[i0:i1] = True
    return burst, u_mean, u_std


def downsample_for_plot(
    t: np.ndarray,
    *arrays: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, ...]:
    n = int(t.size)
    if n <= max_points:
        return (t, *arrays)
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return (t[idx], *(a[idx] for a in arrays))


def select_profile_subset(
    points: list[WallPosition],
    *,
    max_traces: int,
) -> list[WallPosition]:
    if len(points) <= max_traces:
        return list(points)
    idx = np.unique(np.round(np.linspace(0, len(points) - 1, max_traces)).astype(int))
    return [points[i] for i in idx]


def load_velocity_window(
    point: WallPosition,
    *,
    t_start_s: float | None,
    t_end_s: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    for run_name, u_hotwire_all, fs_hz in load_calibrated_runs(point.h5_path):
        if run_name != point.run_name:
            continue
        sl = spectrum_window_slice(
            u_hotwire_all.size,
            fs_hz,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
        )
        u = np.asarray(u_hotwire_all[sl], dtype=float)
        t0 = 0.0 if t_start_s is None else float(t_start_s)
        t = t0 + np.arange(u.size, dtype=float) / float(fs_hz)
        return t, u, float(fs_hz)
    raise KeyError(f"Run {point.run_name!r} not found in {point.h5_path}.")


def build_burst_traces(
    points: list[WallPosition],
    *,
    t_start_s: float,
    t_end_s: float,
    n_sigma: float,
    short_period_s: float,
    min_burst_s: float,
    merge_gap_s: float,
    highpass_cutoff_hz: float | None,
    highpass_order: int,
) -> list[BurstTrace]:
    traces: list[BurstTrace] = []
    for point in points:
        t, u, fs_hz = load_velocity_window(
            point, t_start_s=t_start_s, t_end_s=t_end_s
        )
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0.0:
            u = highpass_filter(
                u, fs_hz, cutoff_hz=highpass_cutoff_hz, order=highpass_order
            )
        burst, u_mean, u_std = detect_burst_mask(
            u,
            fs_hz,
            n_sigma=n_sigma,
            short_period_s=short_period_s,
            min_burst_s=min_burst_s,
            merge_gap_s=merge_gap_s,
        )
        traces.append(
            BurstTrace(
                point=point,
                t_s=t,
                u=u,
                u_mean=u_mean,
                u_std=u_std,
                burst=burst,
                intermittency=float(np.mean(burst)),
            )
        )
    return traces


def plot_stacked_burst_signals(
    traces: list[BurstTrace],
    *,
    n_sigma: float,
    short_period_s: float,
    max_plot_points: int,
    trace_offset_sigma: float,
    highpass_cutoff_hz: float | None,
) -> tuple[plt.Figure, Path]:
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    y_ticks: list[float] = []
    y_labels: list[str] = []

    for i, tr in enumerate(traces):
        color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        offset = i * trace_offset_sigma * tr.u_std
        u_fluc = tr.u - tr.u_mean
        t_p, u_p, b_p = downsample_for_plot(
            tr.t_s, u_fluc, tr.burst.astype(float), max_points=max_plot_points
        )
        burst_bool = b_p > 0.5
        y = u_p + offset

        ax.axhline(offset, color=color, lw=0.6, alpha=0.35, zorder=1)
        ax.axhline(
            offset - n_sigma * tr.u_std,
            color=color,
            lw=0.5,
            ls=":",
            alpha=0.35,
            zorder=1,
        )
        ax.axhline(
            offset + n_sigma * tr.u_std,
            color=color,
            lw=0.5,
            ls=":",
            alpha=0.35,
            zorder=1,
        )

        half = 0.45 * trace_offset_sigma * tr.u_std
        for i0, i1 in contiguous_runs(burst_bool):
            if i1 <= i0:
                continue
            t0 = float(t_p[i0])
            t1 = float(t_p[min(i1, t_p.size) - 1])
            ax.fill_between(
                [t0, t1],
                offset - half,
                offset + half,
                color=color,
                alpha=0.22,
                linewidth=0,
                zorder=0,
            )

        ax.plot(t_p, y, color=color, lw=0.7, zorder=2)
        y_wall = wall_distance_mm(tr.point.geometry, tr.point.y_mm)
        y_ticks.append(offset)
        y_labels.append(
            rf"$y={y_wall:.0f}\,\mathrm{{mm}}$"
            f"\n"
            rf"$\gamma={100.0 * tr.intermittency:.1f}\%$"
        )

    ax.set_xlabel(r"$t$ in $\mathrm{s}$")
    ax.set_ylabel(
        rf"$u_\mathrm{{hp}}-\langle u_\mathrm{{hp}}\rangle$ "
        rf"(offset by ${trace_offset_sigma:g}\,\sigma$)"
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0.0
        else rf"$u-\langle u\rangle$ (offset by ${trace_offset_sigma:g}\,\sigma$)"
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(float(traces[0].t_s[0]), float(traces[0].t_s[-1]))
    hp_note = (
        rf", HP ${highpass_cutoff_hz:g}\,\mathrm{{Hz}}$"
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0.0
        else ""
    )
    ax.set_title(
        rf"Turbulent bursts: RMS$_{{{short_period_s * 1e3:.0f}\mathrm{{ms}}}}$"
        rf"$(u') > {n_sigma:g}\sigma${hp_note}"
    )
    finalize_figure(fig)

    hp_stem = (
        f"_hp{highpass_cutoff_hz:g}Hz"
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0.0
        else ""
    )
    stem = (
        f"stacked_bursts_{T_START_S:g}-{T_END_S:g}s_"
        f"Nsig{n_sigma:g}_win{short_period_s * 1e3:.0f}ms{hp_stem}"
    ).replace(".", "p")
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


# %%
if __name__ == "__main__":
    all_points = collect_wall_positions(DATA_DIR)
    points = dedupe_wall_positions(all_points)
    points = drop_angled_probe_points(points)
    points = merge_preferring_geometry(points, preferred_geometry="6in")
    subset = select_profile_subset(points, max_traces=MAX_TRACES)

    hp_msg = (
        f"HP {HIGHPASS_CUTOFF_HZ:g} Hz"
        if HIGHPASS_CUTOFF_HZ is not None and HIGHPASS_CUTOFF_HZ > 0.0
        else "no high-pass"
    )
    print(
        f"Using {len(subset)} / {len(points)} wall-normal positions "
        f"on [{T_START_S:g}, {T_END_S:g}] s "
        f"({hp_msg}; criterion: rolling RMS > {N_SIGMA:g}σ over "
        f"{SHORT_PERIOD_S * 1e3:.0f} ms)."
    )
    for p in subset:
        print(
            f"  y_wall={wall_distance_mm(p.geometry, p.y_mm):.0f} mm  "
            f"({p.geometry}, local {p.y_mm:.0f} mm)  {p.run_name}"
        )

    traces = build_burst_traces(
        subset,
        t_start_s=T_START_S,
        t_end_s=T_END_S,
        n_sigma=N_SIGMA,
        short_period_s=SHORT_PERIOD_S,
        min_burst_s=MIN_BURST_S,
        merge_gap_s=MERGE_GAP_S,
        highpass_cutoff_hz=HIGHPASS_CUTOFF_HZ,
        highpass_order=HIGHPASS_ORDER,
    )

    for tr in traces:
        n_runs = len(contiguous_runs(tr.burst))
        print(
            f"  y_wall={wall_distance_mm(tr.point.geometry, tr.point.y_mm):.0f} mm: "
            f"σ={tr.u_std:.4f} m/s, γ={100 * tr.intermittency:.1f}%, "
            f"{n_runs} burst run(s)"
        )

    fig, pdf = plot_stacked_burst_signals(
        traces,
        n_sigma=N_SIGMA,
        short_period_s=SHORT_PERIOD_S,
        max_plot_points=MAX_PLOT_POINTS,
        trace_offset_sigma=TRACE_OFFSET_SIGMA,
        highpass_cutoff_hz=HIGHPASS_CUTOFF_HZ,
    )
    print(f"Wrote {pdf.resolve()}")

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)
