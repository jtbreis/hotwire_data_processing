# %% [markdown]
# # Compare literature burst detectors: VITA vs VITA+LEVEL
#
# Two schemes from the reference literature, applied to streamwise hot-wire
# velocity (same TNTI wall-profile loading as ``hotwire_wall_profiles.py``),
# **without** extra band-pass / high-pass filtering:
#
# 1. **Blackwelder & Kaplan (1976)** — classic **VITA** on local ``u``:
#    ``var_T(u) > k · u_rms²`` with ``k = 1.2`` (paper default).
#
# 2. **Morrison, Tsai & Bradshaw (1989)** — **VITA+LEVEL** on local ``u'``:
#    seed where ``var_T > VITH · σ²`` **and** ``|u'| > TH · σ``, then extend
#    with LEVEL alone. Paper values on ``uv``: ``VITH ≈ 0.2``, ``TH ≈ 0.46``.
#
# Both papers normalize thresholds by the **local** long-time RMS of the
# detection signal at that probe location (one dimensionless ``k`` / ``VITH`` /
# ``TH`` everywhere). They do **not** prescribe a single absolute m/s threshold
# for the whole wall-normal range; Morrison instead argues for wall-unit
# scaling (threshold ∼ ``u_τ``) when comparing across Reynolds numbers.

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import uniform_filter1d

from mcflow_plotting.style.colors import OVERLAY_COLORS
from mcflow_plotting.style.figure import (
    finalize_figure,
    save_figure,
    use_lab_matplotlib_style,
)
from hotwire_turbulent_burst import (
    DATA_DIR,
    HIGHPASS_ORDER,
    INCH_TO_MM,
    MAX_PLOT_POINTS,
    PLOTS_DIR,
    WallPosition,
    collect_wall_positions,
    contiguous_runs,
    dedupe_wall_positions,
    drop_angled_probe_points,
    downsample_for_plot,
    highpass_filter,
    load_velocity_window,
    merge_preferring_geometry,
    wall_distance_mm,
)

# %matplotlib inline

# %%
# Station selection: set ALL_LOCATIONS True to process every merged wall position.
ALL_LOCATIONS: bool = True
Y_WALL_TARGET_MM: float = 252.0  # used only when ALL_LOCATIONS is False
T_START_S: float = 0.0
T_END_S: float = 1.0

# Optional high-pass before detection (None / <=0 disables).
# Papers apply VITA to the raw (demeaned) hot-wire record — no extra filter here.
HIGHPASS_CUTOFF_HZ: float | None = 10.0

# Cutoffs compared in the high-pass sensitivity plot (full-record profiles).
# ``None`` means no high-pass (demeaned only).
HIGHPASS_COMPARE_HZ: tuple[float | None, ...] = (None, 5.0, 10.0, 100.0)

# Shared VITA averaging time T [s].
# Blackwelder's T⁺ ≈ 10 is a near-wall BL scale; for this shear / TNTI flow
# a longer window better matches the slower inactive motions.
VITA_T_S: float = 0.050

# --- Paper defaults: local RMS normalization (same dimensionless thresholds
# at every y). Not a fixed absolute m/s threshold across the profile. ---
THRESHOLD_FROM_TURBULENT_SIDE: bool = False
Y_TURBULENT_MIN_MM: float = 600.0
SIGMA_TURB_OVERRIDE_MS: float | None = None

# Blackwelder & Kaplan (1976): var_T > k · u_rms²  (local u_rms)
# Kept for comparison plots; primary analysis uses VITA+LEVEL below.
BK_K: float = 1.2

# Morrison et al. (1989) VITA+LEVEL on u' (local σ):
#   seed: var_T > VITH · σ² and |u'| > TH · σ; then extend with LEVEL.
# Raised above the paper's uv defaults (VITH≈0.2, TH≈0.46) so fully
# developed / turbulent stations (high local σ, continuous activity) are
# sampled less; VITH=TH=1.2 keeps outer γ below the intermittent side.
MOR_VITH: float = 1.2
MOR_TH: float = 1.2

SHOW_PLOTS = True
FIGSIZE = (10.0, 6.2)
STACK_OFFSET_SIGMA: float = 6.0
STACK_FIGSIZE = (10.0, 11.0)

# Wind-tunnel height for wall-normal scaling (41 in → mm).
TUNNEL_HEIGHT_IN: float = 41.0
TUNNEL_HEIGHT_MM: float = TUNNEL_HEIGHT_IN * INCH_TO_MM


def y_over_Y(y_wall_mm: float | np.ndarray) -> float | np.ndarray:
    """Nondimensional wall distance ``y/Y`` with ``Y = 41 in``."""
    return np.asarray(y_wall_mm, dtype=float) / TUNNEL_HEIGHT_MM


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
def short_time_variance(u: np.ndarray, n_win: int) -> np.ndarray:
    """Centred short-time variance ``⟨u²⟩_T − ⟨u⟩_T²`` (uniform window)."""
    u = np.asarray(u, dtype=float)
    n_win = max(1, int(n_win))
    if n_win % 2 == 0:
        n_win += 1  # odd length → true centre sample
    m1 = uniform_filter1d(u, size=n_win, mode="nearest")
    m2 = uniform_filter1d(u * u, size=n_win, mode="nearest")
    return np.maximum(m2 - m1 * m1, 0.0)


def detect_vita_blackwelder(
    u: np.ndarray,
    fs_hz: float,
    *,
    t_s: float,
    k: float,
    sigma_ref: float | None = None,
) -> np.ndarray:
    """
    Blackwelder & Kaplan (1976) VITA detector.

    ``D = 1`` where short-time variance exceeds ``k · σ_ref²``.
    If ``sigma_ref`` is None, ``σ_ref`` is the local record RMS (classic VITA).
    """
    u = np.asarray(u, dtype=float)
    u_fluc = u - float(np.mean(u))
    if sigma_ref is None:
        u_rms2 = float(np.mean(u_fluc * u_fluc))
    else:
        u_rms2 = float(sigma_ref) ** 2
    if u_rms2 <= 0.0:
        return np.zeros(u.size, dtype=bool)
    n_win = max(1, int(round(t_s * fs_hz)))
    var_t = short_time_variance(u_fluc, n_win)
    return var_t > (k * u_rms2)


def detect_vita_plus_level_morrison(
    u: np.ndarray,
    fs_hz: float,
    *,
    t_s: float,
    vith: float,
    th: float,
    sigma_ref: float | None = None,
) -> np.ndarray:
    """
    Morrison, Tsai & Bradshaw (1989) VITA+LEVEL detector on ``u'``.

    Stage 1 (seed): ``var_T > VITH · σ_ref²`` and ``|u'| > TH · σ_ref``.
    Stage 2 (extend): grow each seed with LEVEL ``|u'| > TH · σ_ref`` until first fail.
    If ``sigma_ref`` is None, ``σ_ref`` is the local record RMS.
    """
    u = np.asarray(u, dtype=float)
    u_fluc = u - float(np.mean(u))
    if sigma_ref is None:
        u_rms = float(np.std(u_fluc, ddof=0))
    else:
        u_rms = float(sigma_ref)
    if u_rms <= 0.0:
        return np.zeros(u.size, dtype=bool)
    u_rms2 = u_rms * u_rms
    n_win = max(1, int(round(t_s * fs_hz)))
    var_t = short_time_variance(u_fluc, n_win)
    level = np.abs(u_fluc) > (th * u_rms)
    seed = (var_t > (vith * u_rms2)) & level

    burst = np.zeros(u.size, dtype=bool)
    n = int(u.size)
    # Extend each contiguous seed run to the enclosing LEVEL plateau.
    for i0, i1 in contiguous_runs(seed):
        left = i0
        while left > 0 and level[left - 1]:
            left -= 1
        right = i1
        while right < n and level[right]:
            right += 1
        burst[left:right] = True
    return burst


def load_processed_velocity(
    point: WallPosition,
    *,
    t_start_s: float | None,
    t_end_s: float | None,
    highpass_hz: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(t, u_processed, fs_hz)`` with optional high-pass."""
    t, u, fs_hz = load_velocity_window(
        point, t_start_s=t_start_s, t_end_s=t_end_s
    )
    if highpass_hz is not None and highpass_hz > 0.0:
        u = highpass_filter(u, fs_hz, cutoff_hz=highpass_hz, order=HIGHPASS_ORDER)
    return t, u, fs_hz


def turbulent_side_sigma(
    points: list[WallPosition],
    *,
    y_min_mm: float,
    t_start_s: float | None,
    t_end_s: float | None,
    highpass_hz: float | None,
) -> tuple[float, list[float]]:
    """
    Pooled RMS of ``u'`` over stations with ``y_wall ≥ y_min_mm``.

    Returns ``(σ_turb, per_station_σ)``.
    """
    chunks: list[np.ndarray] = []
    per_station: list[float] = []
    for point in points:
        y_wall = wall_distance_mm(point.geometry, point.y_mm)
        if y_wall < y_min_mm:
            continue
        _t, u, _fs = load_processed_velocity(
            point,
            t_start_s=t_start_s,
            t_end_s=t_end_s,
            highpass_hz=highpass_hz,
        )
        u_fluc = u - float(np.mean(u))
        per_station.append(float(np.std(u_fluc, ddof=0)))
        chunks.append(u_fluc)
    if not chunks:
        raise ValueError(
            f"No stations with y_wall ≥ {y_min_mm:g} mm to define σ_turb."
        )
    pooled = np.concatenate(chunks)
    return float(np.std(pooled, ddof=0)), per_station


def select_nearest_wall_position(y_wall_mm: float):
    all_points = collect_wall_positions(DATA_DIR)
    points = merge_preferring_geometry(
        drop_angled_probe_points(dedupe_wall_positions(all_points)),
        preferred_geometry="6in",
    )
    return min(
        points,
        key=lambda p: abs(wall_distance_mm(p.geometry, p.y_mm) - y_wall_mm),
    )


def plot_method_comparison(
    t: np.ndarray,
    u: np.ndarray,
    *,
    mask_bk: np.ndarray,
    mask_mor: np.ndarray,
    y_wall_mm: float,
    fs_hz: float,
    vita_t_s: float,
    k: float,
    vith: float,
    th: float,
    highpass_hz: float | None,
    max_plot_points: int,
    save: bool = True,
) -> tuple[plt.Figure, Path]:
    use_lab_matplotlib_style()
    u_fluc = u - float(np.mean(u))
    u_std = float(np.std(u_fluc, ddof=0))
    t_p, u_p, bk_p, mor_p = downsample_for_plot(
        t,
        u_fluc,
        mask_bk.astype(float),
        mask_mor.astype(float),
        max_points=max_plot_points,
    )
    bk_bool = bk_p > 0.5
    mor_bool = mor_p > 0.5

    fig, axes = plt.subplots(
        3,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": (2.4, 0.7, 0.7)},
    )
    ax_u, ax_bk, ax_mor = axes

    # Velocity with both masks shaded (different colours).
    for i0, i1 in contiguous_runs(bk_bool):
        ax_u.axvspan(
            float(t_p[i0]),
            float(t_p[min(i1, t_p.size) - 1]),
            color="#4c78a8",
            alpha=0.25,
            lw=0,
            zorder=0,
        )
    for i0, i1 in contiguous_runs(mor_bool):
        ax_u.axvspan(
            float(t_p[i0]),
            float(t_p[min(i1, t_p.size) - 1]),
            color="#e45756",
            alpha=0.20,
            lw=0,
            zorder=0,
        )
    ax_u.plot(t_p, u_p, color="#003f5c", lw=0.7, zorder=2)
    ax_u.axhline(0.0, color="0.5", lw=0.5, alpha=0.5)
    ax_u.set_ylabel(r"$u-\langle u\rangle$ in $\mathrm{m/s}$")
    hp_note = (
        rf", HP ${highpass_hz:g}\,\mathrm{{Hz}}$"
        if highpass_hz is not None and highpass_hz > 0.0
        else ""
    )
    ax_u.set_title(
        rf"$y/Y={float(y_over_Y(y_wall_mm)):.3f}$"
        rf" ($y={y_wall_mm:.0f}\,\mathrm{{mm}}$, $Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$)"
        rf"{hp_note}, "
        rf"$T={vita_t_s * 1e3:.0f}\,\mathrm{{ms}}$, "
        rf"$\sigma={u_std:.4f}\,\mathrm{{m/s}}$"
        "\n"
        r"shade: blue = Blackwelder VITA, red = Morrison VITA+LEVEL"
    )

    ax_bk.fill_between(t_p, 0.0, bk_p, step="mid", color="#4c78a8", alpha=0.85)
    ax_bk.set_ylim(-0.05, 1.15)
    ax_bk.set_yticks([0, 1])
    gamma_bk = float(np.mean(mask_bk))
    n_bk = len(contiguous_runs(mask_bk))
    ax_bk.set_ylabel(r"$D_\mathrm{VITA}$")
    ax_bk.set_title(
        rf"Blackwelder \& Kaplan (1976): "
        rf"$\mathrm{{var}}_T > {k:g}\,\sigma_\mathrm{{ref}}^2$ "
        rf"($\gamma={100 * gamma_bk:.1f}\%$, {n_bk} events)",
        fontsize=10,
    )

    ax_mor.fill_between(t_p, 0.0, mor_p, step="mid", color="#e45756", alpha=0.85)
    ax_mor.set_ylim(-0.05, 1.15)
    ax_mor.set_yticks([0, 1])
    gamma_mor = float(np.mean(mask_mor))
    n_mor = len(contiguous_runs(mask_mor))
    ax_mor.set_ylabel(r"$D_\mathrm{V+L}$")
    ax_mor.set_xlabel(r"$t$ in $\mathrm{s}$")
    ax_mor.set_title(
        rf"Morrison et al.\ (1989) VITA+LEVEL: "
        rf"VITH$={vith:g}$, TH$={th:g}$ on $\sigma_\mathrm{{ref}}$ "
        rf"($\gamma={100 * gamma_mor:.1f}\%$, {n_mor} events)",
        fontsize=10,
    )
    ax_mor.set_xlim(float(t[0]), float(t[-1]))

    finalize_figure(fig, pad=0.5)

    hp_stem = (
        f"_hp{highpass_hz:g}Hz"
        if highpass_hz is not None and highpass_hz > 0.0
        else "_nofilter"
    )
    stem = (
        f"vita_compare_y{y_wall_mm:.0f}mm_"
        f"{T_START_S:g}-{T_END_S:g}s_T{vita_t_s * 1e3:.0f}ms"
        f"_k{k:g}_VITH{vith:g}_TH{th:g}{hp_stem}"
    ).replace(".", "p")
    pdf = PLOTS_DIR / f"{stem}.pdf"
    if save:
        save_figure(fig, pdf)
    return fig, pdf


def event_durations_s(mask: np.ndarray, fs_hz: float) -> np.ndarray:
    """Durations [s] of contiguous True runs in ``mask``."""
    dt = 1.0 / float(fs_hz)
    return np.asarray(
        [(i1 - i0) * dt for i0, i1 in contiguous_runs(mask)],
        dtype=float,
    )


def event_rate_stats(
    mask: np.ndarray,
    *,
    fs_hz: float,
    record_duration_s: float,
) -> dict[str, float | np.ndarray]:
    """Event count, frequency, and duration moments for one detector mask."""
    durs = event_durations_s(mask, fs_hz)
    n = int(durs.size)
    t_rec = max(float(record_duration_s), 1e-12)
    return {
        "n_events": float(n),
        "freq_hz": float(n) / t_rec,
        "dur_mean_s": float(np.mean(durs)) if n else float("nan"),
        "dur_median_s": float(np.median(durs)) if n else float("nan"),
        "durations_s": durs,
    }


def conditional_region_stats(
    u: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Conditional mean and RMS of velocity in burst vs non-burst samples.

    ``u_mean_*``: sample mean of ``u`` in each region.
    ``u_rms_*``: RMS of fluctuations about the *global* record mean,
    ``sqrt(⟨(u − ⟨u⟩)² | region⟩)``.
    """
    u = np.asarray(u, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    u_bar = float(np.mean(u))
    u_fluc2 = (u - u_bar) ** 2

    def _mean_rms(sel: np.ndarray) -> tuple[float, float]:
        if not np.any(sel):
            return float("nan"), float("nan")
        return float(np.mean(u[sel])), float(np.sqrt(np.mean(u_fluc2[sel])))

    mean_b, rms_b = _mean_rms(mask)
    mean_nb, rms_nb = _mean_rms(~mask)
    return {
        "u_mean_burst": mean_b,
        "u_mean_nonburst": mean_nb,
        "u_rms_burst": rms_b,
        "u_rms_nonburst": rms_nb,
        "u_mean_all": u_bar,
        "u_rms_all": float(np.sqrt(np.mean(u_fluc2))),
    }


def agreement_stats(
    a: np.ndarray,
    b: np.ndarray,
    *,
    fs_hz: float,
    record_duration_s: float,
) -> dict[str, float | np.ndarray]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    both = a & b
    either = a | b
    ea = event_rate_stats(a, fs_hz=fs_hz, record_duration_s=record_duration_s)
    eb = event_rate_stats(b, fs_hz=fs_hz, record_duration_s=record_duration_s)
    return {
        "gamma_a": float(np.mean(a)),
        "gamma_b": float(np.mean(b)),
        "overlap_of_a": float(np.mean(both[a])) if np.any(a) else float("nan"),
        "overlap_of_b": float(np.mean(both[b])) if np.any(b) else float("nan"),
        "jaccard": float(np.mean(both) / np.mean(either)) if np.any(either) else float("nan"),
        "n_events_a": ea["n_events"],
        "n_events_b": eb["n_events"],
        "freq_a_hz": ea["freq_hz"],
        "freq_b_hz": eb["freq_hz"],
        "dur_mean_a_s": ea["dur_mean_s"],
        "dur_mean_b_s": eb["dur_mean_s"],
        "dur_median_a_s": ea["dur_median_s"],
        "dur_median_b_s": eb["dur_median_s"],
        "durations_a_s": ea["durations_s"],
        "durations_b_s": eb["durations_s"],
        "record_duration_s": float(record_duration_s),
    }


@dataclass(frozen=True)
class LocationResult:
    point: WallPosition
    y_wall_mm: float
    t_s: np.ndarray
    u: np.ndarray
    u_std: float
    mask_bk: np.ndarray
    mask_mor: np.ndarray
    stats: dict[str, float | np.ndarray]


def all_merged_wall_positions() -> list[WallPosition]:
    all_points = collect_wall_positions(DATA_DIR)
    points = dedupe_wall_positions(all_points)
    points = drop_angled_probe_points(points)
    return merge_preferring_geometry(points, preferred_geometry="6in")


def analyze_location(
    point: WallPosition,
    *,
    t_start_s: float | None,
    t_end_s: float | None,
    highpass_hz: float | None,
    vita_t_s: float,
    k: float,
    vith: float,
    th: float,
    sigma_ref: float | None = None,
) -> LocationResult:
    # Raw calibrated velocity for conditional means/RMS; detection on processed.
    t, u_raw, fs_hz = load_velocity_window(
        point, t_start_s=t_start_s, t_end_s=t_end_s
    )
    if highpass_hz is not None and highpass_hz > 0.0:
        u = highpass_filter(
            u_raw, fs_hz, cutoff_hz=highpass_hz, order=HIGHPASS_ORDER
        )
    else:
        u = u_raw
    mask_bk = detect_vita_blackwelder(
        u, fs_hz, t_s=vita_t_s, k=k, sigma_ref=sigma_ref
    )
    mask_mor = detect_vita_plus_level_morrison(
        u, fs_hz, t_s=vita_t_s, vith=vith, th=th, sigma_ref=sigma_ref
    )
    u_fluc = u - float(np.mean(u))
    record_duration_s = (
        float(t[-1] - t[0]) if t.size > 1 else float(u.size) / float(fs_hz)
    )
    stats = agreement_stats(
        mask_bk,
        mask_mor,
        fs_hz=fs_hz,
        record_duration_s=record_duration_s,
    )
    # Condition *raw* U on detector masks (HP signal has ~zero mean).
    cond_bk = conditional_region_stats(u_raw, mask_bk)
    cond_mor = conditional_region_stats(u_raw, mask_mor)
    for key, val in cond_bk.items():
        stats[f"{key}_a"] = val
    for key, val in cond_mor.items():
        stats[f"{key}_b"] = val
    return LocationResult(
        point=point,
        y_wall_mm=wall_distance_mm(point.geometry, point.y_mm),
        t_s=t,
        u=u,
        u_std=float(np.std(u_fluc, ddof=0)),
        mask_bk=mask_bk,
        mask_mor=mask_mor,
        stats=stats,
    )


def _output_stem(
    *,
    y_tag: str,
    vita_t_s: float,
    k: float,
    vith: float,
    th: float,
    highpass_hz: float | None,
    sigma_ref: float | None = None,
) -> str:
    hp_stem = (
        f"_hp{highpass_hz:g}Hz"
        if highpass_hz is not None and highpass_hz > 0.0
        else "_nofilter"
    )
    ref_stem = (
        f"_sigTurb{sigma_ref:.4g}".replace(".", "p")
        if sigma_ref is not None
        else "_sigLocal"
    )
    return (
        f"vita_compare_{y_tag}_"
        f"{T_START_S:g}-{T_END_S:g}s_T{vita_t_s * 1e3:.0f}ms"
        f"_k{k:g}_VITH{vith:g}_TH{th:g}{ref_stem}{hp_stem}"
    ).replace(".", "p")


def _full_record_stem(y_tag: str, *, sigma_ref: float | None) -> str:
    stem = _output_stem(
        y_tag=y_tag,
        vita_t_s=VITA_T_S,
        k=BK_K,
        vith=MOR_VITH,
        th=MOR_TH,
        highpass_hz=HIGHPASS_CUTOFF_HZ,
        sigma_ref=sigma_ref,
    )
    return stem.replace(f"_{T_START_S:g}-{T_END_S:g}s", "_fullRecord")


def _profile_ylim(ys: np.ndarray) -> tuple[float, float]:
    """Wall-normal axis span in y/Y: full tunnel height [0, 1]."""
    del ys  # data range unused; always show full height
    return 0.0, 1.0


def _profile_y_h(results: list[LocationResult]) -> np.ndarray:
    return y_over_Y(np.asarray([r.y_wall_mm for r in results], dtype=float))


def mean_interface_from_gamma(
    y: np.ndarray,
    gamma: np.ndarray,
) -> dict[str, float]:
    """
    Mean interface location from an intermittency profile γ(y).

    Definitions (all returned in the same units as ``y``):

    - ``y_gamma_half``: where γ crosses 0.5 (NaN if it never does).
    - ``y_gamma_mid``: where γ crosses the mid-level
      ``½(γ_min + γ_max)`` (robust when γ does not span 0–1).
    - ``y_pdf_mean``: first moment of the interface PDF
      ``p(y) ∝ |dγ/dy|`` (Corrsin / TNTI style).
    - ``y_pdf_mode``: location of max ``|dγ/dy|``.
    """
    y = np.asarray(y, dtype=float)
    g = np.asarray(gamma, dtype=float)
    order = np.argsort(y)
    y = y[order]
    g = g[order]

    def _crossing(level: float) -> float:
        above = g >= level
        if not np.any(above) or np.all(above):
            return float("nan")
        # First crossing along increasing y.
        for i in range(len(g) - 1):
            g0, g1 = g[i], g[i + 1]
            if (g0 - level) * (g1 - level) <= 0.0 and g1 != g0:
                w = (level - g0) / (g1 - g0)
                return float(y[i] + w * (y[i + 1] - y[i]))
        return float("nan")

    g_min = float(np.min(g))
    g_max = float(np.max(g))
    g_mid = 0.5 * (g_min + g_max)

    dy = np.diff(y)
    dg = np.diff(g)
    # Interface density on segment midpoints.
    y_mid = 0.5 * (y[:-1] + y[1:])
    w = np.abs(dg)
    # Prefer physical spacing: density ~ |dγ/dy| Δy ≡ |Δγ|.
    w_sum = float(np.sum(w))
    if w_sum > 0.0:
        y_pdf_mean = float(np.sum(y_mid * w) / w_sum)
        i_mode = int(np.argmax(np.abs(dg / np.maximum(dy, 1e-12))))
        y_pdf_mode = float(y_mid[i_mode])
    else:
        y_pdf_mean = float("nan")
        y_pdf_mode = float("nan")

    return {
        "y_gamma_half": _crossing(0.5),
        "y_gamma_mid": _crossing(g_mid),
        "gamma_min": g_min,
        "gamma_max": g_max,
        "gamma_mid_level": g_mid,
        "y_pdf_mean": y_pdf_mean,
        "y_pdf_mode": y_pdf_mode,
    }


def plot_gamma_profile(
    results: list[LocationResult],
    *,
    sigma_ref: float | None = None,
    full_record: bool = True,
) -> tuple[plt.Figure, Path]:
    """Intermittency γ(y/Y) with mean-interface markers."""
    use_lab_matplotlib_style()
    ys = _profile_y_h(results)
    g_bk = np.asarray([r.stats["gamma_a"] for r in results], dtype=float)
    g_mor = np.asarray([r.stats["gamma_b"] for r in results], dtype=float)
    iface_bk = mean_interface_from_gamma(ys, g_bk)
    iface_mor = mean_interface_from_gamma(ys, g_mor)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(g_bk * 100.0, ys, "o-", color="#4c78a8", lw=1.2, ms=5, label="VITA")
    ax.plot(
        g_mor * 100.0,
        ys,
        "s-",
        color="#e45756",
        lw=1.2,
        ms=5,
        label="VITA+LEVEL",
    )

    # Mark mean interface (VITA+LEVEL primary; VITA secondary).
    for iface, color, name, lw in (
        (iface_mor, "#e45756", "VITA+LEVEL", 1.3),
        (iface_bk, "#4c78a8", "VITA", 1.0),
    ):
        y_m = iface["y_pdf_mean"]
        if np.isfinite(y_m):
            ax.axhline(
                y_m,
                color=color,
                ls="-",
                lw=lw,
                alpha=0.9,
                label=rf"$\langle y/Y\rangle_\gamma$ {name} $={y_m:.3f}$",
            )
        y_half = iface["y_gamma_half"]
        if np.isfinite(y_half):
            ax.axhline(
                y_half,
                color=color,
                ls=":",
                lw=lw,
                alpha=0.75,
                label=rf"$y/Y|_{{\gamma=0.5}}$ {name} $={y_half:.3f}$",
            )
        elif np.isfinite(iface["y_gamma_mid"]):
            y_mid = iface["y_gamma_mid"]
            ax.axhline(
                y_mid,
                color=color,
                ls="--",
                lw=lw,
                alpha=0.6,
                label=rf"$y/Y|_{{\gamma_{{\mathrm{{mid}}}}}}$ {name} $={y_mid:.3f}$",
            )

    ax.set_xlabel(r"intermittency $\gamma$ in $\%$")
    ax.set_ylabel(r"$y/Y$")
    win_note = "full record" if full_record else rf"$t\in[{T_START_S:g},{T_END_S:g}]\,\mathrm{{s}}$"
    if sigma_ref is not None:
        ax.set_title(
            rf"Intermittency vs height ({win_note}, "
            rf"$\sigma_\mathrm{{turb}}={sigma_ref:.4f}\,\mathrm{{m/s}}$)"
        )
    else:
        ax.set_title(
            rf"Intermittency vs height "
            rf"($Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$, {win_note}, "
            rf"$T={VITA_T_S * 1e3:.0f}\,\mathrm{{ms}}$, local $\sigma$)"
        )
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_ylim(*_profile_ylim(ys))
    ax.set_xlim(left=0.0)
    finalize_figure(fig)
    stem = _full_record_stem("all-y_gamma", sigma_ref=sigma_ref) if full_record else _output_stem(
        y_tag="all-y_gamma",
        vita_t_s=VITA_T_S,
        k=BK_K,
        vith=MOR_VITH,
        th=MOR_TH,
        highpass_hz=HIGHPASS_CUTOFF_HZ,
        sigma_ref=sigma_ref,
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf, iface_bk, iface_mor


def plot_event_frequency_profile(
    results: list[LocationResult],
    *,
    sigma_ref: float | None = None,
    full_record: bool = True,
) -> tuple[plt.Figure, Path]:
    """Detected event frequency f_e = N_events / T_record vs y/Y."""
    use_lab_matplotlib_style()
    ys = _profile_y_h(results)
    f_bk = np.asarray([r.stats["freq_a_hz"] for r in results], dtype=float)
    f_mor = np.asarray([r.stats["freq_b_hz"] for r in results], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(f_bk, ys, "o-", color="#4c78a8", lw=1.2, ms=5, label="VITA")
    ax.plot(f_mor, ys, "s-", color="#e45756", lw=1.2, ms=5, label="VITA+LEVEL")
    ax.set_xlabel(r"event frequency $f_e$ in $\mathrm{Hz}$")
    ax.set_ylabel(r"$y/Y$")
    win_note = "full record" if full_record else rf"$t\in[{T_START_S:g},{T_END_S:g}]\,\mathrm{{s}}$"
    ax.set_title(
        rf"Detected event frequency vs $y/Y$ "
        rf"($Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$, {win_note}, "
        rf"$T={VITA_T_S * 1e3:.0f}\,\mathrm{{ms}}$)"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(*_profile_ylim(ys))
    finalize_figure(fig)
    stem = (
        _full_record_stem("all-y_freq", sigma_ref=sigma_ref)
        if full_record
        else _output_stem(
            y_tag="all-y_freq",
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            sigma_ref=sigma_ref,
        )
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def plot_event_count_profile(
    results: list[LocationResult],
    *,
    sigma_ref: float | None = None,
    full_record: bool = True,
) -> tuple[plt.Figure, Path]:
    """Number of intermittent (contiguous) events vs y/Y."""
    use_lab_matplotlib_style()
    ys = _profile_y_h(results)
    n_bk = np.asarray([r.stats["n_events_a"] for r in results], dtype=float)
    n_mor = np.asarray([r.stats["n_events_b"] for r in results], dtype=float)
    t_rec = float(results[0].stats["record_duration_s"]) if results else float("nan")

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(n_bk, ys, "o-", color="#4c78a8", lw=1.2, ms=5, label="VITA")
    ax.plot(n_mor, ys, "s-", color="#e45756", lw=1.2, ms=5, label="VITA+LEVEL")
    ax.set_xlabel(r"number of intermittent events $N$")
    ax.set_ylabel(r"$y/Y$")
    win_note = "full record" if full_record else rf"$t\in[{T_START_S:g},{T_END_S:g}]\,\mathrm{{s}}$"
    ax.set_title(
        rf"Event count vs $y/Y$ "
        rf"($Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$, {win_note}, "
        rf"$T_{{\mathrm{{rec}}}}={t_rec:.0f}\,\mathrm{{s}}$, "
        rf"$T={VITA_T_S * 1e3:.0f}\,\mathrm{{ms}}$)"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(*_profile_ylim(ys))
    finalize_figure(fig)
    stem = (
        _full_record_stem("all-y_nEvents", sigma_ref=sigma_ref)
        if full_record
        else _output_stem(
            y_tag="all-y_nEvents",
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            sigma_ref=sigma_ref,
        )
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def plot_conditional_velocity_profiles(
    results: list[LocationResult],
    *,
    sigma_ref: float | None = None,
    full_record: bool = True,
) -> tuple[plt.Figure, Path]:
    """
    Conditional mean and RMS of *raw* U in burst vs non-burst regions
    (masks from the processed / high-passed detector).
    """
    use_lab_matplotlib_style()
    ys = _profile_y_h(results)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0), sharey=True)
    win_note = "full record" if full_record else rf"$t\in[{T_START_S:g},{T_END_S:g}]\,\mathrm{{s}}$"

    # Left: conditional means (VITA+LEVEL primary; VITA dashed)
    ax = axes[0]
    ax.plot(
        [float(r.stats["u_mean_burst_b"]) for r in results],
        ys,
        "s-",
        color="#e45756",
        lw=1.2,
        ms=5,
        label=r"VITA+LEVEL burst",
    )
    ax.plot(
        [float(r.stats["u_mean_nonburst_b"]) for r in results],
        ys,
        "s--",
        color="#e45756",
        lw=1.0,
        ms=4,
        alpha=0.75,
        label=r"VITA+LEVEL non-burst",
    )
    ax.plot(
        [float(r.stats["u_mean_burst_a"]) for r in results],
        ys,
        "o-",
        color="#4c78a8",
        lw=1.0,
        ms=4,
        alpha=0.85,
        label=r"VITA burst",
    )
    ax.plot(
        [float(r.stats["u_mean_nonburst_a"]) for r in results],
        ys,
        "o--",
        color="#4c78a8",
        lw=0.9,
        ms=3,
        alpha=0.65,
        label=r"VITA non-burst",
    )
    ax.plot(
        [float(r.stats["u_mean_all_b"]) for r in results],
        ys,
        ":",
        color="0.35",
        lw=1.0,
        label=r"$\langle U\rangle$ (all)",
    )
    ax.set_xlabel(r"conditional mean $U$ in $\mathrm{m/s}$")
    ax.set_ylabel(r"$y/Y$")
    ax.set_title(r"Mean velocity")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_ylim(0.0, 1.0)

    # Right: conditional RMS about global mean
    ax = axes[1]
    ax.plot(
        [float(r.stats["u_rms_burst_b"]) for r in results],
        ys,
        "s-",
        color="#e45756",
        lw=1.2,
        ms=5,
        label=r"VITA+LEVEL burst",
    )
    ax.plot(
        [float(r.stats["u_rms_nonburst_b"]) for r in results],
        ys,
        "s--",
        color="#e45756",
        lw=1.0,
        ms=4,
        alpha=0.75,
        label=r"VITA+LEVEL non-burst",
    )
    ax.plot(
        [float(r.stats["u_rms_burst_a"]) for r in results],
        ys,
        "o-",
        color="#4c78a8",
        lw=1.0,
        ms=4,
        alpha=0.85,
        label=r"VITA burst",
    )
    ax.plot(
        [float(r.stats["u_rms_nonburst_a"]) for r in results],
        ys,
        "o--",
        color="#4c78a8",
        lw=0.9,
        ms=3,
        alpha=0.65,
        label=r"VITA non-burst",
    )
    ax.plot(
        [float(r.stats["u_rms_all_b"]) for r in results],
        ys,
        ":",
        color="0.35",
        lw=1.0,
        label=r"$u_{\mathrm{rms}}$ (all)",
    )
    ax.set_xlabel(r"conditional RMS in $\mathrm{m/s}$")
    ax.set_title(r"Fluctuation RMS about $\langle U\rangle$")
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_ylim(0.0, 1.0)

    fig.suptitle(
        rf"Burst vs non-burst conditional velocity ({win_note}, "
        rf"raw $U$ on detector masks, $Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$)",
        fontsize=10,
    )
    finalize_figure(fig, pad=0.35)
    stem = (
        _full_record_stem("all-y_condU", sigma_ref=sigma_ref)
        if full_record
        else _output_stem(
            y_tag="all-y_condU",
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            sigma_ref=sigma_ref,
        )
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def plot_mean_duration_profile(
    results: list[LocationResult],
    *,
    sigma_ref: float | None = None,
    full_record: bool = True,
) -> tuple[plt.Figure, Path]:
    """Mean contiguous burst duration vs y/Y."""
    use_lab_matplotlib_style()
    ys = _profile_y_h(results)
    d_bk = np.asarray([r.stats["dur_mean_a_s"] for r in results], dtype=float) * 1e3
    d_mor = np.asarray([r.stats["dur_mean_b_s"] for r in results], dtype=float) * 1e3
    med_bk = np.asarray([r.stats["dur_median_a_s"] for r in results], dtype=float) * 1e3
    med_mor = np.asarray([r.stats["dur_median_b_s"] for r in results], dtype=float) * 1e3

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot(d_bk, ys, "o-", color="#4c78a8", lw=1.2, ms=5, label=r"VITA mean")
    ax.plot(d_mor, ys, "s-", color="#e45756", lw=1.2, ms=5, label=r"VITA+LEVEL mean")
    ax.plot(med_bk, ys, "o--", color="#4c78a8", lw=1.0, ms=4, alpha=0.7, label=r"VITA median")
    ax.plot(
        med_mor,
        ys,
        "s--",
        color="#e45756",
        lw=1.0,
        ms=4,
        alpha=0.7,
        label=r"VITA+LEVEL median",
    )
    ax.axvline(
        VITA_T_S * 1e3,
        color="0.45",
        ls=":",
        lw=1.0,
        label=rf"$T={VITA_T_S * 1e3:.0f}\,\mathrm{{ms}}$",
    )
    ax.set_xlabel(r"burst duration in $\mathrm{ms}$")
    ax.set_ylabel(r"$y/Y$")
    win_note = "full record" if full_record else rf"$t\in[{T_START_S:g},{T_END_S:g}]\,\mathrm{{s}}$"
    ax.set_title(
        rf"Burst duration vs $y/Y$ "
        rf"($Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$, {win_note})"
    )
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_ylim(*_profile_ylim(ys))
    finalize_figure(fig)
    stem = (
        _full_record_stem("all-y_duration", sigma_ref=sigma_ref)
        if full_record
        else _output_stem(
            y_tag="all-y_duration",
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            sigma_ref=sigma_ref,
        )
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def plot_duration_pdf(
    results: list[LocationResult],
    *,
    sigma_ref: float | None = None,
    full_record: bool = True,
) -> tuple[plt.Figure, Path]:
    """Pooled event-duration PDFs (all wall stations) for both detectors."""
    use_lab_matplotlib_style()
    d_bk = np.concatenate(
        [np.asarray(r.stats["durations_a_s"], dtype=float) for r in results]
    )
    d_mor = np.concatenate(
        [np.asarray(r.stats["durations_b_s"], dtype=float) for r in results]
    )
    d_bk_ms = d_bk * 1e3
    d_mor_ms = d_mor * 1e3

    # Log-spaced bins so short VITA events and longer LEVEL events share a scale.
    d_all = np.concatenate([d_bk_ms[d_bk_ms > 0], d_mor_ms[d_mor_ms > 0]])
    if d_all.size == 0:
        bins = np.linspace(0.0, max(VITA_T_S * 1e3 * 2.0, 1.0), 40)
    else:
        lo = max(float(np.min(d_all)), 1e-3)
        hi = float(np.percentile(d_all, 99.5))
        hi = max(hi, VITA_T_S * 1e3 * 1.5, lo * 1.1)
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    if d_bk_ms.size:
        ax.hist(
            d_bk_ms,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.6,
            color="#4c78a8",
            label=rf"VITA ($N={d_bk_ms.size}$)",
        )
    if d_mor_ms.size:
        ax.hist(
            d_mor_ms,
            bins=bins,
            density=True,
            histtype="step",
            lw=1.6,
            color="#e45756",
            label=rf"VITA+LEVEL ($N={d_mor_ms.size}$)",
        )
    ax.axvline(
        VITA_T_S * 1e3,
        color="0.45",
        ls=":",
        lw=1.2,
        label=rf"$T={VITA_T_S * 1e3:.0f}\,\mathrm{{ms}}$",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"burst duration in $\mathrm{ms}$")
    ax.set_ylabel(r"PDF")
    win_note = "full record" if full_record else rf"$t\in[{T_START_S:g},{T_END_S:g}]\,\mathrm{{s}}$"
    ax.set_title(
        rf"Event-duration PDF (pooled over $y/Y$, {win_note}; "
        rf"$Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$)"
    )
    ax.legend(frameon=False, fontsize=8)
    finalize_figure(fig)
    stem = (
        _full_record_stem("all-y_durationPDF", sigma_ref=sigma_ref)
        if full_record
        else _output_stem(
            y_tag="all-y_durationPDF",
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            sigma_ref=sigma_ref,
        )
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def _hp_label(highpass_hz: float | None) -> str:
    if highpass_hz is None or highpass_hz <= 0.0:
        return "no HP"
    return rf"HP ${highpass_hz:g}\,\mathrm{{Hz}}$"


def _hp_stem_tag(highpass_hz: float | None) -> str:
    if highpass_hz is None or highpass_hz <= 0.0:
        return "nofilter"
    return f"hp{highpass_hz:g}Hz".replace(".", "p")


def plot_highpass_comparison(
    points: list[WallPosition],
    *,
    highpass_list: tuple[float | None, ...] = HIGHPASS_COMPARE_HZ,
    sigma_ref: float | None = None,
) -> tuple[plt.Figure, Path]:
    """
    Full-record γ, event frequency, and mean duration vs y/Y for several
    high-pass cutoffs (VITA and VITA+LEVEL side by side).
    """
    use_lab_matplotlib_style()
    series: list[tuple[float | None, list[LocationResult]]] = []
    for hp in highpass_list:
        results = [
            analyze_location(
                p,
                t_start_s=None,
                t_end_s=None,
                highpass_hz=hp,
                vita_t_s=VITA_T_S,
                k=BK_K,
                vith=MOR_VITH,
                th=MOR_TH,
                sigma_ref=sigma_ref,
            )
            for p in points
        ]
        series.append((hp, results))
        print(f"  high-pass compare: analyzed {_hp_label(hp)} ({len(results)} stations)")

    fig, axes = plt.subplots(4, 2, figsize=(9.5, 11.5), sharey=True)
    metrics = (
        ("gamma_a", "gamma_b", r"$\gamma$ in $\%$", 100.0),
        ("n_events_a", "n_events_b", r"number of events $N$", 1.0),
        ("freq_a_hz", "freq_b_hz", r"$f_e$ in $\mathrm{Hz}$", 1.0),
        ("dur_mean_a_s", "dur_mean_b_s", r"mean duration in $\mathrm{ms}$", 1e3),
    )
    col_titles = ("VITA", "VITA+LEVEL")

    for row, (key_a, key_b, xlabel, scale) in enumerate(metrics):
        for col, key in enumerate((key_a, key_b)):
            ax = axes[row, col]
            for i, (hp, results) in enumerate(series):
                ys = _profile_y_h(results)
                vals = np.asarray([r.stats[key] for r in results], dtype=float) * scale
                color = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
                ax.plot(
                    vals,
                    ys,
                    "o-",
                    color=color,
                    lw=1.2,
                    ms=4,
                    label=_hp_label(hp),
                )
            ax.set_xlabel(xlabel)
            ax.set_ylim(0.0, 1.0)
            if col == 0:
                ax.set_ylabel(r"$y/Y$")
            if row == 0:
                ax.set_title(col_titles[col])
            if row == 0 and col == 1:
                ax.legend(frameon=False, fontsize=7, loc="best")
            if row == 3:
                ax.axvline(
                    VITA_T_S * 1e3,
                    color="0.55",
                    ls=":",
                    lw=0.9,
                    zorder=0,
                )

    fig.suptitle(
        rf"High-pass sensitivity (full record, $T={VITA_T_S * 1e3:.0f}\,\mathrm{{ms}}$, "
        rf"$Y={TUNNEL_HEIGHT_IN:g}\,\mathrm{{in}}$)",
        fontsize=11,
    )
    finalize_figure(fig, pad=0.35)
    hp_tags = "-".join(_hp_stem_tag(hp) for hp in highpass_list)
    stem = (
        f"vita_compare_all-y_hpCompare_{hp_tags}_fullRecord_"
        f"T{VITA_T_S * 1e3:.0f}ms_k{BK_K:g}_VITH{MOR_VITH:g}_TH{MOR_TH:g}"
        + ("_sigTurb" if sigma_ref is not None else "_sigLocal")
    ).replace(".", "p")
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def plot_highpass_time_signals(
    *,
    y_targets_mm: tuple[float, ...] = (252.0, 502.0),
    highpass_list: tuple[float | None, ...] = HIGHPASS_COMPARE_HZ,
    t_start_s: float = T_START_S,
    t_end_s: float = T_END_S,
    max_plot_points: int = MAX_PLOT_POINTS,
) -> tuple[plt.Figure, Path]:
    """
    Time traces of demeaned ``u`` at selected wall stations for each high-pass
    cutoff (same window as the zoomed detector plots).
    """
    use_lab_matplotlib_style()
    n_hp = len(highpass_list)
    n_y = len(y_targets_mm)
    fig, axes = plt.subplots(
        n_hp,
        n_y,
        figsize=(4.2 * n_y, 1.55 * n_hp + 0.8),
        sharex=True,
        squeeze=False,
    )

    selected: list[tuple[float, WallPosition]] = []
    for y_tgt in y_targets_mm:
        point = select_nearest_wall_position(y_tgt)
        y_wall = wall_distance_mm(point.geometry, point.y_mm)
        selected.append((y_wall, point))

    for col, (y_wall, point) in enumerate(selected):
        # Load once without HP; re-filter for each cutoff.
        t, u_raw, fs_hz = load_velocity_window(
            point, t_start_s=t_start_s, t_end_s=t_end_s
        )
        for row, hp in enumerate(highpass_list):
            ax = axes[row, col]
            if hp is not None and hp > 0.0:
                u = highpass_filter(
                    u_raw, fs_hz, cutoff_hz=hp, order=HIGHPASS_ORDER
                )
            else:
                u = u_raw
            u_fluc = u - float(np.mean(u))
            t_p, u_p = downsample_for_plot(t, u_fluc, max_points=max_plot_points)
            color = OVERLAY_COLORS[row % len(OVERLAY_COLORS)]
            ax.plot(t_p, u_p, color=color, lw=0.7)
            ax.axhline(0.0, color="0.55", lw=0.4, alpha=0.5)
            ax.set_ylabel(
                rf"{_hp_label(hp)}" "\n" r"$u'$ in $\mathrm{m/s}$",
                fontsize=8,
            )
            if row == 0:
                ax.set_title(
                    rf"$y/Y={float(y_over_Y(y_wall)):.2f}$ "
                    rf"($y={y_wall:.0f}\,\mathrm{{mm}}$)"
                )
            if row == n_hp - 1:
                ax.set_xlabel(r"$t$ in $\mathrm{s}$")
            sigma = float(np.std(u_fluc, ddof=0))
            ax.text(
                0.98,
                0.92,
                rf"$\sigma={sigma:.3f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="0.25",
            )

    fig.suptitle(
        rf"Effect of high-pass on $u'$ "
        rf"($t\in[{t_start_s:g},{t_end_s:g}]\,\mathrm{{s}}$)",
        fontsize=11,
    )
    finalize_figure(fig, pad=0.4)
    y_tag = "-".join(f"y{y:.0f}" for y, _ in selected)
    hp_tags = "-".join(_hp_stem_tag(hp) for hp in highpass_list)
    stem = (
        f"vita_compare_{y_tag}_hpTime_{hp_tags}_"
        f"{t_start_s:g}-{t_end_s:g}s"
    ).replace(".", "p")
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def plot_stacked_all_locations(
    results: list[LocationResult],
    *,
    max_plot_points: int,
    offset_sigma: float,
    sigma_ref: float | None = None,
) -> tuple[plt.Figure, Path]:
    """
    Stacked u' traces for every wall station, shaded like the single-station
    comparison (blue = VITA, red = VITA+LEVEL).
    """
    use_lab_matplotlib_style()
    fig, ax = plt.subplots(figsize=STACK_FIGSIZE)
    y_ticks: list[float] = []
    y_labels: list[str] = []

    for i, r in enumerate(results):
        offset = i * offset_sigma * r.u_std
        u_fluc = r.u - float(np.mean(r.u))
        t_p, u_p, bk_p, mor_p = downsample_for_plot(
            r.t_s,
            u_fluc,
            r.mask_bk.astype(float),
            r.mask_mor.astype(float),
            max_points=max_plot_points,
        )
        half = 0.45 * offset_sigma * r.u_std
        for i0, i1 in contiguous_runs(bk_p > 0.5):
            ax.fill_between(
                [float(t_p[i0]), float(t_p[min(i1, t_p.size) - 1])],
                offset - half,
                offset + half,
                color="#4c78a8",
                alpha=0.22,
                linewidth=0,
                zorder=0,
            )
        for i0, i1 in contiguous_runs(mor_p > 0.5):
            ax.fill_between(
                [float(t_p[i0]), float(t_p[min(i1, t_p.size) - 1])],
                offset - half,
                offset + half,
                color="#e45756",
                alpha=0.18,
                linewidth=0,
                zorder=0,
            )
        ax.plot(t_p, u_p + offset, color="#003f5c", lw=0.55, zorder=2)
        ax.axhline(offset, color="0.6", lw=0.4, alpha=0.4, zorder=1)
        y_ticks.append(offset)
        y_labels.append(
            rf"$y/Y={float(y_over_Y(r.y_wall_mm)):.2f}$"
            f"\n"
            rf"$\gamma_\mathrm{{V}}={100 * float(r.stats['gamma_a']):.1f}\%$"
            f"\n"
            rf"$\gamma_\mathrm{{L}}={100 * float(r.stats['gamma_b']):.1f}\%$"
        )

    ax.set_xlabel(r"$t$ in $\mathrm{s}$")
    ax.set_ylabel(
        rf"$u-\langle u\rangle$ (offset by ${offset_sigma:g}\,\sigma$)"
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlim(float(results[0].t_s[0]), float(results[0].t_s[-1]))
    hp_note = (
        rf", HP ${HIGHPASS_CUTOFF_HZ:g}\,\mathrm{{Hz}}$"
        if HIGHPASS_CUTOFF_HZ is not None and HIGHPASS_CUTOFF_HZ > 0.0
        else ""
    )
    ref_note = (
        rf", $\sigma_\mathrm{{turb}}={sigma_ref:.4f}\,\mathrm{{m/s}}$"
        if sigma_ref is not None
        else ", local $\sigma$"
    )
    ax.set_title(
        rf"All stations{hp_note}{ref_note}: blue = VITA, red = VITA+LEVEL"
    )
    finalize_figure(fig, pad=0.4)
    stem = _output_stem(
        y_tag="all-y_stacked",
        vita_t_s=VITA_T_S,
        k=BK_K,
        vith=MOR_VITH,
        th=MOR_TH,
        highpass_hz=HIGHPASS_CUTOFF_HZ,
        sigma_ref=sigma_ref,
    )
    pdf = PLOTS_DIR / f"{stem}.pdf"
    save_figure(fig, pdf)
    return fig, pdf


def write_multipage_comparisons(
    results: list[LocationResult],
    *,
    vita_t_s: float,
    k: float,
    vith: float,
    th: float,
    highpass_hz: float | None,
    max_plot_points: int,
    sigma_ref: float | None = None,
) -> Path:
    """One 3-panel comparison page per wall station (same layout as single-y plot)."""
    stem = _output_stem(
        y_tag="all-y_pages",
        vita_t_s=vita_t_s,
        k=k,
        vith=vith,
        th=th,
        highpass_hz=highpass_hz,
        sigma_ref=sigma_ref,
    )
    pdf_path = PLOTS_DIR / f"{stem}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for r in results:
            fig, _ = plot_method_comparison(
                r.t_s,
                r.u,
                mask_bk=r.mask_bk,
                mask_mor=r.mask_mor,
                y_wall_mm=r.y_wall_mm,
                fs_hz=1.0,  # unused in plot body
                vita_t_s=vita_t_s,
                k=k,
                vith=vith,
                th=th,
                highpass_hz=highpass_hz,
                max_plot_points=max_plot_points,
                save=False,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return pdf_path


# %%
if __name__ == "__main__":
    if ALL_LOCATIONS:
        points = all_merged_wall_positions()
        print(
            f"Analyzing {len(points)} wall-normal stations "
            f"on [{T_START_S:g}, {T_END_S:g}] s."
        )
    else:
        points = [select_nearest_wall_position(Y_WALL_TARGET_MM)]
        print(
            f"Station: {points[0].run_name} "
            f"(y={wall_distance_mm(points[0].geometry, points[0].y_mm):.1f} mm), "
            f"window [{T_START_S:g}, {T_END_S:g}] s"
        )

    if HIGHPASS_CUTOFF_HZ is not None and HIGHPASS_CUTOFF_HZ > 0.0:
        print(f"Applied high-pass at {HIGHPASS_CUTOFF_HZ:g} Hz.")
    else:
        print("No high-pass filter.")

    sigma_ref: float | None = None
    if THRESHOLD_FROM_TURBULENT_SIDE:
        if SIGMA_TURB_OVERRIDE_MS is not None:
            sigma_ref = float(SIGMA_TURB_OVERRIDE_MS)
            print(
                f"Turbulent-side reference: fixed σ_turb = {sigma_ref:.5f} m/s "
                f"(override; y ≥ {Y_TURBULENT_MIN_MM:g} mm not used for RMS)."
            )
        else:
            sigma_ref, per_sigma = turbulent_side_sigma(
                all_merged_wall_positions(),
                y_min_mm=Y_TURBULENT_MIN_MM,
                t_start_s=T_START_S,
                t_end_s=T_END_S,
                highpass_hz=HIGHPASS_CUTOFF_HZ,
            )
            print(
                f"Turbulent-side reference: y ≥ {Y_TURBULENT_MIN_MM:g} mm, "
                f"σ_turb = {sigma_ref:.5f} m/s "
                f"(pooled over {len(per_sigma)} stations; "
                f"per-station σ = "
                + ", ".join(f"{s:.4f}" for s in per_sigma)
                + ")"
            )
        print(
            f"Thresholds: var_T > {BK_K:g}·σ_turb², "
            f"|u'| > {MOR_TH:g}·σ_turb "
            f"(VITH={MOR_VITH:g})"
        )
    else:
        print(
            f"Local-σ thresholds: k={BK_K:g}, VITH={MOR_VITH:g}, TH={MOR_TH:g}"
        )
    print(f"VITA window T = {VITA_T_S * 1e3:.1f} ms")
    print(
        f"Time-series plots use [{T_START_S:g}, {T_END_S:g}] s; "
        "γ / frequency / duration comparisons use the full record."
    )

    results_zoom = [
        analyze_location(
            p,
            t_start_s=T_START_S,
            t_end_s=T_END_S,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            sigma_ref=sigma_ref,
        )
        for p in points
    ]
    results_full = [
        analyze_location(
            p,
            t_start_s=None,
            t_end_s=None,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            sigma_ref=sigma_ref,
        )
        for p in points
    ]

    print(
        "| y (mm) | geo | y_meas | σ_local | γ_V (%) | f_V (Hz) | "
        "τ̄_V (ms) | γ_L (%) | f_L (Hz) | τ̄_L (ms) | Jaccard |"
    )
    print(
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in results_full:
        s = r.stats
        print(
            f"| {r.y_wall_mm:.0f} | {r.point.geometry} | {r.point.y_mm:.0f} | "
            f"{r.u_std:.4f} | {100 * float(s['gamma_a']):.1f} | "
            f"{float(s['freq_a_hz']):.2f} | {1e3 * float(s['dur_mean_a_s']):.1f} | "
            f"{100 * float(s['gamma_b']):.1f} | {float(s['freq_b_hz']):.2f} | "
            f"{1e3 * float(s['dur_mean_b_s']):.1f} | {float(s['jaccard']):.3f} |"
        )

    print(
        "\nConditional raw $U$ on detector masks (full record). "
        "RMS is about the global record mean.\n"
    )
    print(
        "| y (mm) | y/Y | ⟨U⟩ | ⟨U⟩_B^V | ⟨U⟩_NB^V | u_rms,B^V | u_rms,NB^V | "
        "⟨U⟩_B^L | ⟨U⟩_NB^L | u_rms,B^L | u_rms,NB^L |"
    )
    print(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for r in results_full:
        s = r.stats
        print(
            f"| {r.y_wall_mm:.0f} | {float(y_over_Y(r.y_wall_mm)):.3f} | "
            f"{float(s['u_mean_all_b']):.3f} | "
            f"{float(s['u_mean_burst_a']):.3f} | {float(s['u_mean_nonburst_a']):.3f} | "
            f"{float(s['u_rms_burst_a']):.4f} | {float(s['u_rms_nonburst_a']):.4f} | "
            f"{float(s['u_mean_burst_b']):.3f} | {float(s['u_mean_nonburst_b']):.3f} | "
            f"{float(s['u_rms_burst_b']):.4f} | {float(s['u_rms_nonburst_b']):.4f} |"
        )

    if ALL_LOCATIONS:
        pdf_pages = write_multipage_comparisons(
            results_zoom,
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            max_plot_points=MAX_PLOT_POINTS,
            sigma_ref=sigma_ref,
        )
        print(f"Wrote multipage {pdf_pages.resolve()}")

        fig_stack, pdf_stack = plot_stacked_all_locations(
            results_zoom,
            max_plot_points=MAX_PLOT_POINTS,
            offset_sigma=STACK_OFFSET_SIGMA,
            sigma_ref=sigma_ref,
        )
        print(f"Wrote stacked {pdf_stack.resolve()}")

        fig_g, pdf_g, iface_bk, iface_mor = plot_gamma_profile(
            results_full, sigma_ref=sigma_ref, full_record=True
        )
        print(f"Wrote gamma profile {pdf_g.resolve()}")
        for label, iface in (("VITA", iface_bk), ("VITA+LEVEL", iface_mor)):
            y_pdf = iface["y_pdf_mean"]
            y_mode = iface["y_pdf_mode"]
            y_half = iface["y_gamma_half"]
            y_mid = iface["y_gamma_mid"]
            print(
                f"Mean interface from γ ({label}): "
                f"⟨y/Y⟩_γ = {y_pdf:.4f} "
                f"(y = {y_pdf * TUNNEL_HEIGHT_MM:.1f} mm), "
                f"mode |dγ/dy| at y/Y = {y_mode:.4f}, "
                f"γ∈[{100 * iface['gamma_min']:.1f}, "
                f"{100 * iface['gamma_max']:.1f}] %"
            )
            if np.isfinite(y_half):
                print(
                    f"  γ=0.5 crossing: y/Y = {y_half:.4f} "
                    f"(y = {y_half * TUNNEL_HEIGHT_MM:.1f} mm)"
                )
            else:
                print(
                    f"  no γ=0.5 crossing; mid-γ level "
                    f"{100 * iface['gamma_mid_level']:.1f}% at "
                    f"y/Y = {y_mid:.4f} "
                    f"(y = {y_mid * TUNNEL_HEIGHT_MM:.1f} mm)"
                )

        fig_f, pdf_f = plot_event_frequency_profile(
            results_full, sigma_ref=sigma_ref, full_record=True
        )
        print(f"Wrote frequency profile {pdf_f.resolve()}")

        fig_n, pdf_n = plot_event_count_profile(
            results_full, sigma_ref=sigma_ref, full_record=True
        )
        print(f"Wrote event-count profile {pdf_n.resolve()}")

        fig_cu, pdf_cu = plot_conditional_velocity_profiles(
            results_full, sigma_ref=sigma_ref, full_record=True
        )
        print(f"Wrote conditional velocity profiles {pdf_cu.resolve()}")

        fig_d, pdf_d = plot_mean_duration_profile(
            results_full, sigma_ref=sigma_ref, full_record=True
        )
        print(f"Wrote duration profile {pdf_d.resolve()}")

        fig_pdf, pdf_pdf = plot_duration_pdf(
            results_full, sigma_ref=sigma_ref, full_record=True
        )
        print(f"Wrote duration PDF {pdf_pdf.resolve()}")

        print("Comparing high-pass cutoffs on full records...")
        fig_hp, pdf_hp = plot_highpass_comparison(
            points,
            highpass_list=HIGHPASS_COMPARE_HZ,
            sigma_ref=sigma_ref,
        )
        print(f"Wrote high-pass comparison {pdf_hp.resolve()}")

        fig_hp_t, pdf_hp_t = plot_highpass_time_signals(
            y_targets_mm=(252.0, 502.0),
            highpass_list=HIGHPASS_COMPARE_HZ,
            t_start_s=T_START_S,
            t_end_s=T_END_S,
            max_plot_points=MAX_PLOT_POINTS,
        )
        print(f"Wrote high-pass time signals {pdf_hp_t.resolve()}")

        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig_stack)
            plt.close(fig_g)
            plt.close(fig_f)
            plt.close(fig_n)
            plt.close(fig_cu)
            plt.close(fig_d)
            plt.close(fig_pdf)
            plt.close(fig_hp)
            plt.close(fig_hp_t)
    else:
        r = results_zoom[0]
        fig, pdf = plot_method_comparison(
            r.t_s,
            r.u,
            mask_bk=r.mask_bk,
            mask_mor=r.mask_mor,
            y_wall_mm=r.y_wall_mm,
            fs_hz=1.0,
            vita_t_s=VITA_T_S,
            k=BK_K,
            vith=MOR_VITH,
            th=MOR_TH,
            highpass_hz=HIGHPASS_CUTOFF_HZ,
            max_plot_points=MAX_PLOT_POINTS,
        )
        print(f"Wrote {pdf.resolve()}")
        if SHOW_PLOTS:
            plt.show()
        else:
            plt.close(fig)
