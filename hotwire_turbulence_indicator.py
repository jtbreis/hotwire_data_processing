# %% [markdown]
# # Hot-wire turbulence indicator \(I(t)\) and segmented \(E_{11}\) spectra
#
# **Threshold:** set **``THRESHOLD_DU2``** to the numerical **\(T_H\)** such that **\((\mathrm{d}u/\mathrm{d}t)^2 > T_H\)** marks candidate turbulent samples (units **m² s⁻⁴**). Tune from your data (e.g. Antonia often uses **\(T_H \approx 0.3\,\overline{(\mathrm{d}u/\mathrm{d}t)^2}\)** on the same window — you can set **``THRESHOLD_DU2``** to that product explicitly if you want the same discriminator value).
#
# **\(I(t)\) logic:** start from **\((\mathrm{d}u/\mathrm{d}t)^2 > T_H\)**. Optionally **bridge** internal below-threshold dips of length **≤ ``MERGE_GAP_SAMPLES``** so a burst is not split. Then set **\(I=1\)** only on contiguous **on** runs of length **≥ ``MIN_DURATION_SAMPLES``** (suppresses very short spurious highs).
#
# **Time panels:** bottom trace is **\(u - \bar{u}\)** with **\(\bar{u}\)** the **mean over the spectrum window** (same interval as **``u_spec``**).
#
# **Microscales:** **Taylor λ** and **Kolmogorov η, u_η** from split **\(I\)** masks (see code); spectrum **ε** uses class-wise Welch **\(E_{11}\)** when available.

# %%
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from mcflow_plotting import (
    NU_AIR,
    dissipation_epsilon,
    fit_kolmogorov_reference_line,
    kolmogorov_length,
    kolmogorov_velocity,
    load_calibrated_hotwire_velocity,
    taylor_microscale_from_epsilon,
)
from mcflow_plotting.plots.hotwire import plot_tke_spectrum_kolmogorov_normalized
from mcflow_plotting.style.figure import (
    analysis_plots_dir,
    save_figure,
    use_lab_matplotlib_style,
)


def _jupyter_matplotlib_inline() -> None:
    """Show figures inside the notebook / Interactive Window (no-op in plain ``python``)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")
    except (ImportError, AttributeError):
        pass


_jupyter_matplotlib_inline()

# %%
H5_PATH = Path(
    "/workspaces/hotwire_data_processing/data/hotwire/tti_no_gravity/hotwire_10kHz_TTI_no_gravity_turbulent_side_pitot_calibrated_hotwire.h5"
)
# Numerical threshold T_H on (du/dt)^2 (m^2 s^-4). Tune for your record.
THRESHOLD_DU2 = 30.0
MIN_DURATION_SAMPLES = 10
# Bridge internal below-threshold gaps in (du/dt)^2 of at most this many samples (Antonia-style
# spurious brief drops). None = use ``MIN_DURATION_SAMPLES``; 0 = disable merging.
MERGE_GAP_SAMPLES: int | None = None
# Time-domain zoom (seconds); clamped inside the spectrum window below.
T_START_S: float | None = None
T_END_S: float | None = 0.3
# Indicator + segmented spectra use this slice of the file (None = start / end of record).
SPECTRUM_T_START_S: float | None = None
SPECTRUM_T_END_S: float | None = None
# Welch length cap (optional). None → per-class default from that class's longest contiguous run.
SPECTRUM_NPERSEG: int | None = None
NU_AIR_LOCAL = NU_AIR

MAX_PLOT_POINTS = 25_000
FIGSIZE = (7.5, 6.2)
# Histogram of (du/dt)^2 on the spectrum window (same signal as intermittency threshold).
DU2_PDF_BINS = 120
DU2_PDF_MAX_SAMPLES: int | None = 300_000
DU2_PDF_FIGSIZE = (6.5, 4.25)
HEIGHT_RATIOS = (1.0, 2.0, 3.0)  # I(t), (du/dt)^2, u - mean(u)
# PDFs are always written under this directory (created if missing).
PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
# After saving PDFs, also show figures (e.g. in Jupyter). Set False for headless runs.
SHOW_PLOTS = True

REF_KETA_LO = 0.02
REF_KETA_HI = 0.2
# Minimum masked samples to report pooled Taylor / time-derivative ε.
MIN_MASK_SAMPLES_FOR_SCALES = 32

# %%


def _safe_plot_stem(name: str, *, max_len: int = 72) -> str:
    """Filesystem-friendly slug from HDF5 stem or similar."""
    out: list[str] = []
    for c in name.lower():
        if c.isalnum() or c in "-_":
            out.append(c)
        else:
            out.append("-")
    s = "".join(out).strip("-")
    while "--" in s:
        s = s.replace("--", "-")
    return s[:max_len] if s else "hotwire-run"


def intermittency_plot_pdf_paths(
    plots_dir: Path,
    h5: Path,
    *,
    thr_du2: float,
    min_duration: int,
    merge_gap_eff: int,
    t_zoom0: float,
    t_zoom1: float,
    t_spec0: float,
    t_spec1: float,
    fs_hz: float,
) -> tuple[Path, Path, Path]:
    """Return (timeseries_pdf, segmented_E11_pdf, du2_PDF_histogram_pdf)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_plot_stem(h5.stem)
    thr_s = f"{float(thr_du2):.4g}".replace(".", "p")
    z0 = f"{float(t_zoom0):.4g}".replace(".", "p")
    z1 = f"{float(t_zoom1):.4g}".replace(".", "p")
    s0 = f"{float(t_spec0):.4g}".replace(".", "p")
    s1 = f"{float(t_spec1):.4g}".replace(".", "p")
    fs_i = int(round(float(fs_hz)))
    base = (
        f"{stem}_intermittency_TH{thr_s}_Nmin{int(min_duration)}_"
        f"gap{int(merge_gap_eff)}_zoom{z0}-{z1}s_specWin{s0}-{s1}s_fs{fs_i}Hz"
    )
    return (
        plots_dir / f"{base}_timeseries-I-du2-uPrime.pdf",
        plots_dir / f"{base}_E11-segmented-Kolmogorov.pdf",
        plots_dir / f"{base}_du2dt2-PDF-histogram.pdf",
    )


def du_dt(u: np.ndarray, fs_hz: float) -> np.ndarray:
    """Time derivative ``du/dt`` (same length as ``u``); central differences."""
    dt = 1.0 / float(fs_hz)
    return np.gradient(np.asarray(u, dtype=float), dt, edge_order=2)


def du_dt_squared(u: np.ndarray, fs_hz: float) -> np.ndarray:
    """Central differences via ``np.gradient``; same length as ``u``."""
    d = du_dt(u, fs_hz)
    return d * d


def indicator_from_runs(
    high_derivative: np.ndarray, *, min_consecutive: int
) -> np.ndarray:
    r"""
    Return float array 0/1: **\(I=1\)** on each maximal contiguous run where ``high_derivative``
    is True **and** the run length is **≥ ``min_consecutive``** samples. Shorter True runs stay
    **\(I=0\)**. (Apply **``fill_short_false_gaps_between_true``** to ``high_derivative`` first if
    brief dips in **\((\mathrm{d}u/\mathrm{d}t)^2\)** should not split a region.)
    """
    mask = high_derivative.astype(bool)
    n = int(mask.size)
    I = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        if j - i >= min_consecutive:
            I[i:j] = 1.0
        i = j
    return I


def fill_short_false_gaps_between_true(
    mask: np.ndarray, max_gap_len: int
) -> np.ndarray:
    """
    Set internal **False** runs to **True** when the run is sandwiched between **True** and has
    length **≤ ``max_gap_len``** (repeated until stable), so brief dips do not fragment an
    otherwise above-threshold region.
    """
    if max_gap_len <= 0:
        return np.asarray(mask, dtype=bool)
    m = np.asarray(mask, dtype=bool).copy()
    n = int(m.size)
    if n == 0:
        return m
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


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return ``[(i0, i1), ...]`` half-open intervals where ``mask`` is True."""
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


def average_welch_e11_over_segments(
    u: np.ndarray,
    fs_hz: float,
    segments: list[tuple[int, int]],
    u_mean: float,
    *,
    nperseg: int,
    noverlap: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Mean Welch one-sided PSD over segments; map to ``k1``, ``E11`` with Taylor hypothesis
    using fixed ``u_mean`` (same as ``velocity_series_to_e11_spectrum``).
    """
    f_acc: np.ndarray | None = None
    psd_acc: np.ndarray | None = None
    used = 0
    for i0, i1 in segments:
        seg = np.asarray(u[i0:i1], dtype=float)
        if seg.size < nperseg:
            continue
        uf = seg - float(np.mean(seg))
        f_hz, p_uu = signal.welch(
            uf,
            fs=fs_hz,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            average="mean",
        )
        if f_acc is None:
            f_acc = f_hz
            psd_acc = np.asarray(p_uu, dtype=float).copy()
        else:
            if f_hz.shape != f_acc.shape or not np.allclose(f_hz, f_acc):
                raise RuntimeError(
                    "Welch frequency grid mismatch between segments.")
            psd_acc = psd_acc + p_uu
        used += 1
    if used == 0 or f_acc is None or psd_acc is None:
        return np.array([]), np.array([]), 0
    psd_acc /= float(used)
    k = (2.0 * np.pi * f_acc) / u_mean
    e11 = psd_acc * u_mean / (2.0 * np.pi)
    ok = k > 0.0
    return k[ok], e11[ok], used


def kolmogorov_normalized_run(
    k: np.ndarray,
    e11: np.ndarray,
    *,
    legend: str,
    nu: float,
) -> dict[str, object]:
    eps = dissipation_epsilon(k, e11, nu=nu)
    eta = kolmogorov_length(eps, nu=nu)
    u_eta = kolmogorov_velocity(eps, nu=nu)
    k_eta = k * eta
    e11_norm = e11 / (u_eta**2 * eta)
    return {
        "legend": legend,
        "k_eta": k_eta,
        "e11_norm": e11_norm,
        "eps": eps,
        "eta": eta,
        "u_eta": u_eta,
    }


def microscales_masked_class(
    u: np.ndarray,
    fs_hz: float,
    mask: np.ndarray,
    u_mean: float,
    *,
    k: np.ndarray,
    e11: np.ndarray,
    nu: float,
    min_samples: int,
) -> dict[str, float] | None:
    """
    Pooled statistics on samples where ``mask`` is True: ``u' = u - u_mean`` (global ``Ū`` on
    the spectrum window), ``du'/dt`` from ``np.gradient`` on the full ``u'`` series, then
    means of ``u'^2`` and ``(du'/dt)^2`` restricted to ``mask``. Gives **λ** from time
    derivatives and **ε, η, u_η** from time derivatives and (if ``k,e11`` non-empty) from
    the segmented Welch **E11**.
    """
    u = np.asarray(u, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if int(np.count_nonzero(mask)) < min_samples:
        return None
    u_fluc = u - float(u_mean)
    dt = 1.0 / float(fs_hz)
    du_dt = np.gradient(u_fluc, dt, edge_order=2)
    ms_u = float(np.mean((u_fluc[mask]) ** 2))
    ms_d = float(np.mean((du_dt[mask]) ** 2))
    if ms_u <= 0.0 or ms_d <= 0.0:
        return None
    u_rms = float(np.sqrt(ms_u))
    lambda_time = float(u_mean * u_rms / np.sqrt(ms_d))
    eps_time = float(15.0 * nu * ms_d / (u_mean**2))
    eta_time = kolmogorov_length(eps_time, nu=nu)
    u_eta_time = kolmogorov_velocity(eps_time, nu=nu)
    lambda_eps_time = taylor_microscale_from_epsilon(u_rms, eps_time, nu=nu)

    out: dict[str, float] = {
        "n_mask": float(np.count_nonzero(mask)),
        "u_rms": u_rms,
        "mean_du_dt_sq": ms_d,
        "lambda_time": lambda_time,
        "epsilon_time": eps_time,
        "eta_time": eta_time,
        "u_eta_time": u_eta_time,
        "lambda_eps_time": lambda_eps_time,
    }
    if k.size and e11.size:
        eps_spec = float(dissipation_epsilon(k, e11, nu=nu))
        out["epsilon_spec"] = eps_spec
        out["eta_spec"] = float(kolmogorov_length(eps_spec, nu=nu))
        out["u_eta_spec"] = float(kolmogorov_velocity(eps_spec, nu=nu))
        out["lambda_eps_spec"] = float(
            taylor_microscale_from_epsilon(u_rms, eps_spec, nu=nu)
        )
    return out


def _md_cell(s: str) -> str:
    """Escape pipe characters so table cells stay valid."""
    return str(s).replace("|", r"\|")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """GitHub-style Markdown table (no outer blank lines)."""
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    head = "| " + " | ".join(_md_cell(h) for h in headers) + " |"
    body = ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def _fmt_opt(x: float | None, fmt: str = ".6e", na: str = "—") -> str:
    if x is None:
        return na
    return format(float(x), fmt)


def run_parameters_markdown(
    *,
    h5_name: str,
    thr_du2: float,
    min_duration: int,
    merge_gap_effective: int,
    merge_gap_setting_raw: str,
    t_spec_lo: float,
    t_spec_hi: float,
    u_mean: float,
    fs_hz: float,
    n_spec: int,
) -> str:
    rows = [
        ["HDF5 file", h5_name],
        ["T_H on (du/dt)^2", f"{thr_du2:.6g} m^2 s^-4"],
        ["MIN_DURATION_SAMPLES", str(min_duration)],
        ["MERGE_GAP_SAMPLES (script)", merge_gap_setting_raw],
        ["Internal below-threshold gaps bridged if length ≤ (samples)", str(
            merge_gap_effective)],
        ["Spectrum window (s)", f"{t_spec_lo:.6g} – {t_spec_hi:.6g}"],
        ["mean(u) on spectrum window", f"{u_mean:.6f} m s^-1"],
        ["Sample rate", f"{fs_hz:.0f} Hz"],
        ["N on spectrum window", str(n_spec)],
    ]
    return markdown_table(["Parameter", "Value"], rows)


def microscales_comparison_markdown(
    ms_t: dict[str, float] | None,
    ms_c: dict[str, float] | None,
    *,
    nu: float,
) -> str:
    """Side-by-side microscale estimates for masked I=1 vs I=0 (missing dict → em dash)."""
    rows_out: list[list[str]] = []
    keys: list[tuple[str, str, str]] = [
        ("Masked sample count N", "n_mask", ".0f"),
        ("u_rms masked (m/s)", "u_rms", ".6e"),
        ("mean (du'/dt)^2 on mask", "mean_du_dt_sq", ".6e"),
        ("lambda Taylor time (m)", "lambda_time", ".6e"),
        ("epsilon time Taylor (m^2/s^3)", "epsilon_time", ".6e"),
        ("eta from epsilon_time (m)", "eta_time", ".6e"),
        ("u_eta from epsilon_time (m/s)", "u_eta_time", ".6e"),
        ("lambda from epsilon_time isotropic (m)", "lambda_eps_time", ".6e"),
    ]
    spec_keys: list[tuple[str, str, str]] = [
        ("epsilon Welch E11 (m^2/s^3)", "epsilon_spec", ".6e"),
        ("eta from epsilon_spec (m)", "eta_spec", ".6e"),
        ("u_eta from epsilon_spec (m/s)", "u_eta_spec", ".6e"),
        ("lambda from epsilon_spec isotropic (m)", "lambda_eps_spec", ".6e"),
    ]

    def _v(m: dict[str, float] | None, key: str) -> float | None:
        if m is None or key not in m:
            return None
        return float(m[key])

    for label, key, fmt in keys:
        rows_out.append(
            [
                label,
                _fmt_opt(_v(ms_t, key), fmt=fmt),
                _fmt_opt(_v(ms_c, key), fmt=fmt),
            ]
        )
    for label, key, fmt in spec_keys:
        if _v(ms_t, key) is None and _v(ms_c, key) is None:
            continue
        rows_out.append(
            [
                label,
                _fmt_opt(_v(ms_t, key), fmt=fmt),
                _fmt_opt(_v(ms_c, key), fmt=fmt),
            ]
        )
    rows_out.append(["nu air (m^2/s)", f"{nu:.4e}", f"{nu:.4e}"])
    return markdown_table(["Quantity", "I(t)=1", "I(t)=0"], rows_out)


def microscales_slide_table(
    ms_t: dict[str, float] | None,
    ms_c: dict[str, float] | None,
    *,
    fmt: str = ".3e",
) -> tuple[list[str], list[list[str]]]:
    """
    Header + body rows (strings) for slide microscales: Taylor λ (time), Kolmogorov η and u_η,
    isotropic λ. For η, u_η, and isotropic λ, Welch **E11** ε is used when those keys exist on
    the class dict; otherwise the time-derivative ε path.
    """
    def _v(m: dict[str, float] | None, key: str) -> float | None:
        if m is None or key not in m:
            return None
        return float(m[key])

    def _pick(m: dict[str, float] | None, spec_k: str, time_k: str) -> float | None:
        v = _v(m, spec_k)
        return v if v is not None else _v(m, time_k)

    header = ["Microscale", "I(t)=1", "I(t)=0"]
    body = [
        [
            "Taylor λ from time (m)",
            _fmt_opt(_v(ms_t, "lambda_time"), fmt=fmt),
            _fmt_opt(_v(ms_c, "lambda_time"), fmt=fmt),
        ],
        [
            "Kolmogorov η (m)",
            _fmt_opt(_pick(ms_t, "eta_spec", "eta_time"), fmt=fmt),
            _fmt_opt(_pick(ms_c, "eta_spec", "eta_time"), fmt=fmt),
        ],
        [
            "Kolmogorov u_η (m/s)",
            _fmt_opt(_pick(ms_t, "u_eta_spec", "u_eta_time"), fmt=fmt),
            _fmt_opt(_pick(ms_c, "u_eta_spec", "u_eta_time"), fmt=fmt),
        ],
        [
            "Taylor λ isotropic from ε (m)",
            _fmt_opt(_pick(ms_t, "lambda_eps_spec",
                     "lambda_eps_time"), fmt=fmt),
            _fmt_opt(_pick(ms_c, "lambda_eps_spec",
                     "lambda_eps_time"), fmt=fmt),
        ],
    ]
    return header, body


def microscales_slide_typst_snippet(
    ms_t: dict[str, float] | None,
    ms_c: dict[str, float] | None,
    *,
    fmt: str = ".3e",
) -> str:
    """
    Self-contained Typst ``#table`` (no external file): copy into a ``.typ`` slide.
    """
    def _esc_typst_markup(s: str) -> str:
        return s.replace("\\", "\\\\").replace("]", "\\]")

    def _cell(s: str, *, header: bool = False) -> str:
        t = _esc_typst_markup(s)
        return f"[*{t}*]" if header else f"[{t}]"

    header, body = microscales_slide_table(ms_t, ms_c, fmt=fmt)
    head_line = ",\n    ".join(_cell(h, header=True) for h in header)
    body_lines: list[str] = []
    for row in body:
        body_lines.append("  " + ", ".join(_cell(c) for c in row) + ",")
    body_block = "\n".join(body_lines)
    return "\n".join(
        [
            "// Microscales (slide) — generated by hotwire_turbulence_indicator.py",
            "// η, u_η, isotropic λ: Welch E11 ε when present; else time-derivative ε.",
            "#table(",
            "  columns: 3,",
            "  inset: 6pt,",
            "  align: horizon,",
            "  table.header(",
            f"    {head_line},",
            "  ),",
            body_block,
            ")",
        ]
    )


def welch_segment_summary_markdown(
    *,
    n_t: int,
    nperseg_t: int,
    n_seg_turb: int,
    n_c: int,
    nperseg_c: int,
    n_seg_calm: int,
) -> str:
    rows = [
        ["I=1", str(n_t), str(nperseg_t), str(n_seg_turb)],
        ["I=0", str(n_c), str(nperseg_c), str(n_seg_calm)],
    ]
    return markdown_table(
        ["Class", "Welch segment count", "nperseg", "Contiguous runs (class)"],
        rows,
    )


def spectrum_runs_markdown(runs: list[dict[str, object]]) -> str:
    rows_sp: list[list[str]] = []
    for r in runs:
        rows_sp.append(
            [
                _md_cell(str(r.get("legend", ""))),
                _fmt_opt(float(r["eps"]), ".6e"),  # type: ignore[arg-type]
                _fmt_opt(float(r["eta"]), ".6e"),  # type: ignore[arg-type]
                _fmt_opt(float(r["u_eta"]), ".6e"),  # type: ignore[arg-type]
            ]
        )
    return markdown_table(
        ["Class (Welch E11)", "epsilon (m^2 s^-3)",
         "eta (m)", "u_eta (m s^-1)"],
        rows_sp,
    )


def resolve_nperseg_for_class(
    segments: list[tuple[int, int]],
    *,
    user_nperseg: int | None,
) -> int:
    """Welch ``nperseg`` for one class, never longer than the longest run of that class."""
    lens = [b - a for a, b in segments]
    max_len = max(lens) if lens else 0
    if max_len <= 0:
        return 0
    if user_nperseg is not None:
        n = min(int(user_nperseg), max_len)
    else:
        n = min(65_536, max(256, max_len // 2))
        n = min(n, max_len)
    return max(1, int(n))


# %%
cv = load_calibrated_hotwire_velocity(H5_PATH)
u_all = cv.u.astype(float)
fs = float(cv.fs_hz)
n = u_all.size
t_all = np.arange(n, dtype=float) / fs

s0 = 0 if SPECTRUM_T_START_S is None else max(
    0, int(np.floor(SPECTRUM_T_START_S * fs)))
s1 = n if SPECTRUM_T_END_S is None else min(
    n, int(np.ceil(SPECTRUM_T_END_S * fs)))
if s1 <= s0:
    raise ValueError(f"Empty spectrum window: indices {s0}..{s1}.")

u_spec = u_all[s0:s1]
t_spec = t_all[s0:s1]
u_mean_spec = float(np.mean(u_spec))
du2_spec = du_dt_squared(u_spec, fs)
thr = float(THRESHOLD_DU2)
if not np.isfinite(thr) or thr <= 0.0:
    raise ValueError(
        "THRESHOLD_DU2 must be a finite positive value in m^2 s^-4 (units of (du/dt)^2)."
    )
thr_line_label = r"$T_H$"
_merge_gap = (
    MIN_DURATION_SAMPLES
    if MERGE_GAP_SAMPLES is None
    else int(MERGE_GAP_SAMPLES)
)
mask_above_thr = du2_spec > thr
mask_for_indicator = (
    fill_short_false_gaps_between_true(mask_above_thr, _merge_gap)
    if _merge_gap > 0
    else mask_above_thr
)
I_spec = indicator_from_runs(
    mask_for_indicator, min_consecutive=MIN_DURATION_SAMPLES)

t_spec_lo = float(t_spec[0])
t_spec_hi = float(t_spec[-1])
view_t0 = t_spec_lo if T_START_S is None else float(T_START_S)
view_t1 = t_spec_hi if T_END_S is None else float(T_END_S)
view_t0 = max(view_t0, t_spec_lo)
view_t1 = min(view_t1, t_spec_hi)
if view_t1 <= view_t0:
    raise ValueError(
        "Time zoom is empty after clamping to the spectrum window.")

i_rel0 = max(0, int(np.floor((view_t0 - t_spec_lo) * fs)))
i_rel1 = min(u_spec.size, int(np.ceil((view_t1 - t_spec_lo) * fs)))
u_w = u_spec[i_rel0:i_rel1]
t_w = t_spec[i_rel0:i_rel1]
I_w = I_spec[i_rel0:i_rel1]
du2_w = du2_spec[i_rel0:i_rel1]
u_prime = u_w - u_mean_spec

pdf_timeseries, pdf_spectrum_split, pdf_du2 = intermittency_plot_pdf_paths(
    analysis_plots_dir(PLOTS_ROOT, H5_PATH, "intermittency"),
    H5_PATH,
    thr_du2=thr,
    min_duration=MIN_DURATION_SAMPLES,
    merge_gap_eff=_merge_gap,
    t_zoom0=view_t0,
    t_zoom1=view_t1,
    t_spec0=float(t_spec[0]),
    t_spec1=float(t_spec[-1]),
    fs_hz=fs,
)

step = max(1, len(t_w) // MAX_PLOT_POINTS)
t_p = t_w[::step]
I_p = I_w[::step]
du2_p = du2_w[::step]
up_p = u_prime[::step]

use_lab_matplotlib_style()
fig, (ax_i, ax_du2, ax_u) = plt.subplots(
    3,
    1,
    sharex=True,
    figsize=FIGSIZE,
    layout="constrained",
    gridspec_kw={"height_ratios": list(HEIGHT_RATIOS), "hspace": 0.08},
)
ax_i.step(t_p, I_p, where="post", color="k", lw=0.9)
ax_i.set_ylabel(r"$I(t)$")
ax_i.set_yticks([0.0, 1.0])
ax_i.set_ylim(-0.05, 1.05)
ax_i.set_xlim(float(t_p[0]), float(t_p[-1]))
ax_i.grid(True, alpha=0.25)

ax_du2.plot(t_p, du2_p, color="k", lw=0.7)
ax_du2.axhline(thr, color="0.45", ls="--", lw=1.0, label=thr_line_label)
ax_du2.set_ylabel(r"$(\mathrm{d}u/\mathrm{d}t)^2$ (m$^2$ s$^{-4}$)")
ax_du2.grid(True, alpha=0.25)
ax_du2.legend(loc="upper right", fontsize=8)

ax_u.plot(t_p, up_p, color="k", lw=0.7)
ax_u.set_ylabel(r"$u - \bar{u}$ (m s$^{-1}$)")
ax_u.set_xlabel("Time (s)")
ax_u.grid(True, alpha=0.25)

save_figure(fig, pdf_timeseries)
print(f"Wrote {pdf_timeseries.resolve()}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig)

if DU2_PDF_MAX_SAMPLES is not None and du2_spec.size > DU2_PDF_MAX_SAMPLES:
    _rng = np.random.default_rng(0)
    _idx = _rng.choice(du2_spec.size, size=DU2_PDF_MAX_SAMPLES, replace=False)
    du2_pdf_sample = du2_spec[_idx]
else:
    du2_pdf_sample = du2_spec

use_lab_matplotlib_style()
fig_du2, ax_du2 = plt.subplots(figsize=DU2_PDF_FIGSIZE)
ax_du2.hist(
    du2_pdf_sample,
    bins=DU2_PDF_BINS,
    density=True,
    color="k",
    alpha=0.35,
    edgecolor="0.25",
    linewidth=0.4,
)
ax_du2.set_xlabel(r"$(\mathrm{d}u/\mathrm{d}t)^2$ (m$^2$ s$^{-4}$)")
ax_du2.set_ylabel(r"PDF of $(\mathrm{d}u/\mathrm{d}t)^2$")
ax_du2.axvline(
    thr,
    color="0.45",
    ls="--",
    lw=1.0,
    label=r"$T_H$",
)
ax_du2.legend(loc="upper right", fontsize=8)
ax_du2.grid(True, alpha=0.25)
ax_du2.text(
    0.02,
    0.98,
    rf"$N={du2_spec.size}$, hist $N={du2_pdf_sample.size}$, $f_s={fs:.0f}$ Hz",
    transform=ax_du2.transAxes,
    va="top",
    fontsize=8,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
)
save_figure(fig_du2, pdf_du2)
print(f"Wrote {pdf_du2.resolve()}")
if SHOW_PLOTS:
    plt.show()
else:
    plt.close(fig_du2)

# --- Segmented TKE / E11 spectra (turbulent vs non-turbulent intervals) ---
runs_turb = contiguous_runs(I_spec > 0.5)
runs_calm = contiguous_runs(I_spec < 0.5)
nperseg_t = resolve_nperseg_for_class(runs_turb, user_nperseg=SPECTRUM_NPERSEG)
nperseg_c = resolve_nperseg_for_class(runs_calm, user_nperseg=SPECTRUM_NPERSEG)
if nperseg_t > 0:
    nperseg_t = min(nperseg_t, u_spec.size)
if nperseg_c > 0:
    nperseg_c = min(nperseg_c, u_spec.size)
noverlap_t = nperseg_t // 2 if nperseg_t > 0 else 0
noverlap_c = nperseg_c // 2 if nperseg_c > 0 else 0

k_t, e_t, n_t = (
    average_welch_e11_over_segments(
        u_spec,
        fs,
        runs_turb,
        u_mean_spec,
        nperseg=nperseg_t,
        noverlap=noverlap_t,
    )
    if nperseg_t > 0
    else (np.array([]), np.array([]), 0)
)
k_c, e_c, n_c = (
    average_welch_e11_over_segments(
        u_spec,
        fs,
        runs_calm,
        u_mean_spec,
        nperseg=nperseg_c,
        noverlap=noverlap_c,
    )
    if nperseg_c > 0
    else (np.array([]), np.array([]), 0)
)

mask_t = I_spec > 0.5
mask_c = I_spec < 0.5
ms_t = microscales_masked_class(
    u_spec,
    fs,
    mask_t,
    u_mean_spec,
    k=k_t,
    e11=e_t,
    nu=NU_AIR_LOCAL,
    min_samples=MIN_MASK_SAMPLES_FOR_SCALES,
)
ms_c = microscales_masked_class(
    u_spec,
    fs,
    mask_c,
    u_mean_spec,
    k=k_c,
    e11=e_c,
    nu=NU_AIR_LOCAL,
    min_samples=MIN_MASK_SAMPLES_FOR_SCALES,
)

spectrum_runs: list[dict[str, object]] = []
if n_t > 0 and k_t.size:
    spectrum_runs.append(
        kolmogorov_normalized_run(
            k_t,
            e_t,
            legend=(
                rf"$I(t)=1$: mean Welch over {n_t} segment(s), "
                rf"$n_{{\mathrm{{perseg}}}}={nperseg_t}$"
            ),
            nu=NU_AIR_LOCAL,
        )
    )
if n_c > 0 and k_c.size:
    spectrum_runs.append(
        kolmogorov_normalized_run(
            k_c,
            e_c,
            legend=(
                rf"$I(t)=0$: mean Welch over {n_c} segment(s), "
                rf"$n_{{\mathrm{{perseg}}}}={nperseg_c}$"
            ),
            nu=NU_AIR_LOCAL,
        )
    )

if spectrum_runs:
    r0 = spectrum_runs[0]
    k_eta_ref, e_ref, _c = fit_kolmogorov_reference_line(
        np.asarray(r0["k_eta"], dtype=float),
        np.asarray(r0["e11_norm"], dtype=float),
        k_lo=REF_KETA_LO,
        k_hi=REF_KETA_HI,
    )
    fig_sp, ax_sp = plot_tke_spectrum_kolmogorov_normalized(
        spectrum_runs,  # type: ignore[arg-type]
        k_eta_ref=k_eta_ref,
        e_ref=e_ref,
    )
    save_figure(fig_sp, pdf_spectrum_split)
    print(f"Wrote {pdf_spectrum_split.resolve()}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig_sp)

_merge_gap_setting = (
    "None (defaults to MIN_DURATION_SAMPLES)"
    if MERGE_GAP_SAMPLES is None
    else str(MERGE_GAP_SAMPLES)
)
print()
print("### Run parameters")
print()
print(
    run_parameters_markdown(
        h5_name=H5_PATH.name,
        thr_du2=thr,
        min_duration=MIN_DURATION_SAMPLES,
        merge_gap_effective=_merge_gap,
        merge_gap_setting_raw=_merge_gap_setting,
        t_spec_lo=float(t_spec[0]),
        t_spec_hi=float(t_spec[-1]),
        u_mean=u_mean_spec,
        fs_hz=fs,
        n_spec=int(u_spec.size),
    )
)
print()
print("### Welch segments (segmented E11)")
print()
print(
    welch_segment_summary_markdown(
        n_t=n_t,
        nperseg_t=nperseg_t,
        n_seg_turb=len(runs_turb),
        n_c=n_c,
        nperseg_c=nperseg_c,
        n_seg_calm=len(runs_calm),
    )
)
print()
print("### Microscales (masked by I(t); time + Welch E11)")
print()
print(microscales_comparison_markdown(ms_t, ms_c, nu=NU_AIR_LOCAL))
print()
print("### Microscales — Typst (slides, copy-paste)")
print()
print("```typ")
print(microscales_slide_typst_snippet(ms_t, ms_c))
print("```")
print()
if spectrum_runs:
    print("### Spectrum-class dissipation (Welch E11)")
    print()
    print(spectrum_runs_markdown(spectrum_runs))
else:
    print("### Spectrum-class dissipation (Welch E11)")
    print()
    print(
        markdown_table(
            ["Note", "Value"],
            [
                ["Kolmogorov spectrum figure",
                    "skipped (no usable Welch average)"],
                ["I=1 diagnostics",
                    f"nperseg={nperseg_t}, runs={len(runs_turb)}"],
                ["I=0 diagnostics",
                    f"nperseg={nperseg_c}, runs={len(runs_calm)}"],
            ],
        )
    )
print()
