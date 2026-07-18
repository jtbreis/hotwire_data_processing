# %% [markdown]
# # Hot-wire longitudinal spectrum, dissipation, Kolmogorov scale
#
# Overlay calibrated hot-wire runs from one HDF5. Each run is Kolmogorov-normalized using
# **its own** η and u_η: ε is estimated from that run's Welch **E₁₁(k)**, then
# **kη = k η** and **E₁₁ / (u_η² η)**. Numerics: **``mcflow_plotting.hotwire``**;
# figures: **``mcflow_plotting``**.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcflow_plotting import (
    NU_AIR,
    compute_normalized_spectrum_runs_from_h5,
    fit_kolmogorov_reference_line_runs,
    load_all_calibrated_hotwire_velocities,
    print_spectrum_markdown_tables,
    spectrum_estimator_label,
)
from mcflow_plotting.plots.hotwire import plot_tke_spectrum_kolmogorov_normalized
from mcflow_plotting.style.figure import (
    analysis_plots_dir,
    save_figure,
    use_lab_matplotlib_style,
)

# %matplotlib inline  # uncomment when running in Jupyter

# %%
NU_AIR_LOCAL = NU_AIR  # alias if you want a different ν for this notebook

H5_PATH = Path(
    "/workspaces/hotwire_data_processing/data/hotwire/tnti_aligned_with_gravity/"
    "TNTI_aligned_with_g_400-500mm_pitot_calibrated_hotwire.h5"
)
# Spectrum uses this slice of each run (None = full record).
SPECTRUM_T_START_S: float | None = None
SPECTRUM_T_END_S: float | None = None

SPECTRUM_METHOD = os.environ.get(
    "TKE_SPECTRUM_METHOD", "welch").strip().lower()

REF_KETA_LO = 0.02
REF_KETA_HI = 0.2

# One distinct color per wall-position run (navy, red, green, purple, orange).
RUN_COLORS = (
    "#003f5c",
    "#e45756",
    "#54a24b",
    "#b279a2",
    "#f58518",
)

PLOTS_ROOT = Path(__file__).resolve().parent / "plots"
PLOTS_DIR = analysis_plots_dir(PLOTS_ROOT, H5_PATH, "tke-spectrum")

# Velocity snippet length in Kolmogorov timescales τ_η = η / u_η (scales from full-record Welch E11).
KOLMOGOROV_TIME_SEGMENTS = 4

# %%
runs = compute_normalized_spectrum_runs_from_h5(
    H5_PATH,
    spectrum_method=SPECTRUM_METHOD,
    nu=NU_AIR_LOCAL,
    t_start_s=SPECTRUM_T_START_S,
    t_end_s=SPECTRUM_T_END_S,
)
if not runs:
    raise ValueError(f"No runs found in {H5_PATH}.")

print("Per-run Kolmogorov scales (η, u_η from each run's Welch E11 → ε):")
for r in runs:
    tau_eta = float(r["eta"]) / float(r["u_eta"])
    print(
        f"  {r['legend']}: "
        f"η = {float(r['eta']):.4e} m, "
        f"u_η = {float(r['u_eta']):.4e} m/s, "
        f"τ_η = {tau_eta:.4e} s, "
        f"{KOLMOGOROV_TIME_SEGMENTS}τ_η = {KOLMOGOROV_TIME_SEGMENTS * tau_eta:.4e} s, "
        f"ε = {float(r['eps']):.4e} m²/s³"
    )

_spectrum_label = spectrum_estimator_label(str(runs[0]["spectrum_method"]))

k_eta_ref, e_ref, _c_ref = fit_kolmogorov_reference_line_runs(
    runs,
    k_lo=REF_KETA_LO,
    k_hi=REF_KETA_HI,
)

fig, ax = plot_tke_spectrum_kolmogorov_normalized(
    runs,
    k_eta_ref=k_eta_ref,
    e_ref=e_ref,
    colors=RUN_COLORS,
)

pdf_path = PLOTS_DIR / f"{H5_PATH.stem}_tke-spectrum_{SPECTRUM_METHOD}.pdf"
save_figure(fig, pdf_path)
print(f"Wrote {pdf_path.resolve()}")
plt.show()

# %%
use_lab_matplotlib_style()
velocity_by_run = {
    run_name: cv for run_name, cv in load_all_calibrated_hotwire_velocities(H5_PATH)
}
scale_by_run = {str(r["legend"]): r for r in runs}

fig_ts, axes = plt.subplots(
    len(runs),
    1,
    figsize=(7.5, 2.4 * len(runs)),
    sharex=True,
    squeeze=False,
)
for idx, (run_name, cv) in enumerate(velocity_by_run.items()):
    ax = axes[idx, 0]
    r = scale_by_run[run_name]
    eta = float(r["eta"])
    u_eta = float(r["u_eta"])
    tau_eta = eta / u_eta
    t_seg_s = KOLMOGOROV_TIME_SEGMENTS * tau_eta
    fs = float(cv.fs_hz)
    u = cv.u.astype(float)
    if SPECTRUM_T_START_S is not None or SPECTRUM_T_END_S is not None:
        s0 = 0 if SPECTRUM_T_START_S is None else max(
            0, int(np.floor(SPECTRUM_T_START_S * fs))
        )
        s1 = u.size if SPECTRUM_T_END_S is None else min(
            u.size, int(np.ceil(SPECTRUM_T_END_S * fs))
        )
        u = u[s0:s1]
    n_seg = min(u.size, max(2, int(np.ceil(t_seg_s * fs))))
    u_seg = u[:n_seg]
    u_mean = float(np.mean(u_seg))
    t_tau = (np.arange(n_seg, dtype=float) / fs) / tau_eta
    color = RUN_COLORS[idx % len(RUN_COLORS)]
    ax.plot(t_tau, u_seg - u_mean, color=color, lw=0.8)
    ax.set_ylabel(r"$u'$/m s$^{-1}$")
    ax.set_title(
        f"{run_name}\n"
        rf"$\eta={eta:.3e}$ m, $u_\eta={u_eta:.3e}$ m s$^{{-1}}$, "
        rf"$\tau_\eta={tau_eta:.3e}$ s",
        fontsize=10,
    )
    ax.grid(True, alpha=0.25)
axes[-1, 0].set_xlabel(
    rf"Time / $\tau_\eta$  (window length = {KOLMOGOROV_TIME_SEGMENTS}$\tau_\eta$)"
)
axes[-1, 0].set_xlim(0.0, KOLMOGOROV_TIME_SEGMENTS)
fig_ts.suptitle(
    rf"Hot-wire $u'(t)$ over {KOLMOGOROV_TIME_SEGMENTS} Kolmogorov timescales",
    fontsize=12,
    y=1.01,
)
fig_ts.tight_layout()

signal_pdf = (
    PLOTS_DIR
    / f"{H5_PATH.stem}_uPrime_{KOLMOGOROV_TIME_SEGMENTS}-tau-eta.pdf"
)
save_figure(fig_ts, signal_pdf)
print(f"Wrote {signal_pdf.resolve()}")
plt.show()

# %%
print_spectrum_markdown_tables(
    spectrum_method=str(runs[0]["spectrum_method"]),
    spectrum_description=_spectrum_label,
    spectrum_meta=dict(runs[0]["spectrum_meta"]),
    nu_air=NU_AIR_LOCAL,
    k_ref_lo=REF_KETA_LO,
    k_ref_hi=REF_KETA_HI,
    runs=runs,
)
