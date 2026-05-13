# %% [markdown]
# # Hot-wire longitudinal spectrum, dissipation, Kolmogorov scale
#
# Overlay one or more calibrated hot-wire records. Numerics: **``mcflow_plotting.hotwire``**;
# figures: **``mcflow_plotting``**.

# %%
from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from mcflow_plotting import (
    NU_AIR,
    compute_normalized_spectrum_run,
    fit_kolmogorov_reference_line,
    print_spectrum_markdown_tables,
    spectrum_estimator_label,
)
from mcflow_plotting.plots.hotwire import plot_tke_spectrum_kolmogorov_normalized

# %matplotlib inline  # uncomment when running in Jupyter

# %%
NU_AIR_LOCAL = NU_AIR  # alias if you want a different ν for this notebook

DEFAULT_H5 = Path(
    "/workspace/data/hotwire/tti_no_gravity/"
    "hotwire_10kHz_TTI_no_gravity_turbulent_side_pitot_calibrated_hotwire.h5"
)
DEFAULT_2_H5 = Path(
    "/workspace/data/hotwire/tti_no_gravity/"
    "hotwire_10kHz_TTI_no_gravity_non_turbulent_side_pitot_calibrated_hotwire.h5"
)

DATASETS: list[tuple[Path, str]] = [
    (DEFAULT_H5, "turbulent"),
    (DEFAULT_2_H5, "non_turbulent"),
]

SPECTRUM_METHOD = os.environ.get(
    "TKE_SPECTRUM_METHOD", "welch").strip().lower()

REF_KETA_LO = 0.02
REF_KETA_HI = 0.2

# %%
if not DATASETS:
    raise ValueError(
        "DATASETS must contain at least one (path, legend_label) entry.")

n_ds = len(DATASETS)
runs = [
    compute_normalized_spectrum_run(
        p,
        lab,
        spectrum_method=SPECTRUM_METHOD,
        nu=NU_AIR_LOCAL,
        n_datasets=n_ds,
    )
    for p, lab in DATASETS
]

_spectrum_label = spectrum_estimator_label(str(runs[0]["spectrum_method"]))

k_eta_ref, e_ref, _c_ref = fit_kolmogorov_reference_line(
    runs[0]["k_eta"],
    runs[0]["e11_norm"],
    k_lo=REF_KETA_LO,
    k_hi=REF_KETA_HI,
)

fig, ax = plot_tke_spectrum_kolmogorov_normalized(
    runs,
    k_eta_ref=k_eta_ref,
    e_ref=e_ref,
)
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
