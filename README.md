# Hot-wire data processing

Hot-wire calibration, velocity PDFs, longitudinal TKE spectrum (Welch / FFT), and Taylor-microscale analysis. Core logic lives in the **`mcflow_plotting`** package (including **`mcflow_plotting.hotwire`**); root `.py` files are short example drivers.

## Dev container (Codespaces vs local Docker)

There are two configurations under [`.devcontainer/`](.devcontainer/); both run [`.devcontainer/post-create.sh`](.devcontainer/post-create.sh) to install **`requirements-dev.txt`** and **`mcflow_plotting`** in editable mode (`--no-deps` so `freud-analysis` is not required for the hot-wire scripts).

| Configuration | File | Notes |
|----------------|------|--------|
| **Local Docker** (VS Code / Cursor) | [`.devcontainer/local/devcontainer.json`](.devcontainer/local/devcontainer.json) | No headless Matplotlib override—better if you use an interactive backend in the container. |
| **GitHub Codespaces** | [`.devcontainer/codespaces/devcontainer.json`](.devcontainer/codespaces/devcontainer.json) | Sets `MPLBACKEND=Agg` for a typical headless codespace. |

**Local:** Command Palette → **Dev Containers: Reopen in Container** (or **Rebuild Container**) → choose **Hot-wire (local Docker)**.

**Codespaces:** Use the badge or URL below so **`devcontainer_path`** selects the Codespaces config. If you use **Code → Create codespace** without that parameter, pick **Hot-wire (Codespaces)** under advanced options (or the configuration dropdown).

### GitHub Codespaces

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jtbreis/hotwire_data_processing?devcontainer_path=.devcontainer/codespaces/devcontainer.json)

Opens a new codespace on the default branch with the Codespaces dev container definition.

**Data:** Scripts use paths under `data/` in the workspace. That folder is in `.gitignore` by default—clone or copy HDF5 files into `data/` inside the codespace, or change paths in the scripts.

### Codespaces URL (copy-paste)

`https://codespaces.new/jtbreis/hotwire_data_processing?devcontainer_path=.devcontainer/codespaces/devcontainer.json`

Use **Code → Create codespace on main** from the GitHub repo page if you prefer the UI; then choose the **Hot-wire (Codespaces)** configuration if prompted.

## Local setup

```bash
git clone https://github.com/jtbreis/hotwire_data_processing.git
cd hotwire_data_processing
pip install -r requirements-dev.txt
pip install -e ./mcflow-plotting --no-deps
```

If you need Voronoi helpers under `mcflow_plotting.inertial_particles`, install the full dependency set (may need Conda for `freud-analysis`):

```bash
pip install -e ./mcflow-plotting
```

## Repository layout

| Path | Role |
|------|------|
| `.devcontainer/` | `local/devcontainer.json`, `codespaces/devcontainer.json`, `post-create.sh` |
| `mcflow-plotting/` | Installable `mcflow_plotting` package (style, plots, `hotwire` numerics) |
| `requirements-dev.txt` | Pinned-ish stack for scripts + editable install without `freud` |
| `hotwire_calibration.py`, `tke_spectrum.py`, … | Example drivers (paths and options at top) |
| `data/` | Local datasets (gitignored by default) |
