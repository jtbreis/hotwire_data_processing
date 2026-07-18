#!/usr/bin/env bash
set -euo pipefail
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e ./mcflow-plotting --no-deps

expected="/workspace/data/hotwire/tti_no_gravity"
if [[ ! -d "${expected}" ]]; then
  echo ""
  echo "WARNING: ${expected} is missing inside the container."
  echo "  The bind mount to /workspace/data probably failed or points at the wrong host folder."
  echo "  Or edit the 'source=' path in .devcontainer/devcontainer.json (or local/devcontainer.json)."
  echo "    Default host path: \$HOME/phd-research/data"
  echo "  If Docker cannot find the host path, it often creates an empty directory instead of failing."
  echo ""
fi
