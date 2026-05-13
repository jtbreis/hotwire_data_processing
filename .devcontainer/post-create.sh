#!/usr/bin/env bash
set -euo pipefail
pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e ./mcflow-plotting --no-deps
