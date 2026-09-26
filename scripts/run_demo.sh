#!/usr/bin/env bash
# Offline demo: synthetic EUR/USD data, no API keys, no network.
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate 2>/dev/null || true
python -m bot demo "$@"
