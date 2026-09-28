#!/bin/bash
# Verify the ethical principle properties in principles.vcl with Vehicle + Marabou.
#
#   ./verify.sh [network.onnx] [tolerance]
#
# Defaults: models/base.onnx and tolerance 0.05. The controller constants come from data/meta.json.

set -e
cd "$(dirname "$0")"

# Use vehicle and Marabou from this folder's .venv, whether or not it is activated
if [ -d .venv/bin ]; then
  PATH="$(cd .venv/bin && pwd):$PATH"
fi

NETWORK=${1:-models/base.onnx}
TOLERANCE=${2:-0.05}

vehicle verify \
  -v Marabou \
  -s principles.vcl \
  -n effort:$NETWORK \
  -p tau:5.0 \
  -p kEffort:0.05 \
  -p rMax:1.0 \
  -p tolerance:$TOLERANCE
