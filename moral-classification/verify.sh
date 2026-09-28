#!/bin/bash
# Verify the classifier on one hyperrectangle with Vehicle + Marabou (spec.vcl).
#
#   ./verify.sh [box index] [network.onnx] [cluster]
#
# Defaults: box 0, models/western_base.onnx and the western cluster's choices. The box and
# the cluster's choice for its dilemma (0 stay, 1 swerve) are read from
# embeddings/hyperrectangles.npz. To verify many boxes at once, use src/verify_boxes.py.

set -e
cd "$(dirname "$0")"

# Use vehicle and Marabou from this folder's .venv, whether or not it is activated
if [ -d .venv/bin ]; then
  PATH="$(cd .venv/bin && pwd):$PATH"
fi

BOX=${1:-0}
NETWORK=${2:-models/western_base.onnx}
CLUSTER=${3:-western}

# Write the box to a temporary IDX file, the format Vehicle reads datasets from
BOX_FILE=$(mktemp --suffix=.idx)
trap 'rm -f "$BOX_FILE"' EXIT
LABEL=$(python src/verify_boxes.py --export-box "$BOX" "$BOX_FILE" --cluster "$CLUSTER")
echo "Box $BOX, $CLUSTER choice $LABEL (0 stay, 1 swerve)"
if [ "$LABEL" -lt 0 ]; then
  echo "The $CLUSTER cluster's vote on this dilemma was tied, so the box has no label."
  exit 0
fi

vehicle verify \
  -v Marabou \
  -s spec.vcl \
  -n classifier:$NETWORK \
  -d hyperrectangle:$BOX_FILE \
  -p label:$LABEL
