#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT=${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PYTHON=${PYTHON:-python}
MANIFEST=${MANIFEST:-$CODE_ROOT/configs/dataset_delivery/totalsegmentator_brain_ventricle_offline_manifest.json}
BUNDLE_ROOT=${MEDAI_TOTALSEG_HOME:-${MEDAI_TOTALSEG_BUNDLE:-$CODE_ROOT/offline_assets/totalsegmentator}}
RUNTIME_HOME=${MEDAI_TOTALSEG_RUNTIME_HOME:-$CODE_ROOT/.runtime/totalsegmentator}
ENV_FILE=${ENV_FILE:-$CODE_ROOT/.runtime/totalsegmentator_offline.env}

cd "$CODE_ROOT"

if [ ! -f "$MANIFEST" ]; then
  echo "TOTALSEG_OFFLINE_MANIFEST_MISSING: $MANIFEST" >&2
  exit 2
fi

if [ ! -d "$BUNDLE_ROOT/nnunet/results" ]; then
  echo "TOTALSEG_OFFLINE_ASSET_MISSING: expected nnunet/results under $BUNDLE_ROOT" >&2
  echo "Set MEDAI_TOTALSEG_BUNDLE or MEDAI_TOTALSEG_HOME to the legally synchronized TotalSegmentator offline bundle." >&2
  exit 2
fi

mkdir -p "$(dirname "$RUNTIME_HOME")" "$(dirname "$ENV_FILE")"
if [ "$RUNTIME_HOME" != "$BUNDLE_ROOT" ]; then
  rm -f "$RUNTIME_HOME"
  ln -s "$BUNDLE_ROOT" "$RUNTIME_HOME"
fi

cat > "$ENV_FILE" <<EOF
export MEDAI_TOTALSEG_HOME="$RUNTIME_HOME"
export TOTALSEG_HOME_DIR="$RUNTIME_HOME"
export MEDAI_TOTALSEG_OFFLINE=1
export MEDAI_TOTALSEG_MANIFEST="$MANIFEST"
EOF

MEDAI_TOTALSEG_HOME="$RUNTIME_HOME" \
TOTALSEG_HOME_DIR="$RUNTIME_HOME" \
MEDAI_TOTALSEG_OFFLINE=1 \
MEDAI_TOTALSEG_MANIFEST="$MANIFEST" \
"$PYTHON" tools/dataset_delivery/totalseg_brain_ventricle_offline.py \
  --verify \
  --home "$RUNTIME_HOME" \
  --manifest "$MANIFEST"

echo "TOTALSEG_OFFLINE_BUNDLE_VALIDATED=1"
echo "TOTALSEG_OFFLINE_ENV=$ENV_FILE"
