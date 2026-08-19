#!/usr/bin/env bash
set -euo pipefail

if [ -f "$HOME/.bodymaps_env" ]; then
  source "$HOME/.bodymaps_env"
fi

CODE_ROOT=${CODE_ROOT:-/projects/bodymaps/users/xhan74/medical_agent/code/medical_agent}
PYTHON=${PYTHON:-/home/xhan74/envs/medical_agent/bin/python}

cd "$CODE_ROOT"
exec "$PYTHON" tools/dataset_delivery/labelcritic_72b_acceptance.py submit "$@"
