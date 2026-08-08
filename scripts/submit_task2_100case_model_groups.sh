#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. The formal Task2 production core is now manifest-driven
# and defaults to the 103-case x 22-target workflow.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/task2/submit_task2_formal_103cases.sh" "$@"
