#!/usr/bin/env bash
# Create the project-local Python 3.8 environment used by ToolFrame.
#
# Usage:
#   bash parttoolgen/scripts/bootstrap_parttoolgen_env.sh
#
# Optional overrides:
#   PARTTOOLGEN_PYTHON_VERSION=3.8.20
#   PARTTOOLGEN_CORE_VENV=/path/to/.venv_parttoolgen_py38
#   PARTTOOLGEN_REQUIREMENTS_FILE=/path/to/requirements.txt
#   PARTTOOLGEN_INSTALL_ROLLOUT=1

set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _PTG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _PTG_SCRIPT_DIR="$(pwd)"
fi

PARTTOOLGEN_ROOT="${PARTTOOLGEN_ROOT:-$(cd "${_PTG_SCRIPT_DIR}/../.." && pwd)}"
PARTTOOLGEN_PYTHON_VERSION="${PARTTOOLGEN_PYTHON_VERSION:-3.8.20}"
PARTTOOLGEN_CORE_VENV="${PARTTOOLGEN_CORE_VENV:-${PARTTOOLGEN_ROOT}/.venv_parttoolgen_py38}"
REQUIREMENTS_FILE="${PARTTOOLGEN_REQUIREMENTS_FILE:-${PARTTOOLGEN_ROOT}/requirements.txt}"
ROLLOUT_REQUIREMENTS_FILE="${PARTTOOLGEN_ROOT}/requirements-rollout.txt"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Missing requirements file: ${REQUIREMENTS_FILE}" >&2
  exit 2
fi

if command -v uv >/dev/null 2>&1; then
  uv python install "${PARTTOOLGEN_PYTHON_VERSION}"
  uv venv --seed --python "${PARTTOOLGEN_PYTHON_VERSION}" "${PARTTOOLGEN_CORE_VENV}"
else
  if ! command -v python3.8 >/dev/null 2>&1; then
    echo "Install uv or python3.8 first. Recommended: https://docs.astral.sh/uv/" >&2
    exit 2
  fi
  python3.8 -m venv "${PARTTOOLGEN_CORE_VENV}"
fi

"${PARTTOOLGEN_CORE_VENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${PARTTOOLGEN_CORE_VENV}/bin/python" -m pip install -r "${REQUIREMENTS_FILE}"
if [[ "${PARTTOOLGEN_INSTALL_ROLLOUT:-0}" == "1" ]]; then
  if [[ ! -f "${ROLLOUT_REQUIREMENTS_FILE}" ]]; then
    echo "Missing rollout requirements file: ${ROLLOUT_REQUIREMENTS_FILE}" >&2
    exit 2
  fi
  "${PARTTOOLGEN_CORE_VENV}/bin/python" -m pip install -r "${ROLLOUT_REQUIREMENTS_FILE}"
fi

cat <<EOF

Created ToolFrame environment:
  ${PARTTOOLGEN_CORE_VENV}

Installed requirements:
  ${REQUIREMENTS_FILE}

Next:
  source parttoolgen/scripts/parttoolgen_env.sh
  python parttoolgen/scripts/check_toolframe_env.py
  python -m pytest -q parttoolgen/tests/test_image_grounded_constraint_compiler.py \\
    parttoolgen/tests/test_image_grounded_mainline_adapter.py \\
    parttoolgen/tests/test_evaluation_report.py

EOF
