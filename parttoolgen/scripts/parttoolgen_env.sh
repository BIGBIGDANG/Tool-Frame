#!/usr/bin/env bash
# Source this file before running PartToolGen commands.
#
# The supported runtime is the project-local Python 3.8 env
# `.venv_parttoolgen_py38`. A different env must be selected explicitly through
# PARTTOOLGEN_CORE_VENV for debugging; there is no implicit global-conda
# fallback in a legacy workspace.

_PTG_SAVED_SHELL_OPTIONS="$(set +o)"
_PTG_USER_CORE_VENV_SET="${PARTTOOLGEN_CORE_VENV+x}"
_PTG_USER_CORE_VENV_VALUE="${PARTTOOLGEN_CORE_VENV:-}"
set -euo pipefail

_ptg_finish() {
  eval "${_PTG_SAVED_SHELL_OPTIONS}"
}

_ptg_fail() {
  echo "$1" >&2
  if [[ "${2:-}" != "" ]]; then
    echo "$2" >&2
  fi
  _ptg_finish
}

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _PTG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _PTG_SCRIPT_DIR="$(pwd)"
fi

export PARTTOOLGEN_ROOT="${PARTTOOLGEN_ROOT:-$(cd "${_PTG_SCRIPT_DIR}/../.." && pwd)}"
_PTG_LOCAL_ENV_FILE="${PARTTOOLGEN_ROOT}/.parttoolgen.local.env"
if [[ "${PARTTOOLGEN_SKIP_LOCAL_ENV:-0}" != "1" && -f "${_PTG_LOCAL_ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_PTG_LOCAL_ENV_FILE}"
  set +a
fi
if [[ -n "${_PTG_USER_CORE_VENV_SET}" ]]; then
  export PARTTOOLGEN_CORE_VENV="${_PTG_USER_CORE_VENV_VALUE}"
fi
export PARTTOOLGEN_SRC="${PARTTOOLGEN_SRC:-${PARTTOOLGEN_ROOT}/parttoolgen/src}"
_PTG_LOCAL_ISAACGYM="${PARTTOOLGEN_ROOT}/artifacts/third_party/isaacgym/python"
if [[ -z "${PARTTOOLGEN_ISAACGYM:-}" ]]; then
  if [[ -d "${_PTG_LOCAL_ISAACGYM}" ]]; then
    export PARTTOOLGEN_ISAACGYM="${_PTG_LOCAL_ISAACGYM}"
  else
    export PARTTOOLGEN_ISAACGYM=""
  fi
fi
_PTG_LOCAL_SIMTOOLREAL_SITE="${PARTTOOLGEN_ROOT}/.venv_simtoolreal/lib/python3.8/site-packages"
if [[ -z "${PARTTOOLGEN_SIMTOOLREAL_SITE:-}" ]]; then
  if [[ -d "${_PTG_LOCAL_SIMTOOLREAL_SITE}" ]]; then
    export PARTTOOLGEN_SIMTOOLREAL_SITE="${_PTG_LOCAL_SIMTOOLREAL_SITE}"
  else
    export PARTTOOLGEN_SIMTOOLREAL_SITE=""
  fi
fi
_PTG_LOCAL_PY38_VENV="${PARTTOOLGEN_ROOT}/.venv_parttoolgen_py38"
if [[ -n "${_PTG_USER_CORE_VENV_SET}" ]]; then
  if [[ ! -x "${PARTTOOLGEN_CORE_VENV}/bin/python" ]]; then
    _ptg_fail "PARTTOOLGEN_CORE_VENV is set but has no executable python: ${PARTTOOLGEN_CORE_VENV}/bin/python"
    return 2 2>/dev/null || exit 2
  fi
  :
elif [[ -x "${_PTG_LOCAL_PY38_VENV}/bin/python" ]]; then
  export PARTTOOLGEN_CORE_VENV="${_PTG_LOCAL_PY38_VENV}"
elif [[ -n "${PARTTOOLGEN_CORE_VENV:-}" ]]; then
  if [[ ! -x "${PARTTOOLGEN_CORE_VENV}/bin/python" ]]; then
    _ptg_fail "Configured fallback PARTTOOLGEN_CORE_VENV has no executable python: ${PARTTOOLGEN_CORE_VENV}/bin/python"
    return 2 2>/dev/null || exit 2
  fi
else
  _ptg_fail \
    "Missing project Python 3.8 env: ${_PTG_LOCAL_PY38_VENV}" \
    "Create it or set PARTTOOLGEN_CORE_VENV explicitly."
  return 2 2>/dev/null || exit 2
fi
export PARTTOOLGEN_CORE_VENV
export PARTTOOLGEN_CORE_PYTHON="${PARTTOOLGEN_CORE_VENV}/bin/python"

_PTG_CORE_PY_VERSION="$("${PARTTOOLGEN_CORE_PYTHON}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${_PTG_CORE_PY_VERSION}" != "3.8" ]]; then
  _ptg_fail "ToolFrame requires Python 3.8, got ${_PTG_CORE_PY_VERSION}: ${PARTTOOLGEN_CORE_PYTHON}"
  return 2 2>/dev/null || exit 2
fi
_PTG_PYTHONPATH="${PARTTOOLGEN_SRC}:${PARTTOOLGEN_ROOT}/simtoolreal:${PARTTOOLGEN_ROOT}/simtoolreal/isaacgymenvs:${PARTTOOLGEN_ROOT}/simtoolreal/rl_games"
if [[ -n "${PARTTOOLGEN_ISAACGYM}" ]]; then
  _PTG_PYTHONPATH="${_PTG_PYTHONPATH}:${PARTTOOLGEN_ISAACGYM}"
fi
if [[ -n "${PARTTOOLGEN_SIMTOOLREAL_SITE}" ]]; then
  _PTG_PYTHONPATH="${_PTG_PYTHONPATH}:${PARTTOOLGEN_SIMTOOLREAL_SITE}"
fi
export PYTHONPATH="${_PTG_PYTHONPATH}:${PYTHONPATH:-}"

if [[ ! -x "${PARTTOOLGEN_CORE_PYTHON}" ]]; then
  _ptg_fail "Missing core Python: ${PARTTOOLGEN_CORE_PYTHON}"
  return 2 2>/dev/null || exit 2
fi

export VIRTUAL_ENV="${PARTTOOLGEN_CORE_VENV}"
export VIRTUAL_ENV_PROMPT="$(basename "${PARTTOOLGEN_CORE_VENV}")"
export PATH="${PARTTOOLGEN_CORE_VENV}/bin:${PATH}"
if [[ -d "${PARTTOOLGEN_CORE_VENV}/libpython_shim" ]]; then
  export LD_LIBRARY_PATH="${PARTTOOLGEN_CORE_VENV}/libpython_shim:${LD_LIBRARY_PATH:-}"
elif [[ "${CONDA_DEFAULT_ENV:-}" == "parttoolgen" && -d "${PARTTOOLGEN_ROOT}/artifacts/env_shims/global_parttoolgen_libpython" ]]; then
  export LD_LIBRARY_PATH="${PARTTOOLGEN_ROOT}/artifacts/env_shims/global_parttoolgen_libpython:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${PARTTOOLGEN_CORE_VENV}/lib:${LD_LIBRARY_PATH:-}"
fi
hash -r 2>/dev/null || true
_ptg_finish
unset _PTG_SAVED_SHELL_OPTIONS
unset _PTG_USER_CORE_VENV_SET
unset _PTG_USER_CORE_VENV_VALUE
unset -f _ptg_finish 2>/dev/null || true
unset -f _ptg_fail 2>/dev/null || true
