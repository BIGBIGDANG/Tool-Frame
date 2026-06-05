#!/usr/bin/env bash
set -euo pipefail

MODE="quick"
if [[ "${1:-}" == "--full" ]]; then
  MODE="full"
  shift
elif [[ "${1:-}" == "--quick" ]]; then
  shift
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: parttoolgen/scripts/run_checks.sh [--quick|--full]

--quick  Core-only checks suitable for GitHub CI and ordinary handoff.
--full   Quick checks plus the full pytest suite.
EOF
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "Unknown argument(s): $*" >&2
  echo "Run with --help for usage." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/parttoolgen_env.sh"

cd "${PARTTOOLGEN_ROOT}"

"${PARTTOOLGEN_CORE_PYTHON}" \
  parttoolgen/scripts/check_toolframe_env.py \
  --mode core >/dev/null

"${PARTTOOLGEN_CORE_PYTHON}" -m compileall -q \
  parttoolgen/src/parttoolgen \
  parttoolgen/src/parttoolgen/execution \
  parttoolgen/scripts

"${PARTTOOLGEN_CORE_PYTHON}" \
  parttoolgen/src/parttoolgen/validation/validate_parttoolgen.py \
  --help >/dev/null

"${PARTTOOLGEN_CORE_PYTHON}" \
  parttoolgen/scripts/run_tool_use_benchmark.py \
  --validate-registry >/dev/null

"${PARTTOOLGEN_CORE_PYTHON}" \
  parttoolgen/scripts/run_tool_use_benchmark.py \
  --suite current_all_runnable \
  --preflight-only \
  --skip-rollout \
  --aggregate none >/dev/null

"${PARTTOOLGEN_CORE_PYTHON}" -m pytest -q -p no:cacheprovider \
  parttoolgen/tests/test_check_toolframe_env.py \
  parttoolgen/tests/test_cli_help_smoke.py \
  parttoolgen/tests/test_run_tool_use_benchmark.py \
  parttoolgen/tests/test_run_image_grounded_task_spec.py \
  parttoolgen/tests/test_run_p3sam_tool_ref_pipeline.py \
  parttoolgen/tests/test_scripts_inventory.py \
  parttoolgen/tests/test_image_grounded_tool_refs.py \
  parttoolgen/tests/test_select_p3sam_tool_refs.py \
  parttoolgen/tests/test_tool_selection_evidence.py \
  parttoolgen/tests/test_visual_evidence_package.py

if [[ "${MODE}" == "full" ]]; then
  "${PARTTOOLGEN_CORE_PYTHON}" -m pytest -q -p no:cacheprovider parttoolgen/tests
fi

if [[ -d "artifacts/tool_prior/sampart3d_scale2" ]]; then
  _PTG_CHECK_TMP="$(mktemp -d)"
  trap 'rm -rf "${_PTG_CHECK_TMP:-}"' EXIT
  "${PARTTOOLGEN_CORE_PYTHON}" -m parttoolgen.checks.tool_prior_sanity \
    --run-root artifacts/tool_prior/sampart3d_scale2 \
    --report "${_PTG_CHECK_TMP}/tool_prior_sanity_report.json" \
    >/dev/null
fi
