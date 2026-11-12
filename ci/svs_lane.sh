#!/bin/bash
#
# Runs the SVS CI lane inside the presubmit container (.devcontainer/Dockerfile),
# for .github/workflows/svs.yml and ci/svs_intel_evidence.sh: build, every unit
# test, and the SVS integration tests. With --svs-prebuilt it also records
# whether the LVQ/LeanVec tests ran, which they do only on an Intel CPU, and
# fails if that doesn't match the CPU.
#
# Usage: ci/svs_lane.sh [--svs-prebuilt] [--results=DIR]
#
# Writes DIR/summary.env, one KEY=value per line, and the logs. Readers parse
# it rather than source it. Exits 0 when the lane passes.

set -uo pipefail

ROOT_DIR=$(readlink -f "$(dirname "$0")/..")
cd "${ROOT_DIR}"
# The container's root needn't own the checkout. Without this git refuses it,
# and build.sh, reading the branch as "unknown", deletes the generated
# protobuf sources.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'

LANE=default
RESULTS=${ROOT_DIR}/test-results/svs
for arg in "$@"; do
  case $arg in
  --svs-prebuilt) LANE=prebuilt ;;
  --results=*) RESULTS=$(realpath -m "${arg#*=}") ;;
  *)
    echo "unknown argument: $arg" >&2
    exit 2
    ;;
  esac
done

BUILD_ARGS=()
BUILD_DIR=${ROOT_DIR}/.build-release${BUILD_DIR_SUFFIX:-}
if [[ ${LANE} == prebuilt ]]; then
  BUILD_ARGS+=(--svs-prebuilt)
  BUILD_DIR=${BUILD_DIR}-svs-prebuilt
fi
# The Valkey release integration/run.sh builds: one with the module
# external-memory API, which SVS reports its allocations through.
export VALKEY_VERSION=${VALKEY_VERSION:-9.2.0-rc2}
# Every SVS integration test file, and test_info.py for the SVS INFO fields.
INTEGRATION_FILTER="svs or lvq or leanvec or test_info"
# A reused directory would hold the last run's logs; start clean. Only a
# directory under test-results/ is ever removed.
if [[ ${RESULTS} != "${ROOT_DIR}"/test-results/?* ]]; then
  echo "--results must be under ${ROOT_DIR}/test-results/" >&2
  exit 2
fi
rm -rf "${RESULTS}"
mkdir -p "${RESULTS}"

strip_color() { sed 's/\x1b\[[0-9;]*m//g'; }

# --configure regenerates the protobuf sources in a reused build tree, as
# local runs use.
TZ=UTC ci/build_ubuntu.sh "${BUILD_ARGS[@]}" --configure --run-tests \
  --unittest-output="${RESULTS}/unit" >"${RESULTS}/unit.log" 2>&1
UNIT_EXIT=$?
TZ=UTC ci/build_ubuntu.sh "${BUILD_ARGS[@]}" --skip-prepare-env --no-build \
  --run-integration-tests="${INTEGRATION_FILTER}" \
  --integration-output="${RESULTS}/integration" >"${RESULTS}/integration.log" 2>&1
INTEGRATION_EXIT=$?

UNIT_RESULT=$(strip_color <"${RESULTS}/unit.log" |
  grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+' | tail -1)
INTEGRATION_RESULT=$(strip_color <"${RESULTS}/integration.log" |
  grep -E '^=+ .*[0-9]+ (passed|failed).* in ' | tail -1 | sed -E 's/^=+ //; s/ in [0-9.]+s.*//')
CPU_VENDOR=$(awk -F': ' '/^vendor_id/ {print $2; exit}' /proc/cpuinfo)
CPU_MODEL=$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo)
VALKEY_VERSION=$("${RESULTS}/integration/valkey-server" --version 2>/dev/null | head -1)

# Whether the LVQ/LeanVec tests ran or skipped: the unit suite that needs
# support, and the integration file.
count() { grep -cE "$1" || true; }
LVQ_UNIT_PASSED=0
LVQ_UNIT_SKIPPED=0
LVQ_INTEGRATION_PASSED=0
LVQ_INTEGRATION_SKIPPED=0
LVQ_RAN=no
LVQ_UNIT_EXIT=0
if [[ ${LANE} == prebuilt ]]; then
  # Bounded like ctest's run of the same binary (build.sh's TEST_TIMEOUT); a
  # timeout exits 124.
  timeout --kill-after=10 "${TEST_TIMEOUT:-60}" \
    "${BUILD_DIR}/tests/vector_svs_test" --gtest_filter='Kinds/VectorSVSLvqLeanVecTest.*' \
    >"${RESULTS}/lvq_unit.log" 2>&1
  LVQ_UNIT_EXIT=$?
  LVQ_UNIT_PASSED=$(count '^\[       OK \]' <"${RESULTS}/lvq_unit.log")
  # Each skipped test, not gtest's closing list of them.
  LVQ_UNIT_SKIPPED=$(count '^\[  SKIPPED \] Kinds/.* \([0-9]+ ms\)$' <"${RESULTS}/lvq_unit.log")
  PYTEST_LOG=${RESULTS}/integration/valkey-test-framework/pytest_output.log
  if [[ -f ${PYTEST_LOG} ]]; then
    # One line per test; parallel runs print the outcome before the name.
    lvq_lines=$(strip_color <"${PYTEST_LOG}" | grep -E 'test_svs_lvq_leanvec\.py::' || true)
    LVQ_INTEGRATION_PASSED=$(count '\bPASSED\b' <<<"${lvq_lines}")
    LVQ_INTEGRATION_SKIPPED=$(count '\bSKIPPED\b' <<<"${lvq_lines}")
  fi
  if ((LVQ_UNIT_EXIT == 0 && LVQ_UNIT_PASSED > 0 && LVQ_UNIT_SKIPPED == 0 && LVQ_INTEGRATION_PASSED > 0 &&
    LVQ_INTEGRATION_SKIPPED == 0)); then
    LVQ_RAN=yes
  fi
fi

STATUS=0
REASON=""
if ((UNIT_EXIT != 0)); then
  STATUS=1
  REASON+="unit tests failed; "
fi
if ((INTEGRATION_EXIT != 0)); then
  STATUS=1
  REASON+="integration tests failed; "
fi
if ((LVQ_UNIT_EXIT != 0)); then
  STATUS=1
  REASON+="the LVQ/LeanVec unit suite exited ${LVQ_UNIT_EXIT}; "
fi
# The prebuilt library enables LVQ/LeanVec on Intel CPUs only; anything else
# means the capability gate or the tests' skips are wrong.
if [[ ${LANE} == prebuilt ]]; then
  if [[ ${CPU_VENDOR} == GenuineIntel && ${LVQ_RAN} == no ]]; then
    STATUS=1
    REASON+="LVQ/LeanVec tests didn't all run on an Intel CPU; "
  elif [[ ${CPU_VENDOR} != GenuineIntel ]] && ((LVQ_UNIT_PASSED > 0 || LVQ_INTEGRATION_PASSED > 0)); then
    STATUS=1
    REASON+="LVQ/LeanVec tests ran on a ${CPU_VENDOR} CPU; "
  fi
fi

emit() { printf '%s=%s\n' "$1" "${2//$'\n'/ }"; }
{
  emit LANE "${LANE}"
  emit LANE_SHA "$(git rev-parse HEAD 2>/dev/null)"
  emit CPU_VENDOR "${CPU_VENDOR}"
  emit CPU_MODEL "${CPU_MODEL}"
  emit UNIT_EXIT "${UNIT_EXIT}"
  emit UNIT_RESULT "${UNIT_RESULT}"
  emit INTEGRATION_EXIT "${INTEGRATION_EXIT}"
  emit INTEGRATION_RESULT "${INTEGRATION_RESULT}"
  emit LVQ_RAN "${LVQ_RAN}"
  emit LVQ_UNIT "${LVQ_UNIT_PASSED} passed, ${LVQ_UNIT_SKIPPED} skipped"
  emit LVQ_INTEGRATION "${LVQ_INTEGRATION_PASSED} passed, ${LVQ_INTEGRATION_SKIPPED} skipped"
  emit PREBUILT "$(grep -oE 'svs-c-api[^"/]*\.tar\.gz' third_party/CMakeLists.txt | head -1)"
  emit VALKEY_VERSION "${VALKEY_VERSION}"
  emit STATUS "${STATUS}"
  emit REASON "${REASON%; }"
} >"${RESULTS}/summary.env"
cat "${RESULTS}/summary.env"
exit ${STATUS}
