#!/bin/bash
#
# Runs the svs-prebuilt lane of .github/workflows/svs.yml on this machine, for
# a PR whose CI runner wasn't an Intel CPU, and with --post records the result
# on the PR: the svs-prebuilt/intel commit status on the PR's head commit, and
# a comment with the environment and results.
#
# Usage: ci/svs_intel_evidence.sh <pr-number> [--post]
#
# Needs an Intel CPU, docker, an authenticated gh, and a checkout of the PR's
# head commit without uncommitted changes. Behind a proxy, export HTTP_PROXY,
# HTTPS_PROXY and NO_PROXY. REPO overrides the repository (default: origin's).

set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

PR=""
POST=no
for arg in "$@"; do
  case $arg in
  --post) POST=yes ;;
  *[!0-9]*) die "usage: $0 <pr-number> [--post]" ;;
  *) PR=$arg ;;
  esac
done
[[ -n ${PR} ]] || die "usage: $0 <pr-number> [--post]"

ROOT_DIR=$(readlink -f "$(dirname "$0")/..")
cd "${ROOT_DIR}"
REPO=${REPO:-$(git remote get-url origin | sed -E 's|.*github.com[/:]([^/]+/[^/]+)$|\1|; s|\.git$||')}
CONTEXT=svs-prebuilt/intel
BASE=intel-svs-vamana-submodule-leanvec-lvq

grep -q '^vendor_id.*GenuineIntel' /proc/cpuinfo || die "this machine's CPU isn't Intel"
read -r HEAD_SHA PR_BASE < <(gh pr view "${PR}" -R "${REPO}" --json headRefOid,baseRefName \
  --jq '"\(.headRefOid) \(.baseRefName)"')
[[ ${PR_BASE} == "${BASE}" ]] || die "PR #${PR} targets ${PR_BASE}, not ${BASE}"
[[ $(git rev-parse HEAD) == "${HEAD_SHA}" ]] ||
  die "check out PR #${PR}'s head commit ${HEAD_SHA} first (gh pr checkout ${PR})"
# Untracked files count too (the run would collect an untracked test); only
# this script's results and cache may exist.
[[ -z $(git status --porcelain --untracked-files=all | grep -vE '^\?\? (test-results|\.ccache)/') ]] ||
  die "the checkout has uncommitted or untracked files"
git submodule update --init --recursive

docker build -q \
  --build-arg HTTP_PROXY="${HTTP_PROXY:-}" --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --build-arg NO_PROXY="${NO_PROXY:-}" \
  -t presubmit-image -f .devcontainer/Dockerfile . >/dev/null

# A worktree's .git points into the main repository by absolute path.
GIT_COMMON_DIR=$(cd "$(git rev-parse --git-common-dir)" && pwd)
RESULTS=test-results/svs-intel-evidence
# Created here, so the container (root) doesn't own it.
mkdir -p test-results
# build.sh points the checkout's compile_commands.json at its build tree;
# restore whatever was there (a link, a file or nothing) on any exit.
COMPILE_DB_BACKUP=$(mktemp -d)
if [[ -e compile_commands.json || -L compile_commands.json ]]; then
  cp -P compile_commands.json "${COMPILE_DB_BACKUP}/"
fi
restore_compile_db() {
  rm -f compile_commands.json
  if [[ -e ${COMPILE_DB_BACKUP}/compile_commands.json || -L ${COMPILE_DB_BACKUP}/compile_commands.json ]]; then
    cp -P "${COMPILE_DB_BACKUP}/compile_commands.json" .
  fi
  rm -rf "${COMPILE_DB_BACKUP}"
}
trap restore_compile_db EXIT
LANE_EXIT=0
docker run --privileged --rm --user 0:0 \
  -v "${ROOT_DIR}:/workspace" -v "${GIT_COMMON_DIR}:${GIT_COMMON_DIR}" -w /workspace \
  -e CCACHE_DIR=/workspace/.ccache -e CCACHE_MAXSIZE=1G \
  -e HTTP_PROXY="${HTTP_PROXY:-}" -e HTTPS_PROXY="${HTTPS_PROXY:-}" -e NO_PROXY="${NO_PROXY:-}" \
  -e http_proxy="${http_proxy:-}" -e https_proxy="${https_proxy:-}" -e no_proxy="${no_proxy:-}" \
  -e BUILD_DIR_SUFFIX=-evidence \
  presubmit-image \
  bash -c "ci/svs_lane.sh --svs-prebuilt --results=/workspace/${RESULTS}; status=\$?; \
           find /workspace -xdev -user 0 -exec chown -h $(id -u):$(id -g) {} +; \
           exit \$status" || LANE_EXIT=$?

# The summary is the container's output: parsed, never sourced, so it can
# neither run code here nor replace HEAD_SHA or LANE_EXIT.
read_summary() {  # read_summary <file> <key>...: assigns only the named keys
  local file=$1 line key want
  shift
  while IFS= read -r line; do
    key=${line%%=*}
    for want in "$@"; do
      [[ ${key} == "${want}" ]] && printf -v "${key}" '%s' "${line#*=}"
    done
  done <"${file}"
  return 0
}
[[ -f ${RESULTS}/summary.env ]] || die "the lane didn't finish; see ${RESULTS}/*.log"
LANE_SHA="" LVQ_RAN=no
read_summary "${RESULTS}/summary.env" LANE_SHA CPU_MODEL UNIT_RESULT INTEGRATION_RESULT \
  LVQ_RAN LVQ_UNIT LVQ_INTEGRATION PREBUILT VALKEY_VERSION REASON
[[ ${LANE_SHA} == "${HEAD_SHA}" ]] || die "the lane ran on ${LANE_SHA}, not ${HEAD_SHA}"
if [[ ${LANE_EXIT} == 0 && ${LVQ_RAN} == yes ]]; then
  STATE=success
else
  STATE=failure
fi

COMMENT=$(
  cat <<EOF
**svs-prebuilt/intel: ${STATE}** for ${HEAD_SHA}, run locally with \`ci/svs_intel_evidence.sh\`.

| | |
| --- | --- |
| CPU | ${CPU_MODEL} |
| Unit tests | ${UNIT_RESULT:-none} |
| SVS integration tests | ${INTEGRATION_RESULT:-none} |
| LVQ/LeanVec unit suite | ${LVQ_UNIT} |
| LVQ/LeanVec integration tests | ${LVQ_INTEGRATION} |
| Prebuilt SVS C API | \`${PREBUILT}\` |
| Valkey server | ${VALKEY_VERSION:-unknown} |
${REASON:+
Failed: ${REASON}}
EOF
)
echo "${COMMENT}"

if [[ ${POST} == no ]]; then
  echo
  echo "Not posted. Re-run with --post to set ${CONTEXT} on ${HEAD_SHA} and comment on PR #${PR}."
  exit $((LANE_EXIT != 0))
fi
COMMENT_URL=$(gh api -X POST "repos/${REPO}/issues/${PR}/comments" -f body="${COMMENT}" --jq .html_url)
DESCRIPTION="Local on ${CPU_MODEL}: LVQ/LeanVec unit ${LVQ_UNIT}; integration ${LVQ_INTEGRATION}"
gh api -X POST "repos/${REPO}/statuses/${HEAD_SHA}" -f state="${STATE}" -f context="${CONTEXT}" \
  -f description="${DESCRIPTION:0:140}" -f target_url="${COMMENT_URL}" >/dev/null
echo "Posted ${COMMENT_URL}; ${CONTEXT} is ${STATE} on ${HEAD_SHA}."
[[ ${STATE} == success ]]
