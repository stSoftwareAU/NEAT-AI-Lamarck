#!/usr/bin/env bash
# Refuse container image pins that are missing a tag or a digest (Issue #212).
#
# Why this exists:
#   A bare digest — `semgrep/semgrep@sha256:a9ea…` — is immutable, but every
#   dependency updater (Renovate's `github-actions` manager, Dependabot's
#   `docker` ecosystem) resolves a bump from the *tag* and rewrites the digest
#   beside it. A tagless digest gives them nothing to resolve, so the pin reads
#   as hardened while silently freezing at whatever the tag meant on the day it
#   was written. A bare tag is the opposite failure: mutable, so a compromised
#   namespace can re-push it under CI.
#
# What counts as pinned:
#   Both halves — `name:tag@sha256:<64 hex>`, e.g.
#   `semgrep/semgrep:1.86.0@sha256:a9ea…`. The digest keeps the image
#   byte-for-byte immutable; the tag is what an updater bumps from.
#
# Deliberate exceptions are suppressed in-source, on or above the offending
# line:
#   # best-practice-ignore: BP-CONTAINER-PIN-<image> — <reason>
#
# Exit codes: 0 every image is pinned (or suppressed), 1 a bad pin was found,
# 2 invalid invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="$REPO_ROOT/.github/workflows"
VERBOSE=0
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-workflow-container-pins.sh [--dir DIR] [--verbose]

Scans every workflow in DIR (default .github/workflows) and fails when an
`image:` value is not pinned as `name:tag@sha256:<64 hex>`.

Suppress a deliberate exception with a comment on or above the line:
  # best-practice-ignore: BP-CONTAINER-PIN-<image> — <reason>

Exit codes: 0 all pinned, 1 a bad pin was found, 2 invalid usage.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --dir)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --dir requires a directory argument" >&2
        usage >&2
        exit 2
      fi
      WORKFLOW_DIR="$2"
      shift 2
      ;;
    --verbose)
      VERBOSE=1
      shift
      ;;
    *)
      echo "FAIL: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -d "$WORKFLOW_DIR" ]]; then
  echo "FAIL: workflow directory not found: $WORKFLOW_DIR" >&2
  exit 2
fi

DIGEST='sha256:[0-9a-f]{64}'
TAG='[A-Za-z0-9_][A-Za-z0-9._-]*'

# Strip surrounding quotes a workflow author may have written around a value.
unquote() {
  local token="$1"
  token="${token%\"}"
  token="${token#\"}"
  token="${token%\'}"
  token="${token#\'}"
  printf '%s' "$token"
}

report_ok() {
  [[ "$VERBOSE" -eq 1 ]] && echo "OK   $1"
  return 0
}

report_fail() {
  echo "FAIL $1" >&2
  EXIT_CODE=1
}

# Decide whether one image reference carries both a tag and a digest.
check_image() {
  local spec="$1" location="$2"
  local repository tag_part last

  # A GitHub Actions expression, e.g. ${{ env.VERSION }}, resolves at run time.
  if [[ "$spec" =~ \$\{\{ ]]; then
    report_fail "$location: '$spec' is resolved at run time — pin the image as 'name:<tag>@sha256:<digest>'"
    return
  fi

  if [[ "$spec" != *@* ]]; then
    report_fail "$location: '$spec' has no digest — pin it as '$spec:<tag>@sha256:<digest>' so the tag cannot be re-pushed under CI"
    return
  fi

  repository="${spec%@*}"
  if [[ ! "${spec#*@}" =~ ^${DIGEST}$ ]]; then
    report_fail "$location: '$spec' has a malformed digest — expected 'sha256:' followed by 64 hex characters"
    return
  fi

  # Only the final path component can carry the tag; earlier colons are
  # registry ports (`registry:5000/team/image`).
  last="${repository##*/}"
  if [[ "$last" != *:* ]]; then
    report_fail "$location: '$spec' pins a bare digest with no tag — write '$repository:<tag>@${spec#*@}' so dependency updaters can bump it"
    return
  fi

  tag_part="${last#*:}"
  if [[ ! "$tag_part" =~ ^${TAG}$ ]]; then
    report_fail "$location: '$spec' has a malformed tag '$tag_part'"
    return
  fi

  report_ok "$location: $spec"
}

while IFS= read -r workflow; do
  workflow_name="$(basename "$workflow")"
  line_number=0
  previous_line=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    if [[ "$line$previous_line" =~ best-practice-ignore:[[:space:]]*BP-CONTAINER-PIN ]]; then
      report_ok "$workflow_name:$line_number: suppressed by best-practice-ignore"
      previous_line="$line"
      continue
    fi
    if [[ "$line" =~ ^[[:space:]]*image:[[:space:]]*([^[:space:]#].*)$ ]]; then
      value="${BASH_REMATCH[1]}"
      value="${value%%[[:space:]]#*}" # drop a trailing comment
      value="${value%"${value##*[![:space:]]}"}"
      check_image "$(unquote "$value")" "$workflow_name:$line_number"
    fi
    previous_line="$line"
  done < "$workflow"
done < <(find "$WORKFLOW_DIR" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) | sort)

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "check-workflow-container-pins: FAILED — pin the images above as 'name:<tag>@sha256:<digest>', or suppress a deliberate exception with '# best-practice-ignore: BP-CONTAINER-PIN-<image> — <reason>'" >&2
  exit 1
fi

echo "check-workflow-container-pins: every workflow container image is pinned by tag and digest"
