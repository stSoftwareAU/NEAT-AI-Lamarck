#!/usr/bin/env bash
# Refuse a cargo-quality.yml trigger that duplicates ci.yml's gate (Issue #213).
#
# Why this exists:
#   `cargo-quality.yml` exists to cover the feature-branch PRs `ci.yml` skips,
#   but its `branches: ["**"]` filter is a genuine wildcard: it also matches
#   `Develop` and `milestone/**`, the exact branches `ci.yml`'s `quality` job
#   already gates. Every PR into those branches then ran the same
#   `cargo fmt --all -- --check` and a near-identical clippy lint twice, for
#   zero extra coverage.
#
# What the gate asserts, from the two workflows' own `pull_request` filters:
#   1. `ci.yml` declares which branches it gates (a missing filter means it
#      gates every branch, so no exclusion could ever remove the overlap).
#   2. `cargo-quality.yml` still runs on `pull_request` — deleting the trigger
#      is not a fix, it is a loss of feature-branch coverage.
#   3. No branch `ci.yml` gates is also matched by `cargo-quality.yml`.
#   4. A feature branch — one `ci.yml` does not gate — is still matched by
#      `cargo-quality.yml`.
#
# Branch globs follow GitHub's filter semantics: `*` does not cross `/`, `**`
# does, `?` matches one character other than `/`.
#
# Exit codes: 0 no duplicated coverage, 1 an overlap (or a coverage loss) was
# found, 2 invalid invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CI_WORKFLOW="$REPO_ROOT/.github/workflows/ci.yml"
QUALITY_WORKFLOW="$REPO_ROOT/.github/workflows/cargo-quality.yml"
VERBOSE=0
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-cargo-quality-overlap.sh [--ci PATH] [--quality PATH] [--verbose]

Fails when the standalone cargo-quality workflow's `pull_request` branch filter
also matches a branch the CI workflow already gates, or when it no longer
matches any feature branch at all.

  --ci PATH        Authoritative CI workflow (default .github/workflows/ci.yml)
  --quality PATH   Standalone quality workflow
                   (default .github/workflows/cargo-quality.yml)
  --verbose        Report each branch that was checked, not just failures.

Exit codes: 0 no duplicated coverage, 1 an overlap or coverage loss, 2 invalid
usage.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --ci)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --ci requires a workflow path" >&2
        usage >&2
        exit 2
      fi
      CI_WORKFLOW="$2"
      shift 2
      ;;
    --quality)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --quality requires a workflow path" >&2
        usage >&2
        exit 2
      fi
      QUALITY_WORKFLOW="$2"
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

for workflow in "$CI_WORKFLOW" "$QUALITY_WORKFLOW"; do
  if [[ ! -f "$workflow" ]]; then
    echo "FAIL: workflow file not found: $workflow" >&2
    exit 2
  fi
done

report_ok() {
  [[ "$VERBOSE" -eq 1 ]] && echo "OK   $1"
  return 0
}

report_fail() {
  echo "FAIL $1" >&2
  EXIT_CODE=1
}

# Emit `<key> <pattern>` for every entry of `on.pull_request.branches` and
# `on.pull_request.branches-ignore`, in either the inline (`[a, b]`) or block
# (`- a`) form. Filters belonging to other events (`push`) are ignored.
pr_branch_filter() {
  awk '
    function unquote(value) {
      gsub(/^[ \t]+|[ \t]+$/, "", value)
      if (value ~ /^".*"$/ || value ~ /^'"'"'.*'"'"'$/) {
        value = substr(value, 2, length(value) - 2)
      }
      return value
    }
    function emit_inline(name, list,   parts, i, n) {
      gsub(/^\[|\]$/, "", list)
      n = split(list, parts, ",")
      for (i = 1; i <= n; i++) {
        if (unquote(parts[i]) != "") print name " " unquote(parts[i])
      }
    }
    {
      line = $0
      sub(/\r$/, "", line)
      sub(/(^|[ \t])#.*$/, "", line)
      if (line ~ /^[ \t]*$/) next
      indent = match(line, /[^ \t]/) - 1
      content = substr(line, indent + 1)
      sub(/[ \t]+$/, "", content)

      # A top-level key resets the scope; only `on:` is of interest. YAML 1.1
      # readers fold the unquoted key to `true`, so accept every spelling.
      if (indent == 0) {
        in_on = (content ~ /^("on"|'"'"'on'"'"'|on|true):/)
        in_pr = 0
        key = ""
        next
      }
      if (!in_on) next

      if (key != "" && indent > key_indent && content ~ /^-[ \t]*/) {
        item = content
        sub(/^-[ \t]*/, "", item)
        if (unquote(item) != "") print key " " unquote(item)
        next
      }
      if (key != "" && indent <= key_indent) key = ""
      if (in_pr && indent <= pr_indent) in_pr = 0

      if (content ~ /^pull_request:/) {
        in_pr = 1
        pr_indent = indent
        next
      }
      if (!in_pr) next

      if (content ~ /^branches(-ignore)?:/) {
        name = content
        sub(/:.*$/, "", name)
        value = content
        sub(/^[^:]*:[ \t]*/, "", value)
        if (value ~ /^\[/) {
          emit_inline(name, value)
        } else if (value == "") {
          key = name
          key_indent = indent
        } else {
          print name " " unquote(value)
        }
      }
    }
  ' "$1"
}

# Translate a GitHub branch glob into an ERE anchored at both ends.
glob_to_regex() {
  local pattern="$1" out="" index=0 char
  while ((index < ${#pattern})); do
    char="${pattern:index:1}"
    case "$char" in
      '*')
        if [[ "${pattern:index+1:1}" == '*' ]]; then
          out+='.*'
          index=$((index + 1))
        else
          out+='[^/]*'
        fi
        ;;
      '?') out+='[^/]' ;;
      [a-zA-Z0-9_/-]) out+="$char" ;;
      *) out+="\\$char" ;;
    esac
    index=$((index + 1))
  done
  printf '^%s$' "$out"
}

glob_matches() {
  local pattern="$1" branch="$2" regex
  regex="$(glob_to_regex "$pattern")"
  [[ "$branch" =~ $regex ]]
}

# A concrete branch name that the pattern gates, so two globs can be compared
# through a value both can be tested against.
sample_branch() {
  local sample="${1//\*\*/sample\/branch}"
  sample="${sample//\*/sample}"
  printf '%s' "${sample//\?/x}"
}

read_patterns() {
  local workflow="$1" wanted="$2" key pattern
  while read -r key pattern; do
    [[ "$key" == "$wanted" ]] && printf '%s\n' "$pattern"
  done < <(pr_branch_filter "$workflow")
}

quality_name="$(basename "$QUALITY_WORKFLOW")"
ci_name="$(basename "$CI_WORKFLOW")"

mapfile -t CI_BRANCHES < <(read_patterns "$CI_WORKFLOW" branches)
mapfile -t QUALITY_INCLUDE < <(read_patterns "$QUALITY_WORKFLOW" branches)
mapfile -t QUALITY_IGNORE < <(read_patterns "$QUALITY_WORKFLOW" branches-ignore)

if ! grep -qE '^[[:space:]]*pull_request:' "$QUALITY_WORKFLOW"; then
  echo "FAIL $quality_name: no pull_request trigger — feature-branch PRs would lose their fmt + clippy gate entirely" >&2
  exit 1
fi

if [[ "${#CI_BRANCHES[@]}" -eq 0 ]]; then
  echo "FAIL $ci_name: no 'on.pull_request.branches' filter — it gates every branch, so $quality_name duplicates it on every PR" >&2
  exit 1
fi

# Does the quality workflow run for a PR into this branch?
quality_runs_on() {
  local branch="$1" pattern
  for pattern in ${QUALITY_IGNORE[@]+"${QUALITY_IGNORE[@]}"}; do
    glob_matches "$pattern" "$branch" && return 1
  done
  if [[ "${#QUALITY_INCLUDE[@]}" -gt 0 ]]; then
    for pattern in "${QUALITY_INCLUDE[@]}"; do
      glob_matches "$pattern" "$branch" && return 0
    done
    return 1
  fi
  return 0
}

# Rule 3 — nothing ci.yml gates may also be matched here.
for ci_pattern in "${CI_BRANCHES[@]}"; do
  branch="$(sample_branch "$ci_pattern")"
  if quality_runs_on "$branch"; then
    report_fail "$quality_name: also runs on '$ci_pattern' (e.g. '$branch'), which $ci_name already gates — exclude it, e.g. 'branches-ignore: [$ci_pattern]'"
  else
    report_ok "$quality_name: '$ci_pattern' excluded — gated once, by $ci_name"
  fi
done

# Rule 4 — the feature branches this workflow exists for must still be covered.
FEATURE_SAMPLES=("issue-213-fmt-gate" "feature/stacked/pr")
covered=0
considered=0
for sample in "${FEATURE_SAMPLES[@]}"; do
  gated_by_ci=0
  for ci_pattern in "${CI_BRANCHES[@]}"; do
    glob_matches "$ci_pattern" "$sample" && gated_by_ci=1
  done
  [[ "$gated_by_ci" -eq 1 ]] && continue
  considered=$((considered + 1))
  if quality_runs_on "$sample"; then
    covered=$((covered + 1))
    report_ok "$quality_name: covers feature branch '$sample'"
  else
    report_fail "$quality_name: no feature branch coverage for '$sample' — the workflow exists for the branches $ci_name skips"
  fi
done

if [[ "$considered" -gt 0 && "$covered" -eq 0 ]]; then
  report_fail "$quality_name: no feature branch is matched at all — the branch filter excludes every branch $ci_name does not gate"
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "check-cargo-quality-overlap: FAILED — $quality_name must cover only the branches $ci_name skips (Issue #213)" >&2
  exit 1
fi

echo "check-cargo-quality-overlap: $quality_name covers only the branches $ci_name does not gate"
