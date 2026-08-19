#!/usr/bin/env bash
# Refuse floating package-manager installs inside workflow `run:` blocks
# (Issue #169).
#
# Why this exists:
#   `uses:` references are SHA-pinned by policy (Issues #24 and #100), but a
#   `run:` block is not a manifest — no dependency quarantine covers it. An
#   install such as `npm install -g markdownlint-cli2` resolves whatever the
#   registry serves at that instant, so a hijacked release executes on the
#   runner, with the workflow's token and secrets in scope, the moment it is
#   published. Requiring an exact `pkg@x.y.z` pin means a compromised upload
#   cannot reach CI until a human bumps the pin in a reviewed PR.
#
# What counts as pinned:
#   An exact SemVer version — `markdownlint-cli2@0.23.2`, `@redocly/cli@2.6.1`.
#   Ranges (`^0.23.0`), dist-tags (`@latest`), workflow expressions and bare
#   package names are all floating and fail the gate.
#
# Deliberate floats are suppressed in-source, on or above the offending line:
#   # best-practice-ignore: BP-CI-INSTALL-PIN-npm-<package> — <reason>
#
# Exit codes: 0 every install is pinned (or suppressed), 1 a float was found,
# 2 invalid invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKFLOW_DIR="$REPO_ROOT/.github/workflows"
VERBOSE=0
EXIT_CODE=0

usage() {
  cat <<'EOF'
Usage: check-workflow-npm-pins.sh [--dir DIR] [--verbose]

Scans every workflow in DIR (default .github/workflows) and fails when a
`run:` step installs a package without an exact `name@x.y.z` version.

Suppress a deliberate float with a comment on or above the line:
  # best-practice-ignore: BP-CI-INSTALL-PIN-npm-<package> — <reason>

Exit codes: 0 all pinned, 1 a floating install was found, 2 invalid usage.
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

# An exact SemVer pin: 1.2.3, 1.2.3-rc.1, 1.2.3+build.4.
EXACT_VERSION='[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?'
PACKAGE_NAME='(@[a-z0-9][a-z0-9._-]*/)?[a-zA-Z0-9][a-zA-Z0-9._-]*'

# Strip surrounding quotes a workflow author may have written around a spec.
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

# Decide whether one package spec carries an exact pin.
check_spec() {
  local spec="$1" location="$2"
  local name="${spec%%@*}"
  # Scoped packages start with '@', so split on the *last* '@' for them.
  if [[ "$spec" == @* ]]; then
    name="@${spec:1}"
    name="${name%%@*}"
  fi

  if [[ "$spec" != *@* ]]; then
    report_fail "$location: '$spec' has no version — pin it as '$spec@<exact-version>'"
    return
  fi

  if [[ "$spec" =~ ^${PACKAGE_NAME}@${EXACT_VERSION}$ ]]; then
    report_ok "$location: $spec"
    return
  fi

  report_fail "$location: '$spec' is not an exact version — pin '$name@<exact-version>' so a hijacked release cannot execute on the runner"
}

scan_line() {
  local line="$1" location="$2"
  local -a tokens=()
  local mode="none" token spec
  read -r -a tokens <<< "$line"

  for token in ${tokens[@]+"${tokens[@]}"}; do
    case "$token" in
      '&&' | '||' | ';' | '|' | '>' | '>>' | '&')
        mode="none"
        continue
        ;;
      \#*)
        # A trailing comment ends the command.
        return
        ;;
    esac

    case "$mode" in
      none)
        case "$token" in
          npm | pnpm | yarn | bun) mode="subcommand" ;;
          npx | pnpx | bunx) mode="packages" ;;
        esac
        ;;
      subcommand)
        case "$token" in
          -*) ;; # a flag ahead of the subcommand, e.g. `yarn --silent add`
          global) ;; # `yarn global add <pkg>` — keep looking for the verb
          install | i | add | dlx | exec | create) mode="packages" ;;
          *) mode="none" ;; # ci, run, test, … install nothing by name
        esac
        ;;
      packages)
        spec="$(unquote "$token")"
        case "$spec" in
          -* | '') continue ;;                # flags
          */* ) [[ "$spec" != @*/* ]] && continue ;; # paths and flag values
          .* | '~'*) continue ;;              # local paths
        esac
        check_spec "$spec" "$location"
        ;;
    esac
  done
}

while IFS= read -r workflow; do
  workflow_name="$(basename "$workflow")"
  line_number=0
  previous_line=""
  while IFS= read -r line || [[ -n "$line" ]]; do
    line_number=$((line_number + 1))
    if [[ "$line$previous_line" =~ best-practice-ignore:[[:space:]]*BP-CI-INSTALL-PIN ]]; then
      report_ok "$workflow_name:$line_number: suppressed by best-practice-ignore"
      previous_line="$line"
      continue
    fi
    scan_line "$line" "$workflow_name:$line_number"
    previous_line="$line"
  done < "$workflow"
done < <(find "$WORKFLOW_DIR" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) | sort)

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "check-workflow-npm-pins: FAILED — pin the installs above, or suppress a deliberate float with '# best-practice-ignore: BP-CI-INSTALL-PIN-npm-<package> — <reason>'" >&2
  exit 1
fi

echo "check-workflow-npm-pins: every workflow install is pinned to an exact version"
