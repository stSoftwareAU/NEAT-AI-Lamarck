#!/usr/bin/env bash
# Require a push-based Dependabot advisory channel for the cargo dependencies
# (Issue #215).
#
# Why this exists:
#   `cargo audit` runs on every pull request and on a weekly Monday cron
#   (`.github/workflows/cargo-audit.yml`), and `rustsec/audit-check` runs inside
#   `.github/workflows/security.yml` on pull requests. Both are pull-based: the
#   lockfile does not change but the RustSec advisory database does, so an
#   advisory published against an already-pinned dependency is only seen at the
#   next PR or the next cron — a detection lag of up to ~6 days. Registering the
#   cargo ecosystem in `.github/dependabot.yml` activates GitHub's native
#   Dependabot advisory feed, which alerts the moment the advisory lands. This
#   gate keeps that registration from being deleted or quietly disabled.
#
# What the gate asserts, from the config's own contents:
#   1. The config exists where GitHub reads it (`.github/dependabot.yml`).
#   2. It declares `version: 2` — Dependabot rejects any other value.
#   3. Its `updates:` list carries a `package-ecosystem: cargo` entry, so the
#      crate's dependencies are actually in the feed.
#   4. That entry names a `directory` (or `directories`) and a
#      `schedule.interval` Dependabot accepts, so the entry is usable rather
#      than a stub GitHub would reject.
#
# Exit codes: 0 the advisory channel is configured, 1 it is missing or unusable,
# 2 invalid invocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="$REPO_ROOT/.github/dependabot.yml"
ECOSYSTEM="cargo"
VERBOSE=0
EXIT_CODE=0

# Dependabot's accepted `schedule.interval` values.
VALID_INTERVALS=("daily" "weekly" "monthly" "quarterly" "semiannually" "yearly")

usage() {
  cat <<'EOF'
Usage: check-dependabot-config.sh [--config PATH] [--ecosystem NAME] [--verbose]

Fails when the repository has no usable Dependabot advisory channel for its
cargo dependencies — the push-based half of the supply-chain scanning that
`cargo audit` covers only on pull requests and a weekly cron.

  --config PATH      Dependabot config (default .github/dependabot.yml)
  --ecosystem NAME   Ecosystem that must be registered (default cargo)
  --verbose          Report each requirement checked, not just failures.

Exit codes: 0 configured, 1 missing or unusable, 2 invalid usage.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --config requires a path" >&2
        usage >&2
        exit 2
      fi
      CONFIG="$2"
      shift 2
      ;;
    --ecosystem)
      if [[ $# -lt 2 ]]; then
        echo "FAIL: --ecosystem requires a name" >&2
        usage >&2
        exit 2
      fi
      ECOSYSTEM="$2"
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

report_ok() {
  [[ "$VERBOSE" -eq 1 ]] && echo "OK   $1"
  return 0
}

report_fail() {
  echo "FAIL $1" >&2
  EXIT_CODE=1
}

# An absent config is the very finding this gate exists for, so it is a failure
# (exit 1), not a usage error.
if [[ ! -f "$CONFIG" ]]; then
  echo "FAIL no Dependabot config at $CONFIG — without it GitHub raises no advisory alert for cargo dependencies, so a freshly published RustSec advisory is only seen at the next pull request or weekly cargo-audit cron (Issue #215)" >&2
  echo "check-dependabot-config: FAILED — add a 'package-ecosystem: \"$ECOSYSTEM\"' entry under 'updates:' in $CONFIG" >&2
  exit 1
fi

# Emit `<index> <dotted.key> <value>` for every scalar inside each item of the
# top-level `updates:` list, plus `root version <value>` for the file's version
# key. Nested mappings (`schedule:` → `interval:`) are flattened onto the dotted
# path, so `schedule.interval` is addressable directly. Comments are stripped
# first, so a commented-out entry is invisible — exactly as it is to Dependabot.
config_entries() {
  awk '
    function unquote(value) {
      gsub(/^[ \t]+|[ \t]+$/, "", value)
      if (value ~ /^".*"$/ || value ~ /^'"'"'.*'"'"'$/) {
        value = substr(value, 2, length(value) - 2)
      }
      return value
    }
    function path_prefix(   i, out) {
      out = ""
      for (i = 1; i <= depth; i++) out = out names[i] "."
      return out
    }
    {
      line = $0
      sub(/\r$/, "", line)
      sub(/(^|[ \t])#.*$/, "", line)
      if (line ~ /^[ \t]*$/) next
      indent = match(line, /[^ \t]/) - 1
      content = substr(line, indent + 1)
      sub(/[ \t]+$/, "", content)

      # A top-level key resets the scope.
      if (indent == 0) {
        in_updates = (content ~ /^updates:/)
        depth = 0
        if (content ~ /^version:/) {
          value = content
          sub(/^version:[ \t]*/, "", value)
          print "root version " unquote(value)
        }
        next
      }
      if (!in_updates) next

      # A list item at the shallowest indent seen starts a new update entry;
      # deeper `- ` lines are sequence values of the key above them.
      if (content ~ /^-[ \t]*/ && (item_indent == "" || indent <= item_indent)) {
        item_indent = indent
        entry++
        depth = 0
        content = substr(content, 2)
        sub(/^[ \t]*/, "", content)
        indent = indent + 2
        if (content == "") next
      } else if (content ~ /^-[ \t]*/) {
        # Sequence value under the current key, e.g. `directories: [- "/"]`.
        if (entry > 0 && depth > 0) {
          value = content
          sub(/^-[ \t]*/, "", value)
          prefix = path_prefix()
          sub(/\.$/, "", prefix)
          if (unquote(value) != "") print entry " " prefix " " unquote(value)
        }
        next
      }

      if (entry == 0) next
      while (depth > 0 && indent <= indents[depth]) depth--

      name = content
      sub(/:.*$/, "", name)
      name = unquote(name)
      value = content
      if (content !~ /^[^:]+:/) next
      sub(/^[^:]*:[ \t]*/, "", value)
      value = unquote(value)

      if (value == "") {
        depth++
        names[depth] = name
        indents[depth] = indent
      } else {
        print entry " " path_prefix() name " " value
      }
    }
  ' "$1"
}

ENTRIES="$(config_entries "$CONFIG")"

entry_value() {
  local index="$1" key="$2" line
  while read -r line; do
    [[ -z "$line" ]] && continue
    # shellcheck disable=SC2086 # deliberate word split of "<index> <key> <value>"
    set -- $line
    if [[ "$1" == "$index" && "$2" == "$key" ]]; then
      shift 2
      printf '%s' "$*"
      return 0
    fi
  done <<< "$ENTRIES"
  return 1
}

config_name="$(basename "$CONFIG")"

# Rule 2 — Dependabot only accepts version 2.
version="$(entry_value root version || true)"
if [[ "$version" != "2" ]]; then
  if [[ -z "$version" ]]; then
    report_fail "$config_name: no top-level 'version' key — Dependabot ignores the file; it must declare 'version: 2'"
  else
    report_fail "$config_name: 'version: $version' is not a schema Dependabot reads — it must declare 'version: 2'"
  fi
else
  report_ok "$config_name: declares version: 2"
fi

# Rule 3 — the ecosystem carrying the advisory feed must be registered.
ecosystem_index=""
while read -r index key value; do
  [[ "$key" == "package-ecosystem" && "$value" == "$ECOSYSTEM" ]] || continue
  ecosystem_index="$index"
  break
done <<< "$ENTRIES"

if [[ -z "$ecosystem_index" ]]; then
  report_fail "$config_name: no 'package-ecosystem: \"$ECOSYSTEM\"' entry under 'updates:' — GitHub raises no advisory alert for the crate's dependencies, leaving the weekly cargo-audit cron as the only new-CVE detection"
else
  report_ok "$config_name: registers the '$ECOSYSTEM' ecosystem (update entry $ecosystem_index)"

  # Rule 4 — the entry must be usable: a directory and an accepted interval.
  directory="$(entry_value "$ecosystem_index" directory || true)"
  if [[ -z "$directory" ]]; then
    directory="$(entry_value "$ecosystem_index" directories || true)"
  fi
  if [[ -z "$directory" ]]; then
    report_fail "$config_name: the '$ECOSYSTEM' entry names no 'directory' (or 'directories') — Dependabot rejects the entry, so the feed never activates"
  else
    report_ok "$config_name: '$ECOSYSTEM' entry scans '$directory'"
  fi

  interval="$(entry_value "$ecosystem_index" schedule.interval || true)"
  if [[ -z "$interval" ]]; then
    report_fail "$config_name: the '$ECOSYSTEM' entry has no 'schedule.interval' — it is required, one of: ${VALID_INTERVALS[*]}"
  else
    valid=0
    for candidate in "${VALID_INTERVALS[@]}"; do
      [[ "$interval" == "$candidate" ]] && valid=1
    done
    if [[ "$valid" -eq 1 ]]; then
      report_ok "$config_name: '$ECOSYSTEM' entry runs on a '$interval' schedule"
    else
      report_fail "$config_name: 'schedule.interval: $interval' is not accepted by Dependabot — use one of: ${VALID_INTERVALS[*]}"
    fi
  fi
fi

if [[ "$EXIT_CODE" -ne 0 ]]; then
  echo "check-dependabot-config: FAILED — $config_name must register a usable '$ECOSYSTEM' ecosystem so advisories are pushed, not only polled (Issue #215)" >&2
  exit 1
fi

echo "check-dependabot-config: $config_name registers a usable '$ECOSYSTEM' advisory channel"
