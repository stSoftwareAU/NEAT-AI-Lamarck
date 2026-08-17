#!/usr/bin/env bash
# check-lamarck-version-order.sh — refuse neat_ai_lamarck version downgrades.
#
# Usage: check-lamarck-version-order.sh <base_version> <head_version>
#
# Compares two plain MAJOR.MINOR.PATCH tokens with `sort -V` (fleet-portable
# semver order, same as NEAT-AI's update-package-version.yml).
#
# Exit codes:
#   0  head >= base (equal or ahead — OK)
#   1  head < base (downgrade — refuse)
#   2  usage / parse error
set -euo pipefail

die() {
  echo "check-lamarck-version-order.sh: $1" >&2
  exit 2
}

[ "$#" -eq 2 ] || die "usage: check-lamarck-version-order.sh <base_version> <head_version>"

base="${1#v}"
head="${2#v}"

for v in "$base" "$head"; do
  if ! printf '%s' "$v" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    die "malformed version (expected x.y.z): $v"
  fi
done

# sort -V: lowest first. If head is the lowest and differs from base, head < base.
lowest="$(printf '%s\n%s\n' "$base" "$head" | sort -V | head -n1)"
if [[ "$lowest" == "$head" && "$head" != "$base" ]]; then
  echo "check-lamarck-version-order.sh: version downgraded: $base -> $head" >&2
  exit 1
fi

echo "check-lamarck-version-order.sh: OK ($base -> $head)"
exit 0
