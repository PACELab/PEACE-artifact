#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [-n|--dry-run] <crossvalid-root>" >&2
  exit 1
}

dry_run=false
root=""

while (( "$#" )); do
  case "$1" in
    -n|--dry-run) dry_run=true; shift ;;
    -h|--help) usage ;;
    *) if [[ -z "${root}" ]]; then root="$1"; shift; else usage; fi ;;
  esac
done

[[ -n "${root:-}" ]] || usage
command -v zip >/dev/null 2>&1 || { echo "zip not found on PATH"; exit 1; }

zip_one() {
  local src="$1"
  local dst="${src}.zip"

  if $dry_run; then
    echo "[DRY-RUN] Would create: $dst"
    echo "[DRY-RUN] Would remove: $src"
    return 0
  fi

  echo "Compressing: $src -> $dst"
  rm -f "$dst" 2>/dev/null || true
  if zip -jq "$dst" "$src" >/dev/null; then
    rm -f "$src"
    echo "OK: archived and removed original"
    return 0
  else
    echo "ERROR: zip failed for $src (original kept)" >&2
    return 1
  fi
}

processed=0
failed=0

# AutoML: any file containing 'AutoML' under AutoML dirs, excluding *.txt and *.zip
while IFS= read -r -d '' f; do
  if zip_one "$f"; then ((processed++)); else ((failed++)); fi
done < <(find "$root" -type f -path '*/AutoML/*' -name '*AutoML*' ! -name '*.txt' ! -name '*.zip' -print0)

# Non-AutoML: *.pkl whose basename contains the parent dir name (modeltype)
while IFS= read -r -d '' f; do
  parent="$(basename "$(dirname "$f")")"
  [[ "$parent" == "AutoML" ]] && continue
  base="$(basename "$f")"
  [[ "$base" == *"$parent"*.pkl ]] || continue
  if zip_one "$f"; then ((processed++)); else ((failed++)); fi
done < <(find "$root" -type f -name '*.pkl' ! -name '*.zip' -print0)

if $dry_run; then
  echo "[DRY-RUN] Matches: $processed, would fail: $failed"
else
  echo "Archived: $processed, failed: $failed"
fi
