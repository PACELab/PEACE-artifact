#!/usr/bin/env bash
# testing.sh — test the MPS combination generator with optional downsampling
# Usage:
#   ./testing.sh <N> [SOURCE_FILE]
# Env options:
#   EVEN_ONLY=0|1           # if 1, allowed values = 20 40 60 80   (keeps all-100s special case)
#   STEP=10                 # step for 10..90 values (ignored if EVEN_ONLY=1 or VALUES set)
#   VALUES="10 20 ... 90"   # explicit allowed values (overrides EVEN_ONLY/STEP)
#   SAMPLE_STRIDE=1         # keep every k-th combo after generation
#   SAMPLE_MOD=0            # if >0, enable hash-bucket sampling
#   SAMPLE_BUCKET=0         # keep combos where hash % SAMPLE_MOD == SAMPLE_BUCKET

set -Eeuo pipefail

# --- Args ---
N="${1:-3}"                    # default n=3; override via arg
SOURCE_FILE="${2:-}"           # optionally: path to your script to source

# --- Env opts (downsampling controls) ---
STEP="${STEP:-10}"
EVEN_ONLY="${EVEN_ONLY:-0}"
VALUES="${VALUES:-}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-1}"
SAMPLE_MOD="${SAMPLE_MOD:-0}"
SAMPLE_BUCKET="${SAMPLE_BUCKET:-0}"

# --- Optional: source user's generator if provided ---
if [[ -n "$SOURCE_FILE" ]]; then
  # Expect it defines: generate_mps_combinations and fills global array "combinations"
  # shellcheck disable=SC1090
  source "$SOURCE_FILE"
fi

# --- Build allowed value set (10..90 by default) ---
_build_values() {
  local vals=()
  if [[ -n "$VALUES" ]]; then
    read -r -a vals <<<"$VALUES"
  else
    if (( EVEN_ONLY == 1 )); then
      vals=(20 40 60 80)
    else
      local v=10
      while (( v <= 90 )); do
        vals+=("$v")
        (( v += STEP ))
      done
    fi
  fi
  echo "${vals[*]}"
}

# --- Fallback generator if none was supplied ---
if ! declare -F generate_mps_combinations >/dev/null 2>&1; then
  generate_mps_combinations() {
      local n=$1 total=100
      combinations=()  # global

      # special all-100s (n times)
      local all100=""
      for ((i=1; i<=n; i++)); do all100+="${all100:+ }100"; done
      combinations+=("$all100")

      _gen_combo "" "$n" "$total"
  }

  _gen_combo() {
      local prefix="$1" n="$2" total="$3"
      local percentages
      read -r -a percentages <<< "$(_build_values)"

      if (( n == 1 )); then
          # last token consumes remaining total if valid
          if (( total >= 10 && total <= 100 )); then
              local combo="${prefix:+$prefix }$total"
              combinations+=("$combo")
          fi
          return
      fi

      for pct in "${percentages[@]}"; do
          # require integer token
          [[ "$pct" =~ ^-?[0-9]+$ ]] || continue
          (( pct <= total )) || continue
          (( pct > 0 )) || continue
          _gen_combo "${prefix:+$prefix }$pct" $((n-1)) $((total - pct))
      done
  }
fi

# --- Preconditions ---
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: Bash 4+ required." >&2
  exit 2
fi

# --- Generate ---
generate_mps_combinations "$N"
if ! declare -p combinations >/dev/null 2>&1; then
  echo "ERROR: combinations array not defined by generator." >&2
  exit 3
fi

# --- Post-filters: stride and deterministic hash-bucket sampling ---
if (( SAMPLE_STRIDE > 1 )); then
  filtered=()
  for i in "${!combinations[@]}"; do
    (( i % SAMPLE_STRIDE == 0 )) && filtered+=("${combinations[$i]}")
  done
  combinations=("${filtered[@]}")
fi

if (( SAMPLE_MOD > 0 )); then
  filtered=()
  for c in "${combinations[@]}"; do
    # djb2-like hash
    h=5381
    while IFS= read -r -n1 ch; do
      [[ -z "$ch" ]] && continue
      printf -v ord '%d' "'$ch"
      h=$(( ((h << 5) + h) + ord ))
    done <<< "$c"
    (( h < 0 )) && h=$(( -h ))
    (( h % SAMPLE_MOD == SAMPLE_BUCKET )) && filtered+=("$c")
  done
  combinations=("${filtered[@]}")
fi

# --- Show summary ---
echo "n=$N"
echo "Allowed values: $(_build_values)"
echo "Sampling: STRIDE=$SAMPLE_STRIDE, MOD=$SAMPLE_MOD, BUCKET=$SAMPLE_BUCKET"
echo "Total combinations: ${#combinations[@]}"
for c in "${combinations[@]}"; do
    echo "$c"
done

# --- Tests ---
fail=0

# 1) No trailing whitespace + well-formed tokens
for c in "${combinations[@]}"; do
  if [[ "$c" =~ [[:space:]]$ ]]; then
    echo "FAIL: Trailing whitespace in combo: '$c'"
    fail=1
  fi
  norm="$(echo "$c" | xargs)"  # trims + squeezes spaces
  if [[ "$norm" != "$c" ]]; then
    echo "WARN: Extra internal spacing found, normalized: '$c' -> '$norm'"
  fi
done

# 2) Sum check (allow the special '100 ... 100' with N tokens)
all100=""
for ((i=1; i<=N; i++)); do all100+="${all100:+ }100"; done

for c in "${combinations[@]}"; do
  if [[ "$c" == "$all100" ]]; then
    continue
  fi
  sum=0
  tok_count=0
  bad_token=0
  for tok in $c; do
    (( tok_count++ ))
    if ! [[ "$tok" =~ ^-?[0-9]+$ ]]; then
      echo "FAIL: Non-integer token '$tok' in combo '$c'"
      bad_token=1
      fail=1
      break
    fi
    (( sum += tok ))
  done
  if (( bad_token == 0 )); then
    if (( tok_count != N )); then
      echo "FAIL: Expected $N tokens, found $tok_count in '$c'"
      fail=1
    fi
    if (( sum != 100 )); then
      echo "FAIL: Sum != 100 for combo '$c' (sum=$sum)"
      fail=1
    fi
  fi
done

# 3) Uniqueness
declare -A seen=()
for c in "${combinations[@]}"; do
  key="$(echo "$c" | xargs)"  # normalize spacing
  if [[ -n "${seen[$key]:-}" ]]; then
    echo "FAIL: Duplicate combo detected: '$c'"
    fail=1
  fi
  seen[$key]=1
done

# --- Report ---
if (( fail != 0 )); then
  echo "Some tests FAILED."
  exit 1
fi

echo "All tests passed ✅"

# --- Directory-name form (underscore-joined) ---
echo
echo "Directory-name examples (first 15):"
count=0
for c in "${combinations[@]}"; do
  dir="${c// /_}"
  echo "  $dir"
  (( count++ ))
  (( count >= 15 )) && break
done
